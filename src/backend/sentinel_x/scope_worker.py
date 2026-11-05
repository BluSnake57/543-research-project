"""Isolated, offline Transformers inference. Only sparse measurements leave this process."""

from __future__ import annotations

import json
import re
import uuid
from pathlib import Path

from .config import manifest, model_profile
from .scope import Capture, resolve_hook, validate_sae


def placement(device, offload, profile):
    if device == "cpu":
        return "cpu", "cpu"
    if offload not in ("none", "sae", "modules"):
        raise ValueError("Unknown CPU offload policy")
    if offload != "modules":
        return "cuda", "cpu" if offload == "sae" else "cuda"
    return {
        "model.visual": "cpu",
        "model.language_model.embed_tokens": "cpu",
        "lm_head": "cpu",
        "model.language_model.norm": "cuda",
        **{f"model.language_model.layers.{i}": "cuda" for i in range(profile["layers"])},
    }, "cpu"


def preflight(device, context=8192, model="9b", offload="none"):
    import psutil
    import torch

    model_profile(model)
    if device not in ("cuda", "cpu"):
        raise ValueError("Choose CUDA or the explicit slower CPU fallback in Models")
    # Admission estimates are intentionally distinct from measured peak-memory claims.
    cpu_required = (12 if model == "9b" else 6) + context / 8192
    gpu_required = (9 if model == "9b" else 3.5) + context / 8192
    if offload == "sae":
        gpu_required -= 1 if model == "9b" else 0.25
    if offload == "modules":
        gpu_required = (4.3 if model == "9b" else 2.5) + context / 8192
    if device == "cuda":
        if not torch.cuda.is_available():
            raise ValueError(
                "CUDA is unavailable. Repair the NVIDIA driver or explicitly select CPU."
            )
        available, _ = torch.cuda.mem_get_info()
        if gpu_required > min(5.5, available / 1024**3):
            raise ValueError(
                "Insufficient CUDA memory within the 6 GiB limit. Use CPU offloading, "
                "reduce context, or select the 2B fallback."
            )
        if not torch.cuda.is_bf16_supported():
            raise ValueError("CUDA BF16 support is required; select CPU on this device.")
    if cpu_required > min(15, psutil.virtual_memory().available / 1024**3):
        raise ValueError(
            "Insufficient CPU memory within the 16 GiB limit; select 2B or reduce context."
        )
    return torch.bfloat16 if device == "cuda" else torch.float32


def parse_output(text, tools):
    calls = []
    pattern = r"<tool_call>\s*(.*?)\s*</tool_call>"
    for payload in re.findall(pattern, text, re.S):
        functions = {t["function"]["name"]: t["function"] for t in tools}
        match = re.fullmatch(r"<function=([\w-]+)>\s*(.*?)\s*</function>", payload, re.S)
        if not match or match[1] not in functions:
            raise ValueError("Model returned an invalid tool call")
        schema = functions[match[1]].get("parameters", {})
        properties = schema.get("properties", {})
        arguments = {}
        param_pattern = r"<parameter=([\w-]+)>\s*(.*?)\s*</parameter>"
        if re.sub(param_pattern, "", match[2], flags=re.S).strip():
            raise ValueError("Malformed tool parameters")
        for key, raw in re.findall(param_pattern, match[2], re.S):
            if key not in properties or key in arguments:
                raise ValueError("Unknown or repeated tool parameter")
            arguments[key] = raw if properties[key].get("type") == "string" else json.loads(raw)
        if set(schema.get("required", [])) - arguments.keys():
            raise ValueError("Missing required tool parameter")
        value = {"name": match[1], "arguments": arguments}
        calls.append(
            {
                "id": "call_" + uuid.uuid4().hex,
                "type": "function",
                "function": {"name": value["name"], "arguments": json.dumps(value["arguments"])},
            }
        )
    content = re.sub(pattern, "", text, flags=re.S).strip()
    if "<tool_call>" in content or "</tool_call>" in content:
        raise ValueError("Model returned an unfinished tool call")
    result = {"role": "assistant", "content": content}
    if calls:
        result["tool_calls"] = calls
    return result


class Engine:
    def __init__(self, config):
        import torch
        from transformers import (
            AutoConfig,
            AutoTokenizer,
            Qwen3_5ForConditionalGeneration,
            BitsAndBytesConfig,
        )
        from .downloads import Models
        from .state import Store

        self.config = config
        self.device = config["device"]
        profile = model_profile(config.get("model", "9b"))
        offload = config.get("offload", "none") if self.device == "cuda" else "none"
        dtype = preflight(self.device, config["context"], config.get("model", "9b"), offload)
        device_map, sae_device = placement(self.device, offload, profile)
        if self.device == "cuda":
            total = torch.cuda.get_device_properties(0).total_memory
            torch.cuda.set_per_process_memory_fraction(min(1.0, 5.5 * 1024**3 / total))
        root = Path(config["root"])
        self.catalog = manifest()
        models = Models(Store(root), self.catalog)
        for key in profile["keys"]:
            models.verify(key)
        qwen_key, scope_key = profile["keys"]
        self.catalog = {"qwen": self.catalog[qwen_key], "scope": self.catalog[scope_key]}
        folder = root / "models"
        qwen_folder, scope_folder = folder / qwen_key, folder / scope_key
        sae_config = json.loads((scope_folder / "config.json").read_text())
        expected = {
            "d_model": profile["hidden"],
            "d_sae": profile["width"],
            "k": 50,
            "hook_point": "resid_post",
            "base_model": self.catalog["qwen"]["repo"] + "-Base",
            "num_layers": profile["layers"],
        }
        if any(sae_config.get(k) != v for k, v in expected.items()):
            raise ValueError("Scope configuration does not match the pinned residual SAE")
        model_config = AutoConfig.from_pretrained(
            qwen_folder, local_files_only=True, trust_remote_code=False
        )
        if (
            model_config.model_type != "qwen3_5"
            or model_config.text_config.hidden_size != profile["hidden"]
            or model_config.text_config.num_hidden_layers != profile["layers"]
        ):
            raise ValueError("Selected Qwen architecture does not match its Scope SAE")
        state = torch.load(
            scope_folder / "layer16.sae.pt", map_location="cpu", weights_only=True, mmap=True
        )
        validate_sae(state, profile["hidden"], profile["width"])
        # Encoder remains float32, matching the published extraction algorithm.
        self.weight = state["W_enc"].to(device=sae_device, dtype=torch.float32)
        self.bias = state["b_enc"].to(device=sae_device, dtype=torch.float32)
        del state
        self.tokenizer = AutoTokenizer.from_pretrained(
            qwen_folder, local_files_only=True, trust_remote_code=False
        )
        self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
            qwen_folder,
            local_files_only=True,
            trust_remote_code=False,
            dtype=dtype,
            device_map=device_map,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=dtype,
                llm_int8_enable_fp32_cpu_offload=offload == "modules",
            ),
            attn_implementation="sdpa",
        ).eval()
        self.layer = resolve_hook(self.model, profile["hidden"], profile["layers"])

    def generate(self, request, send, cancelled):
        import torch

        thinking = request["thinking"]
        # The pinned template expects argument mappings, while the app stores JSON strings.
        messages = json.loads(json.dumps(request["messages"]))
        for message in messages:
            for call in message.get("tool_calls", []):
                arguments = call["function"]["arguments"]
                if isinstance(arguments, str):
                    call["function"]["arguments"] = json.loads(arguments)
        template = self.tokenizer.apply_chat_template(
            messages,
            tools=request["tools"] or None,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=thinking,
        )
        ids = self.tokenizer(template, add_special_tokens=False, return_tensors="pt")["input_ids"]
        prompt_ids = ids[0].tolist()
        if len(prompt_ids) + request["max_tokens"] > self.config["context"]:
            raise ValueError(
                "Request exceeds context; reduce attachments or increase context in Models"
            )
        capture = Capture(self.weight, self.bias, self.tokenizer, self.catalog, request["kind"])
        capture.record.update(
            quantization="bitsandbytes NF4 double quantization",
            device=self.device,
            cpu_offload=self.config.get("offload", "none"),
        )
        send({"event": "scope", "record": capture.record})
        hook = self.layer.register_forward_hook(capture.hook)
        cache = None
        generated = []
        answer_ids, thinking_ids = [], []
        phase = "thinking" if thinking else "answer"
        in_tool = False
        prompt_length = len(prompt_ids)
        end_ids = self.model.generation_config.eos_token_id
        end_ids = [end_ids] if isinstance(end_ids, int) else end_ids or []
        close_think = self.tokenizer.convert_tokens_to_ids("</think>")
        open_think = self.tokenizer.convert_tokens_to_ids("<think>")
        send({"event": "stage", "stage": "Processing prompt"})
        emitted = {"thinking": "", "answer": ""}
        try:
            with torch.inference_mode():
                for start in range(0, prompt_length, 256):
                    if cancelled.is_set():
                        send({"event": "scope", "record": capture.finish("cancelled")})
                        return
                    chunk = prompt_ids[start : start + 256]
                    capture.prepare(chunk, ["prompt"] * len(chunk))
                    result = self.model(
                        input_ids=torch.tensor([chunk], device=self.device),
                        past_key_values=cache,
                        use_cache=True,
                        logits_to_keep=1,
                    )
                    cache = result.past_key_values
                    send(
                        {
                            "event": "prompt_progress",
                            "processed": start + len(chunk),
                            "total": prompt_length,
                        }
                    )
                for step in range(request["max_tokens"] + 1):
                    if cancelled.is_set():
                        capture.finish("cancelled")
                        send({"event": "scope", "record": capture.record})
                        return
                    if step:
                        result = self.model(
                            input_ids=ids.to(self.device),
                            past_key_values=cache,
                            use_cache=True,
                            logits_to_keep=1,
                        )
                        cache = result.past_key_values
                    # Process the final emitted token once in the same cached execution so
                    # every displayed token has its own residual, not its predecessor's.
                    if step == request["max_tokens"] or (generated and generated[-1] in end_ids):
                        break
                    token = int(result.logits[0, -1].argmax().item())
                    if phase == "thinking" and len(thinking_ids) >= self.config["thinking_budget"]:
                        token = close_think
                        capture.record["thinking_budget_reached"] = True
                    generated.append(token)
                    token_phase = phase
                    if phase == "answer":
                        piece = self.tokenizer.decode([token])
                        in_tool = in_tool or piece == "<tool_call>"
                        if in_tool:
                            token_phase = "tool_call"
                        if piece == "</tool_call>":
                            in_tool = False
                    if token == close_think:
                        phase = "answer"
                        token_phase = "boundary"
                    elif token == open_think:
                        if not thinking:
                            raise ValueError("Model emitted thinking with Thinking Off")
                        token_phase = "boundary"
                    elif token not in end_ids:
                        target = thinking_ids if phase == "thinking" else answer_ids
                        target.append(token)
                        decoded = self.tokenizer.decode(target, skip_special_tokens=False)
                        # Hold incomplete UTF-8 tokens until the decoder has a full character.
                        if not decoded.endswith("\ufffd") and decoded.startswith(emitted[phase]):
                            delta = decoded[len(emitted[phase]) :]
                            if delta:
                                send(
                                    {
                                        "event": "piece",
                                        "text": delta,
                                        "reasoning": phase == "thinking",
                                    }
                                )
                                emitted[phase] = decoded
                    else:
                        token_phase = "boundary"
                    ids = torch.tensor([[token]])
                    capture.prepare([token], [token_phase])
                    if step % 32 == 0:
                        send({"event": "scope_progress", "tokens": capture.position})
                capture.record.update(
                    prompt_tokens=prompt_length,
                    generated_tokens=len(generated),
                    output_truncated=bool(generated and generated[-1] not in end_ids),
                )
                if capture.record["output_truncated"]:
                    capture.record["truncated"] = True
                record = capture.finish()
                send({"event": "scope", "record": record})
                answer = parse_output(
                    self.tokenizer.decode(answer_ids, skip_special_tokens=False), request["tools"]
                )
                send({"event": "result", "message": answer})
        except Exception as exc:
            capture.record.update(status="failed", error=str(exc))
            send({"event": "scope", "record": capture.record})
            raise
        finally:
            hook.remove()
            del cache


def worker_main(connection, cancelled, config):
    """Spawn target, usable by source and frozen applications."""
    try:
        engine = Engine(config)
        connection.send({"event": "ready"})
        while True:
            request = connection.recv()
            if request.get("command") == "shutdown":
                return
            try:
                engine.generate(request, connection.send, cancelled)
            except Exception as exc:
                connection.send({"event": "error", "message": str(exc)})
    except (EOFError, BrokenPipeError):
        pass
    except Exception as exc:
        connection.send({"event": "error", "message": str(exc)})
    finally:
        connection.close()
