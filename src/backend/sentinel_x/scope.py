"""Qwen-Scope residual measurements, with no inferred semantic feature labels.

Algorithm and hook: pinned SAE model card and config (see docs/SCOPE.md).
"""

from __future__ import annotations

import uuid

LIMITATION = (
    "Base-trained SAE applied to the corresponding quantized conversational model. "
    "Measured Qwen activations, not validated causal explanations or GraphSAGE internals."
)
HOOK = "model.language_model.layers.16:output (resid_post, zero-based)"


def sparse_topk(residual, weight, bias, k=50, batch_size=32):
    """Yield bounded sparse rows. Upstream uses neither centering nor ReLU."""
    import torch

    if residual.ndim != 2 or weight.ndim != 2 or residual.shape[1] != weight.shape[1]:
        raise ValueError("Scope residual/encoder dimensions do not match")
    if bias.shape != (weight.shape[0],) or not 0 < k <= weight.shape[0] or batch_size < 1:
        raise ValueError("Invalid Scope bias, Top-K or batch size")
    for start in range(0, len(residual), batch_size):
        values = (
            residual[start : start + batch_size].to(device=weight.device, dtype=weight.dtype)
            @ weight.T
            + bias
        )
        if not torch.isfinite(values).all():
            raise ValueError("Nonfinite Scope activations")
        selected, indices = values.topk(k, dim=-1)
        yield indices.cpu().tolist(), selected.float().cpu().tolist()


def validate_sae(state, hidden=4096, width=65536):
    shapes = {
        "W_enc": (width, hidden),
        "W_dec": (hidden, width),
        "b_enc": (width,),
        "b_dec": (hidden,),
    }
    if set(state) != set(shapes) or any(tuple(state[k].shape) != v for k, v in shapes.items()):
        raise ValueError("Scope checkpoint tensor dimensions differ from the pinned configuration")


def resolve_hook(model, hidden=4096, count=32):
    """Fail closed on another architecture, rather than guessing a similar layer."""
    config = model.config
    text = config.text_config
    if (
        config.model_type != "qwen3_5"
        or text.hidden_size != hidden
        or text.num_hidden_layers != count
    ):
        raise ValueError("Scope requires the selected pinned Qwen architecture")
    layers = model.model.language_model.layers
    if len(layers) != count or type(layers[16]).__name__ != "Qwen3_5DecoderLayer":
        raise ValueError(
            "Scope residual hook location is incompatible with this Transformers version"
        )
    return layers[16]


class Capture:
    def __init__(self, weight, bias, tokenizer, manifest, kind, limit=8192, k=50):
        self.weight, self.bias, self.tokenizer = weight, bias, tokenizer
        self.limit, self.k = limit, k
        self.pending = []
        self.position = 0
        self.record = {
            "version": 1,
            "call_id": uuid.uuid4().hex,
            "kind": kind,
            "status": "running",
            "model": manifest["qwen"]["repo"],
            "model_revision": manifest["qwen"]["revision"],
            "sae": manifest["scope"]["repo"],
            "sae_revision": manifest["scope"]["revision"],
            "layer": 16,
            "hook": HOOK,
            "top_k": k,
            "interpretation": LIMITATION,
            "tokens": [],
            "truncated": False,
            "observed_tokens": 0,
        }

    def prepare(self, ids, phases):
        if len(ids) != len(phases):
            raise ValueError("Scope token phase alignment failed")
        self.pending = list(zip(ids, phases))

    def hook(self, module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        count = len(self.pending)
        if self.record["status"] == "failed":
            self.position += count
            return
        try:
            if hidden.ndim != 3 or hidden.shape[0] != 1 or hidden.shape[1] != count:
                raise ValueError("Scope hook token alignment failed")
            available = max(0, self.limit - len(self.record["tokens"]))
            take = min(count, available)
            offset = 0
            for indices, values in sparse_topk(
                hidden[0, :take].detach(), self.weight, self.bias, self.k
            ):
                for feature_ids, activations in zip(indices, values):
                    token_id, phase = self.pending[offset]
                    self.record["tokens"].append(
                        {
                            "position": self.position + offset,
                            "token_id": token_id,
                            "token": self.tokenizer.decode([token_id]),
                            "phase": phase,
                            "feature_ids": feature_ids,
                            "values": activations,
                        }
                    )
                    offset += 1
            self.record["truncated"] |= take < count
        except Exception as exc:
            # Keep generation alive but never claim that its measurement succeeded.
            self.record.update(status="failed", error=str(exc))
        self.position += count
        self.record["observed_tokens"] = self.position

    def finish(self, status="complete"):
        if self.record["status"] != "failed":
            self.record["status"] = "incomplete" if self.record["truncated"] else status
            if not self.record["tokens"] and status != "cancelled":
                self.record.update(status="failed", error="No residual activations were captured")
        return self.record


def strongest(records, count=20):
    """Rank distinct features by maximum activation across all measured tokens."""
    best = {}
    for record in records:
        for token in record.get("tokens", []):
            for ident, value in zip(token["feature_ids"], token["values"]):
                if ident not in best or value > best[ident]["value"]:
                    best[ident] = {
                        "id": ident,
                        "value": value,
                        "token": token["token"],
                        "position": token["position"],
                        "phase": token["phase"],
                        "call_id": record["call_id"],
                    }
    return sorted(best.values(), key=lambda x: (-x["value"], x["id"]))[:count]


def explanation_text(records):
    lines = [LIMITATION]
    for r in records:
        lines.append(
            f"Call {r['call_id']} · {r['kind']} · {r['status']} · layer {r['layer']} · {r.get('observed_tokens', 0)} tokens measured"
        )
        if r.get("error"):
            lines.append(r["error"])
        if r.get("truncated"):
            lines.append("Token capture was truncated; this explanation is incomplete.")
    for f in strongest(records):
        lines.append(
            f"Feature {f['id']} · {f['value']:.6g} · token {f['token']!r} · "
            f"{f['phase']} position {f['position']} · call {f['call_id']}"
        )
    return "\n".join(lines)
