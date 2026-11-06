"""Shared batch/interactive semantic protocol and cache provenance."""

import hashlib
import math

SEMANTIC_DEFINITIONS = {
    "urgency": ("Pressure to react, share, click, believe, or act immediately."),
    "sensationalism": (
        "Exaggerated, shocking, dramatic, provocative, or attention-grabbing presentation."
    ),
    "emotional_language": (
        "Emotionally charged wording intended to trigger "
        "fear, anger, outrage, excitement, sympathy, "
        "or another strong emotional reaction."
    ),
    "authority_framing": (
        "Reliance on experts, officials, institutions, "
        "celebrities, scientists, governments, or other "
        "authority figures to strengthen the message."
    ),
    "conspiracy_framing": (
        "Suggestions of secrecy, suppression, cover-ups, "
        "hidden actors, coordinated deception, or "
        "deliberately concealed information."
    ),
}


FEATURES = tuple(SEMANTIC_DEFINITIONS)
PROTOCOL_VERSION = 2


def build_prompt(source_text, feature_names=FEATURES):
    if not feature_names or any(n not in FEATURES for n in feature_names):
        raise ValueError("Unknown or empty semantic feature schema")
    requested = "\n".join(f"- {name}" for name in feature_names)

    template = ",\n".join(f'  "{name}": 0.0' for name in feature_names)

    definitions = "\n\n".join(f"{name}:\n{SEMANTIC_DEFINITIONS[name]}" for name in feature_names)

    prompt = f"""
    Analyze the social-media post below.

    Return ONLY a JSON object containing exactly
    these numerical semantic features:

    {requested}

    Required JSON format:
    {{
    {template}
    }}

    Every value must be between 0.0 and 1.0.

    Feature definitions:

    {definitions}

    Do not determine whether the post is true or false.
    Do not retrieve evidence.
    Do not explain the scores.
    Return only the requested JSON object.

    POST:
    {source_text}
    """.strip()
    return prompt


def validate_features(result, feature_names=FEATURES):
    if not isinstance(result, dict):
        raise ValueError("Qwen returned an invalid semantic feature object")
    output = {}
    for name in feature_names:
        if name not in result:
            raise ValueError(f"Qwen omitted required feature: {name}")
        value = result[name]
        if isinstance(value, bool):
            raise ValueError(f"Qwen returned a nonnumeric value for {name}")
        try:
            value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Qwen returned a nonnumeric value for {name}") from exc
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"Qwen returned {name} outside [0,1]")
        output[name] = value
    return output


def protocol(store, feature_names, thinking=True):
    from .config import model_profile, manifest

    profile = model_profile(store.setting("scope_model", "9b"))
    return {
        "version": PROTOCOL_VERSION,
        "features": list(feature_names),
        "prompt_sha256": hashlib.sha256(build_prompt("", feature_names).encode()).hexdigest(),
        "model_profile": store.setting("scope_model", "9b"),
        "thinking": thinking,
        "thinking_budget": store.setting("thinking_budget", 256),
        "context": store.setting("context", 8192),
        "quantization": "bitsandbytes NF4 double quantization",
        "artifacts": {
            key: {field: manifest()[key][field] for field in ("repo", "revision")}
            for key in profile["keys"]
        },
    }


def cache_compatible(record, text, expected):
    return (
        record.get("protocol") == expected
        and record.get("source_sha256") == hashlib.sha256(text.encode()).hexdigest()
    )


def feature_cache_protocol(path):
    import json
    from pathlib import Path

    records = json.loads(Path(path).read_text())
    protocols = {json.dumps(r.get("protocol"), sort_keys=True) for r in records.values()}
    if len(protocols) != 1:
        raise ValueError("Mixed feature protocols; generate one consistent cache")
    return json.loads(next(iter(protocols)))
