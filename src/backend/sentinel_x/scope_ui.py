"""Plain-language views of saved measurements; never infer feature meanings."""

from html import escape
import re

PHASES = {"answer": "Answer", "thinking": "Thinking", "prompt": "Prompt"}
STATUSES = {
    "complete": "Measured",
    "running": "Measuring…",
    "incomplete": "Partial measurements",
    "failed": "Measurement failed",
    "cancelled": "Measurement cancelled",
}


def visible_token(token, include_formatting=False):
    text = token.get("token", "")
    if include_formatting:
        return True
    # Hide whitespace, control markers and punctuation-only pieces, not ordinary words.
    return bool(
        text.strip()
        and any(c.isalnum() for c in text)
        and not re.fullmatch(r"\s*<\|.*?\|>\s*|\s*</?(?:think|tool_call)>\s*", text)
    )


def chosen_call(records, state):
    requested = state.get("call")
    if requested is not None and 0 <= requested < len(records):
        return requested
    return next(
        (
            i
            for i in range(len(records) - 1, -1, -1)
            if records[i].get("kind") in ("response", "formatting")
        ),
        len(records) - 1,
    )


def link(key, action, value, label):
    return f'<a href="scopeview:{escape(key, quote=True)}:{action}:{value}">{escape(label)}</a>'


def render_scope(key, records, expanded, edited=False, state=None, evidence=False):
    state = state or {}
    statuses = ", ".join(dict.fromkeys(STATUSES.get(r["status"], r["status"]) for r in records))
    result = [
        f'<p><a href="scope:{escape(key, quote=True)}">'
        f"{'▾' if expanded else '▸'} Explanation · Scope · {escape(statuses)}</a></p>"
    ]
    if edited:
        result.append("<p><i>Scope analysis belongs to the original response.</i></p>")
    if not expanded or not records:
        return "".join(result)
    result.append(
        "<p><b>Activity inside the model</b><br>"
        "Explore where internal patterns were active while the model processed text. "
        "This does not prove why it answered this way or whether the answer is correct.</p>"
    )
    if evidence:
        result.append(
            f'<p><a href="details:{escape(key)}">What supports this answer? View evidence and research details</a></p>'
        )
    else:
        result.append(
            "<p>No supporting evidence is linked to this response. Internal activity is not evidence.</p>"
        )
    index = chosen_call(records, state)
    record = records[index]
    if len(records) > 1:
        result.append(
            "<p>Model step: "
            + " · ".join(
                f"<b>{i + 1}. {escape(r.get('kind', 'response'))}</b>"
                if i == index
                else link(key, "call", i, f"{i + 1}. {r.get('kind', 'response')}")
                for i, r in enumerate(records)
            )
            + "</p>"
        )
    if record.get("kind") == "auxiliary":
        result.append(
            "<p>This step is a helper model call, not the displayed answer. "
            "It may supply inputs to GraphSAGE; these measurements do not explain GraphSAGE itself.</p>"
        )
    if record["status"] in ("failed", "cancelled", "incomplete") or record.get("truncated"):
        result.append(
            "<p><b>These measurements are partial or unavailable.</b> "
            "Missing activity must not be read as absence of a pattern.</p>"
        )
    if record["status"] == "running":
        result.append(
            f"<p>Measuring… {record.get('observed_tokens', 0)} text pieces processed. "
            "Saved text activity will appear when this step finishes.</p>"
        )
    phase = state.get("phase", "answer")
    if phase not in PHASES:
        phase = "answer"
    result.append(
        "<p>View: "
        + " · ".join(
            f"<b>{label}</b>" if name == phase else link(key, "phase", name, label)
            for name, label in PHASES.items()
        )
        + "</p>"
    )
    formatting = state.get("formatting", False)
    result.append(
        "<p>"
        + link(
            key,
            "formatting",
            "toggle",
            "Hide formatting tokens" if formatting else "Include formatting tokens",
        )
        + "</p>"
    )
    tokens = record.get("tokens", [])
    available = [
        (i, t)
        for i, t in enumerate(tokens)
        if (
            t.get("phase") == phase
            or formatting
            and phase == "answer"
            and t.get("phase") == "boundary"
        )
        and visible_token(t, formatting)
    ]
    if phase == "prompt":
        result.append(
            "<p>The prompt includes system instructions, conversation history and your request.</p>"
        )
    elif phase == "thinking":
        result.append("<p>This view contains model-emitted thinking, when captured.</p>")
    else:
        result.append(
            "<p>Excerpts from generated text; this may include the structured response format. "
            "Text pieces can be parts of words.</p>"
        )
    if not available:
        result.append(
            "<p>No saved text activity is available in this view. "
            "Try another view or include formatting tokens.</p>"
        )
    else:
        result.append(
            f"<p>{len(available)} saved text pieces in this view. "
            "<b>Explore up to 8 peaks in measured activity:</b> select a highlighted excerpt. "
            "These are activation peaks, not importance or confidence scores.</p>"
        )
        # Choose peaks by activation, with nearby context; keep snippets within this phase/call.
        ranked = sorted(available, key=lambda item: max(item[1].get("values") or [0]), reverse=True)
        selected = state.get("token")
        picks = []
        for candidate in ranked:
            if all(abs(candidate[0] - previous[0]) > 3 for previous in picks):
                picks.append(candidate)
            if len(picks) == 8:
                break
        phase_tokens = [(i, t) for i, t in enumerate(tokens) if t.get("phase") == phase]
        positions = {i: j for j, (i, _) in enumerate(phase_tokens)}
        for i, token in picks:
            center = positions.get(i)
            context = (
                phase_tokens[max(0, center - 3) : center + 4]
                if center is not None
                else [(i, token)]
            )
            parts = []
            for j, t in context:
                text = t.get("token", "")
                if not formatting and (
                    re.fullmatch(r"<\|.*?\|>", text) or t.get("phase") == "boundary"
                ):
                    continue
                parts.append("<b>" + escape(text) + "</b>" if j == i else escape(text))
            result.append(
                f'<p><a href="scopeview:{escape(key)}:token:{i}">…'
                + "".join(parts).replace("\n", " ")
                + "…</a>"
                + (" ✓ Selected" if selected == i else "")
                + "</p>"
            )
        if selected is not None and any(i == selected for i, _ in available):
            token = tokens[selected]
            result.append(
                "<p><b>Selected text piece:</b> " + escape(repr(token["token"])) + "<br>"
                f"{len(token.get('feature_ids', []))} sparse feature entries were saved here. "
                "Feature meanings have not been established for this view. "
                "Open technical details to inspect the measurements.</p>"
            )
    technical = state.get("technical", False)
    result.append(
        "<p>"
        + link(key, "technical", "toggle", ("▾" if technical else "▸") + " Technical details")
        + "</p>"
    )
    if technical:
        from .scope import strongest

        selected = state.get("token")
        filtered = [t for i, t in available if selected is None or i == selected]
        result.append(
            "<p>Largest activation per feature in the selected text piece or current view. "
            "Values are not percentages. Feature IDs have no verified semantic labels here.</p>"
        )
        result.append(
            "<pre>"
            + escape(
                "\n".join(
                    f"{name}: {record.get(name, 'Not recorded')}"
                    for name in (
                        "model",
                        "model_revision",
                        "sae",
                        "sae_revision",
                        "layer",
                        "hook",
                        "quantization",
                        "device",
                        "cpu_offload",
                        "call_id",
                        "status",
                        "truncated",
                        "error",
                    )
                )
            )
            + "</pre>"
        )
        result.append(
            "<table><tr><th>Feature ID</th><th>Activation</th><th>Text piece</th><th>Position</th></tr>"
        )
        for feature in strongest([{**record, "tokens": filtered}]):
            result.append(
                f"<tr><td>{feature['id']}</td><td>{feature['value']:.6g}</td>"
                f"<td>{escape(repr(feature['token']))}</td><td>{feature['position']}</td></tr>"
            )
        result.append(
            "</table><p>Positions count from the start of this model call, including its prompt. "
            "Base-trained SAE applied to a quantized conversational model; interpretation remains experimental.</p>"
        )
    return "".join(result)
