# Local skill format

A skill is a directory with `SKILL.md` instructions and a `tools.json` manifest. Import it from the Skills view or put it under the application-data `skills` directory. IDs must be unique. Enable/disable skills in the UI.

```json
{
  "id": "evidence-helper",
  "name": "Evidence helper",
  "tools": [{
    "handler": "retrieval.search",
    "function": {
      "name": "evidence_search",
      "description": "Search local evidence for a factual claim.",
      "parameters": {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
        "additionalProperties": false
      }
    }
  }]
}
```

The bundled example is `src/backend/sentinel_x/assets/skills/graphsage`. Supported handler/function bindings are:

| Handler | Qwen function | Arguments |
| --- | --- | --- |
| `retrieval.search` | `evidence_search` | `query` |
| `graphsage.predict` | `graphsage_predict` | `graph_ids`, optional `checkpoint_id` |

The dotted handler name is internal; function names use the tool-calling-compatible underscore form. New executable capabilities require an explicitly registered backend handler and validation, not a downloaded script. Local skill files never execute Python or shell commands. Evidence search remains available as a core tool even if no optional skills are enabled.

`graphsage.predict` returns one record per cascade: `graph_id`, `cascade_id`, `viral_probability`, `viral`, `decision_cutoff`, `model_id`, `observation_minutes`, `size_threshold`, `estimated`, warnings, and provenance. No selected checkpoint, incomplete metadata, or an unaccepted estimated graph produces an explicit tool error. Qwen can only request prediction for graphs attached to the active conversation or its previous turns. A bundle returns individual cascade results, not an invented aggregate probability.

Tool requests/results are visible in job activity and persisted with the conversation. Five model/tool iterations and at most eight tool calls per iteration bound execution. Failed tools return errors to Qwen so it can explain what the user needs to resolve.
