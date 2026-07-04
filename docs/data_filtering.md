# Data Filtering & Drop Logic

This document describes how the generation and evaluation stages decide which
rows to **keep**, **drop**, or **salvage**. It complements the "Behavior
Summary" section of the top-level `README.md`, which lists the pre-existing
input-level drops (`missing_tool_call_tool_result`, `missing_reference_answer`,
`prompt_too_long`, ...). Everything below covers the filters that act on the
model's *patch output* and on *engine failures*.

A row can leave the pipeline for three reasons: the model itself declares the
turn unusable (`drop_reason` in the patch), a deterministic post-check rejects
the assembled patch, or inference for the row failed entirely. Each is a
distinct `drop_reason` so drop rates can be attributed.

## 1. Model-declared drops (`drop_reason` in the patch)

The prompt asks the model to refuse a turn when it cannot form a coherent
"speak-while-acting" trajectory. When it does, it emits a `drop_reason` and
leaves the speech/private-state fields empty, and the assembler forwards that
reason unchanged.

`irrelevant_or_hallucinated_tool_step` is emitted in two situations:

- **Ungrounded tool step** — a tool call references an entity, URL, location,
  account, product, or date that does not appear in the current user message,
  the conversation `context`, the metadata, or any earlier tool result. Making
  such a call "sound natural" would teach the model to hallucinate, so the turn
  is dropped instead.
- **Closing-acknowledgement-only turn** — the current user message is only a
  sign-off ("謝謝", "太棒了", "thanks") with no new request, while the
  `tool_steps` belong to work already completed in `context`. There is nothing
  new to act on, so the turn cannot become a speak-while-acting trajectory and
  is dropped.

Grounding is only checkable because each turn now carries the earlier turns of
the conversation — see [Multi-turn context](#multi-turn-context).

## 2. Deterministic post-checks on the assembled patch

These run in `assemble_agent_stitch_s` after the model returns, before a row is
accepted.

### 2.1 Tolerant patch parsing (`parse_patch_json`) — salvage, not drop

Local models frequently wrap JSON in Markdown fences or emit a trailing comma.
Rather than counting those as parse-failure drops, the patch text is parsed
leniently:

1. Strip Markdown code fences, including an unclosed opening fence.
2. Try a direct `json.loads`.
3. On failure, extract the outermost `{ ... }` span and retry, once as-is and
   once with trailing commas removed.
4. Only if every attempt fails does the row become a JSON-parse failure.

This recovers otherwise-valid patches and keeps the parse-failure drop reserved
for genuinely broken output.

### 2.2 Rewriter meta-leakage (`rewriter_meta_leakage`) — drop

The private-state chunks must read as the agent's own live reasoning, not as a
description of the rewriting task. A patch is dropped when any speech or
private-state field mentions the rewriting setup — phrases such as
`tool_steps`, `reference_answer`, `maintain fidelity`, `as instructed`,
`rewriter`, `patch json`, or `the input json`.

The check (`patch_meta_leakage`) scans `first_say`, `final_say`,
`final_private_state`, and every step's `pre_tool_private_state` and
`post_tool_say`. A match sets `drop_reason = "rewriter_meta_leakage"` and the
row is not assembled. This removes samples that would otherwise train the model
to talk about "the provided data" instead of staying in character.

## 3. Inference-failure rows (`inference_error`)

When the engine runs with `should_continue_on_error=True`, a row whose
inference fails is passed through carrying an `__inference_error__` marker and
**bypasses postprocess entirely** — so it has no `raw_patch` (generate) and no
`eval_scores` (eval). Both stages normalize these rows instead of letting the
missing fields crash downstream steps or silently disappear.

- **Generate** (`process_patch_row`): a row with an inference-error marker or no
  `raw_patch` is emitted as `status = "dropped"`,
  `drop_reason = "inference_error"`, with the original error text preserved.
- **Eval** (`normalize_scored_row`): a row without `eval_scores` is restored to
  its original schema, given an `inference_error` `error_score`, and has the
  internal helper columns stripped so the `scored` output keeps a uniform
  schema. Rows that already carry `eval_scores` pass through untouched.

The result is that engine failures are accounted for as an explicit
`inference_error` drop rather than being lost.

## Multi-turn context

Grounding and the closing-acknowledgement drop both depend on knowing what
happened earlier in the conversation. Canonicalization now separates the two:

- `user_request` is always the **last** user message — the turn currently being
  rewritten.
- Every earlier user message (and its assistant answer, when present) is moved
  into a `context` list, oldest first, with long entries compacted.

`context` is threaded into the prompt payload so the model can distinguish a
brand-new request from an acknowledgement of already-finished work, and can
treat prior turns as a valid grounding source for tool calls.

## Related robustness changes

Two changes are not filters themselves but reduce avoidable drops and
low-quality rows:

- **Output budget scaled by translation length** — the model echoes the full
  user message back as `translated_user`, so the output-token budget is grown
  in proportion to the user message length. Long messages no longer truncate
  the JSON mid-string (which would otherwise surface as a parse-failure drop).
- **Sampling temperature and SAY variation** — a low temperature collapsed every
  bridge `SAY` into the same stock sentence, which the eval judge penalizes as
  template-like speech. The default temperature is raised (overridable via
  `GENERATE_TEMPERATURE`), and the prompt forbids repeating an opening across a
  turn's `SAY` chunks.
