# Agent-STITCH-S Pipeline

This directory contains the generation and evaluation pipeline for Agent-STITCH-S synthetic data.

## Layout

```text
pipeline.py          Python entrypoint for generate, eval, or all
run_pipeline.sh      Slurm entrypoint for generate, eval, or all
generate/            Generation code
eval/                Evaluation code
prompts/             Prompt files
scripts/             Compatibility Slurm wrappers
```

`__pycache__/` is Python cache output and is not part of the source layout.

## Data Flow

1. Generate reads raw agent conversations.
2. Generate canonicalizes tool calls and tool results.
3. Long tool observations are compacted before entering the prompt; full tool calls/results are kept for final assembly.
4. A model writes only STITCH-S patch fields: spoken SAY chunks and private state.
5. The deterministic assembler restores original tool calls/results and writes assembled SFT rows.
6. Eval reads assembled SFT rows and scores each row.
7. Eval writes scored parquet rows with an `eval_scores` JSON string and split views for `keep` and `review` rows.

## Behavior Summary

Generation filters out rows that cannot produce useful STITCH-S data:

```text
missing_tool_call_tool_result
missing_reference_answer
missing_tool_call_tool_result+missing_reference_answer
```

Local generation also estimates prompt length before inference. Rows over the local model context budget are written to `failed` with `drop_reason=prompt_too_long`. API generation does not do this preflight; API failures are written as `api_generation_failed`.

Eval keeps all scored rows in `scored` and additionally writes filtered parquet views for `keep` and `review`. `reject` rows remain available in `scored`.

Filtering that acts on the model's patch output and on engine failures
(`irrelevant_or_hallucinated_tool_step`, `rewriter_meta_leakage`,
`inference_error`, tolerant patch parsing, and multi-turn context handling) is
documented in [docs/data_filtering.md](docs/data_filtering.md).

## Python Usage

Generate only:

```bash
python3 pipeline.py generate \
  --backend local \
  --data voidful/agent-sft \
  --out ./out_full
```

Generate selected ids only:

```bash
python3 pipeline.py generate \
  --backend local \
  --data voidful/agent-sft \
  --out ./out_full \
  --only-ids ./ids.txt
```

Eval only:

```bash
python3 pipeline.py eval \
  --backend local \
  --input ./out_full/runs/<run_id>/success \
  --out ./out_eval
```

Generate then eval:

```bash
python3 pipeline.py all \
  --generate-backend local \
  --backend local \
  --data voidful/agent-sft \
  --out ./out_full \
  --eval-out ./out_eval
```

`all` evaluates the latest generated `runs/*/success` directory under `--out` unless `--eval-input` is provided.

## API Backend

Generation via API:

```bash
python3 pipeline.py generate \
  --backend api \
  --data voidful/agent-sft \
  --out ./out_full_api \
  --api-key "$API_KEY" \
  --model google/gemma-4-26B-A4B-it \
  --concurrency 4 \
  --max-rows 100
```

Eval via API:

```bash
python3 pipeline.py eval \
  --backend api \
  --input ./out_full_api/runs/<run_id>/success \
  --out ./out_eval_api \
  --api-key "$API_KEY" \
  --model google/gemma-4-26B-A4B-it \
  --concurrency 4
```

For OpenAI-compatible endpoints, pass `--base-url` or set `API_BASE_URL`.

## Slurm Usage

Local generate:

```bash
sbatch --export=ALL,HF_TOKEN,MODE=generate run_pipeline.sh
```

Local eval:

```bash
sbatch --export=ALL,HF_TOKEN,MODE=eval,EVAL_INPUT=/path/to/success run_pipeline.sh
```

Generate then eval:

```bash
sbatch --export=ALL,HF_TOKEN,MODE=all run_pipeline.sh
```

API-only generate:

```bash
sbatch --export=ALL,MODE=generate,GENERATE_BACKEND=api,API_KEY=... run_pipeline.sh
```

API-only eval:

```bash
sbatch --export=ALL,MODE=eval,EVAL_BACKEND=api,API_KEY=...,EVAL_INPUT=/path/to/success run_pipeline.sh
```

API generate and API eval:

```bash
sbatch --export=ALL,MODE=all,GENERATE_BACKEND=api,EVAL_BACKEND=api,API_KEY=... run_pipeline.sh
```

## Common Environment Variables

```text
MODE                 generate, eval, or all
GENERATE_BACKEND     local or api
EVAL_BACKEND         local or api
DATA                 generation input, default voidful/agent-sft
OUT                  generation output directory, default ./out_full
ONLY_IDS             optional id filter file for generation
EVAL_INPUT           eval input success parquet directory
EVAL_OUT             eval output directory, default ./out_eval_scores_gemma
HF_TOKEN             required for local backend when loading gated HF models/data
API_KEY              API key for API backend. Multiple keys can be comma-separated.
API_BASE_URL         optional OpenAI-compatible endpoint
GENERATE_MODEL       API generation model
EVAL_MODEL           API eval model
API_CONCURRENCY      API parallel request count
API_TIMEOUT          API request timeout seconds
API_RETRIES          API retry count per row
MAX_ROWS             smoke-test row limit for API generation or eval
```

Local/Ray tuning variables:

```text
MODEL_ID
MAX_MODEL_LEN
MAX_OUTPUT_TOKENS
GENERATE_MAX_OUTPUT_TOKENS
EVAL_MAX_OUTPUT_TOKENS
MIN_OUTPUT_TOKENS
PROMPT_TOKEN_SAFETY_MARGIN
CHAT_TEMPLATE_TOKEN_OVERHEAD
TENSOR_PARALLEL_SIZE
VLLM_CONCURRENCY
VLLM_BATCH_SIZE
GENERATE_VLLM_BATCH_SIZE
EVAL_VLLM_BATCH_SIZE
MAX_CONCURRENT_BATCHES
RAW_INPUT_BLOCKS
LLM_INPUT_BLOCKS
PREVIEW_ROWS
GPU_PREFLIGHT
JOB_TMP_ROOT
TMPDIR
RAY_TMPDIR
```

For multiple API keys:

```bash
export API_KEY="key1,key2,key3"
```

API requests and retries are assigned round-robin across the keys. Keys are not printed in logs.

## Outputs

Generate writes:

```text
OUT/runs/<run_id>/success
OUT/runs/<run_id>/failed
OUT/runs/<run_id>/preview/success.jsonl
OUT/runs/<run_id>/preview/failed.jsonl
```

Eval writes:

```text
EVAL_OUT/runs/<run_id>/scored
EVAL_OUT/runs/<run_id>/keep
EVAL_OUT/runs/<run_id>/review
EVAL_OUT/runs/<run_id>/preview/scored.jsonl
EVAL_OUT/runs/<run_id>/preview/keep.jsonl
EVAL_OUT/runs/<run_id>/preview/review.jsonl
```
