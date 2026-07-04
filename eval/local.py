#!/usr/bin/env python3
import argparse
import json
import os
import re
from datetime import datetime

import ray
from ray.data.llm import vLLMEngineProcessorConfig, build_processor

from eval.scoring import (
    SPLIT_BUCKETS,
    SYSTEM_PROMPT,
    build_judge_payload,
    parse_json_response,
    row_eval_bucket,
    validate_score,
)


MODEL_ID = os.environ.get("MODEL_ID", "google/gemma-4-26B-A4B-it")
MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "16384"))
MAX_OUTPUT_TOKENS = int(os.environ.get("MAX_OUTPUT_TOKENS", "2048"))
PROMPT_TOKEN_SAFETY_MARGIN = int(os.environ.get("PROMPT_TOKEN_SAFETY_MARGIN", "256"))
CHAT_TEMPLATE_TOKEN_OVERHEAD = int(os.environ.get("CHAT_TEMPLATE_TOKEN_OVERHEAD", "512"))
PROMPT_TOKEN_BUDGET = MAX_MODEL_LEN - MAX_OUTPUT_TOKENS - PROMPT_TOKEN_SAFETY_MARGIN

TENSOR_PARALLEL_SIZE = int(os.environ.get("TENSOR_PARALLEL_SIZE", "1"))
VLLM_CONCURRENCY = int(os.environ.get("VLLM_CONCURRENCY", "8"))
VLLM_BATCH_SIZE = int(os.environ.get("VLLM_BATCH_SIZE", "32"))
MAX_CONCURRENT_BATCHES = int(os.environ.get("MAX_CONCURRENT_BATCHES", "4"))
RAW_INPUT_BLOCKS = int(os.environ.get("RAW_INPUT_BLOCKS", str(max(128, VLLM_CONCURRENCY * 16))))
LLM_INPUT_BLOCKS = int(os.environ.get("LLM_INPUT_BLOCKS", str(max(64, VLLM_CONCURRENCY * 8))))
PREVIEW_ROWS = int(os.environ.get("PREVIEW_ROWS", "100"))


def compact(value, limit):
    text = json.dumps(value, ensure_ascii=False, default=str) if not isinstance(value, str) else value
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 20] + "...[TRUNCATED]"


def estimate_tokens(text):
    return max(1, len(text or "") // 2)


def build_messages(row, row_index):
    payload = build_judge_payload(row, row_index)
    user_content = (
        "請評分以下單筆 Agent-STITCH-S 資料，只輸出 JSON。\n\n"
        + json.dumps(payload, ensure_ascii=False, default=str)
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def row_prompt_fits(row):
    messages = build_messages(row, row.get("__eval_row_index", 0))
    prompt_text = "\n".join(m["content"] for m in messages)
    return estimate_tokens(prompt_text) + CHAT_TEMPLATE_TOKEN_OVERHEAD <= PROMPT_TOKEN_BUDGET


def prompt_too_long_row(row):
    out = {k: v for k, v in row.items() if k != "__eval_row_index"}
    out["eval_scores"] = json.dumps(
        {
            "scores": {
                "speech_first": 0,
                "tool_waiting_safety": 0,
                "temporal_causality": 0,
                "incremental_update": 0,
                "stitch_markers": 0,
                "silence_gap": 0,
                "grounding": 0,
                "customer_service_quality": 0,
                "tool_validity": 0,
            },
            "total": 0,
            "score_total": 0,
            "percentage": 0.0,
            "keep": False,
            "bucket": "reject",
            "max_silence_gap_sec": None,
            "timing_estimated": True,
            "temporal_causality_errors": [],
            "grounding_errors": [],
            "critical_errors": ["prompt_too_long"],
            "minor_errors": [],
            "deduction_reasons": ["prompt_too_long"],
            "customer_service_comment": "Judge prompt exceeded local model context budget.",
            "comment": "Judge prompt exceeded local model context budget.",
            "eval_error": "prompt_too_long",
        },
        ensure_ascii=False,
    )
    return out


def add_row_index(row):
    row = dict(row)
    row["__eval_row_index"] = row.get("__eval_row_index", 0)
    return row


def preprocess(row):
    row_index = row.get("__eval_row_index", 0)
    messages = build_messages(row, row_index)
    prompt_text = "\n".join(m["content"] for m in messages)
    prompt_tokens = estimate_tokens(prompt_text) + CHAT_TEMPLATE_TOKEN_OVERHEAD
    max_tokens = max(512, min(MAX_OUTPUT_TOKENS, MAX_MODEL_LEN - prompt_tokens - PROMPT_TOKEN_SAFETY_MARGIN))
    original_row = {k: v for k, v in row.items() if k != "__eval_row_index"}
    return {
        "row_json": json.dumps(original_row, ensure_ascii=False, default=str),
        "fallback_payload_json": json.dumps(build_judge_payload(row, row_index), ensure_ascii=False, default=str),
        "messages": messages,
        "sampling_params": {
            "temperature": 0.0,
            "max_tokens": max_tokens,
        },
    }


def error_score(error, raw_text=None, fallback_payload=None):
    fallback_silence = None
    try:
        if fallback_payload:
            fallback_silence = json.loads(fallback_payload)["deterministic_signals"]["estimated_max_silence_gap_sec"]
    except Exception:
        fallback_silence = None
    return validate_score(
        {
            "scores": {},
            "max_silence_gap_sec": fallback_silence,
            "timing_estimated": True,
            "temporal_causality_errors": [],
            "grounding_errors": [],
            "deduction_reasons": [error],
            "customer_service_comment": "Local judge output could not be parsed.",
            "eval_error": error,
            "raw_judge_output": compact(raw_text or "", 2000),
        },
        fallback_silence,
    )


def normalize_scored_row(row):
    """
    當 should_continue_on_error=True 時，engine 失敗的 row 會帶著
    __inference_error__ 直接繞過 postprocess，沒有 eval_scores。
    這裡把它們還原成原始 row 並給 inference_error 分數，同時把
    helper 欄位拿掉，保持輸出 schema 一致。
    """
    row = dict(row)
    err = str(row.pop("__inference_error__", "") or "")
    if "eval_scores" in row and not err:
        return row
    original_row = json.loads(row.get("row_json") or "{}")
    score = error_score(
        f"inference_error: {err or 'missing eval output'}",
        None,
        row.get("fallback_payload_json"),
    )
    original_row["eval_scores"] = json.dumps(score, ensure_ascii=False)
    return original_row


def postprocess(row):
    original_row = json.loads(row["row_json"])
    raw = row.get("generated_text") or row.get("response") or ""
    try:
        fallback_payload = json.loads(row.get("fallback_payload_json") or "{}")
        fallback_silence = fallback_payload.get("deterministic_signals", {}).get("estimated_max_silence_gap_sec")
        score = validate_score(parse_json_response(raw), fallback_silence)
    except Exception as exc:
        score = error_score(f"judge_json_parse_failed: {exc}", raw, row.get("fallback_payload_json"))
    original_row["eval_scores"] = json.dumps(score, ensure_ascii=False)
    return original_row


def make_run_id():
    now = datetime.now().astimezone()
    return now, now.strftime("%Y%m%d_%H%M%S")


def write_preview(ds, preview_path):
    os.makedirs(os.path.dirname(preview_path), exist_ok=True)
    rows = ds.take(PREVIEW_ROWS) if PREVIEW_ROWS > 0 else []
    with open(preview_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    return len(rows)


def run_pipeline(input_path, output_dir, max_rows=None):
    started_at, run_id = make_run_id()
    print(f"Run timestamp: {started_at.isoformat()} (run_id={run_id})")

    try:
        ray.init(address=os.environ.get("RAY_ADDRESS", "auto"))
    except Exception:
        print("Could not connect to existing Ray cluster. Starting a local Ray instance...")
        ray.init()

    print(f"Reading parquet input: {input_path}")
    raw_ds = ray.data.read_parquet(input_path)
    if max_rows is not None:
        raw_ds = ray.data.from_items(raw_ds.take(max_rows))

    raw_ds = raw_ds.repartition(RAW_INPUT_BLOCKS, shuffle=False)
    indexed_ds = raw_ds.map(add_row_index)
    llm_input_ds = indexed_ds.filter(row_prompt_fits).repartition(LLM_INPUT_BLOCKS, shuffle=False)
    prompt_too_long_ds = indexed_ds.filter(lambda r: not row_prompt_fits(r)).map(prompt_too_long_row)

    print(
        "vLLM eval config: "
        f"model={MODEL_ID}, "
        f"tensor_parallel_size={TENSOR_PARALLEL_SIZE}, "
        f"concurrency={VLLM_CONCURRENCY}, "
        f"batch_size={VLLM_BATCH_SIZE}, "
        f"max_concurrent_batches={MAX_CONCURRENT_BATCHES}, "
        f"max_model_len={MAX_MODEL_LEN}, "
        f"max_output_tokens={MAX_OUTPUT_TOKENS}"
    )

    config = vLLMEngineProcessorConfig(
        model_source=MODEL_ID,
        engine_kwargs={
            "tensor_parallel_size": TENSOR_PARALLEL_SIZE,
            "max_model_len": MAX_MODEL_LEN,
            "enable_chunked_prefill": True,
            "enable_prefix_caching": True,
            "kv_cache_dtype": "fp8",
            "gpu_memory_utilization": 0.90,
            "max_num_seqs": 128,
            "max_num_batched_tokens": 32768,
            "trust_remote_code": True,
        },
        concurrency=(VLLM_CONCURRENCY, VLLM_CONCURRENCY),
        batch_size=VLLM_BATCH_SIZE,
        should_continue_on_error=True,
        max_concurrent_batches=MAX_CONCURRENT_BATCHES,
        experimental={"max_tasks_in_flight_per_actor": 8},
    )

    processor = build_processor(config, preprocess=preprocess, postprocess=postprocess)

    print("Running local Gemma judge inference...")
    scored_ds = processor(llm_input_ds).map(normalize_scored_row).union(prompt_too_long_ds).materialize()

    run_output_dir = os.path.join(output_dir, "runs", run_id)
    scored_output_path = os.path.join(run_output_dir, "scored")
    preview_path = os.path.join(run_output_dir, "preview", "scored.jsonl")
    scored_ds.write_parquet(scored_output_path)
    preview_count = write_preview(scored_ds, preview_path)

    split_counts = {}
    for bucket in SPLIT_BUCKETS:
        split_ds = scored_ds.filter(lambda r, bucket=bucket: row_eval_bucket(r) == bucket).materialize()
        split_output_path = os.path.join(run_output_dir, bucket)
        split_preview_path = os.path.join(run_output_dir, "preview", f"{bucket}.jsonl")
        os.makedirs(split_output_path, exist_ok=True)
        split_count = split_ds.count()
        if split_count:
            split_ds.write_parquet(split_output_path)
        write_preview(split_ds, split_preview_path)
        split_counts[bucket] = split_count

    print(f"Scored dataset written to {scored_output_path}")
    print(f"Preview rows written to {preview_path}: {preview_count}")
    for bucket in SPLIT_BUCKETS:
        print(
            f"{bucket.title()} dataset written to "
            f"{os.path.join(run_output_dir, bucket)} "
            f"(rows: {split_counts[bucket]})"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Local Gemma Agent-STITCH-S eval scoring pipeline.")
    parser.add_argument("--input", required=True, help="Input assembled STITCH-S parquet path/directory.")
    parser.add_argument("--out", required=True, help="Output directory.")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional smoke-test row limit.")
    args = parser.parse_args()
    run_pipeline(args.input, args.out, max_rows=args.max_rows)
