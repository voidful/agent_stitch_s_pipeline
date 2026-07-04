# -*- coding: utf-8 -*-
import csv
import json
import os
import re
from datetime import datetime
from pathlib import Path
import ray
import sitecustomize  # noqa: F401
from ray.data.llm import vLLMEngineProcessorConfig, build_processor

# ==========================================
# 1. 系統 Prompt 與推論參數設定
# ==========================================

MODEL_ID = os.environ.get("MODEL_ID", "google/gemma-4-26B-A4B-it")
MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "16384"))
MAX_OUTPUT_TOKENS = int(os.environ.get("MAX_OUTPUT_TOKENS", "4096"))
MIN_OUTPUT_TOKENS = int(os.environ.get("MIN_OUTPUT_TOKENS", "512"))
PROMPT_TOKEN_SAFETY_MARGIN = int(os.environ.get("PROMPT_TOKEN_SAFETY_MARGIN", "256"))
CHAT_TEMPLATE_TOKEN_OVERHEAD = int(os.environ.get("CHAT_TEMPLATE_TOKEN_OVERHEAD", "512"))
PROMPT_TOKEN_BUDGET = MAX_MODEL_LEN - MAX_OUTPUT_TOKENS - PROMPT_TOKEN_SAFETY_MARGIN
if PROMPT_TOKEN_BUDGET <= 0:
    raise ValueError(
        "Invalid token budget: MAX_MODEL_LEN must be larger than "
        "MAX_OUTPUT_TOKENS + PROMPT_TOKEN_SAFETY_MARGIN. "
        f"Got MAX_MODEL_LEN={MAX_MODEL_LEN}, "
        f"MAX_OUTPUT_TOKENS={MAX_OUTPUT_TOKENS}, "
        f"PROMPT_TOKEN_SAFETY_MARGIN={PROMPT_TOKEN_SAFETY_MARGIN}."
    )
PREVIEW_ROWS = int(os.environ.get("PREVIEW_ROWS", "100"))
TENSOR_PARALLEL_SIZE = int(os.environ.get("TENSOR_PARALLEL_SIZE", "1"))
VLLM_CONCURRENCY = int(os.environ.get("VLLM_CONCURRENCY", "8"))
VLLM_BATCH_SIZE = int(os.environ.get("VLLM_BATCH_SIZE", "64"))
MAX_CONCURRENT_BATCHES = int(os.environ.get("MAX_CONCURRENT_BATCHES", "8"))
RAW_INPUT_BLOCKS = int(os.environ.get("RAW_INPUT_BLOCKS", str(max(128, VLLM_CONCURRENCY * 16))))
LLM_INPUT_BLOCKS = int(os.environ.get("LLM_INPUT_BLOCKS", str(max(64, VLLM_CONCURRENCY * 8))))

def load_prompt_file(filename):
    prompt_path = Path(__file__).resolve().parent.parent / "prompts" / filename
    with prompt_path.open("r", encoding="utf-8") as f:
        return f.read().strip()


SYSTEM_PROMPT = load_prompt_file("generate_stitch_s.txt")

# ==========================================
# 2. 資料前處理與壓縮 (Compactor & Canonicalizer)
# ==========================================

def compact_observation(text, max_chars=2400):
    """
    針對過長的工具觀測結果進行截斷，保留頭部與尾部的關鍵資訊。
    """
    if not isinstance(text, str):
        text = str(text)
    if len(text) <= max_chars:
        return text
    
    # 頭部保留 1400 字元，尾部保留 800 字元
    head = text[:1400]
    tail = text[-800:]
    return head + "\n...[TRUNCATED]...\n" + tail


def parse_json_maybe(value, fallback=None):
    if fallback is None:
        fallback = value
    if isinstance(value, str):
        try:
            return json.loads(value) if value else fallback
        except Exception:
            return fallback
    return value if value is not None else fallback


def dumps_json_field(value, empty_value=None):
    if value is None:
        value = [] if empty_value is None else empty_value
    return json.dumps(value, ensure_ascii=False)


def loads_json_field(value, fallback=None):
    if fallback is None:
        fallback = []
    if isinstance(value, str):
        try:
            return json.loads(value) if value else fallback
        except Exception:
            return fallback
    return value if value is not None else fallback


def normalize_text_content(content):
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False)


def strip_json_fence(text):
    if not isinstance(text, str):
        return text
    stripped = text.strip()
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return stripped


def parse_patch_json(raw_text):
    """
    寬容地解析模型輸出的 patch JSON:
    1. 去掉 markdown code fence（含沒有閉合的 fence）。
    2. 直接 json.loads。
    3. 失敗時抽出最外層 {...}，並嘗試修掉 trailing comma。
    """
    text = strip_json_fence(raw_text)
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text)
    try:
        return json.loads(text)
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        candidate = text[start:end + 1]
        for fixer in (lambda t: t, lambda t: re.sub(r",\s*([}\]])", r"\1", t)):
            try:
                return json.loads(fixer(candidate))
            except Exception:
                continue
    # 讓呼叫端拿到原始錯誤訊息。
    return json.loads(text)


def clean_say_text(text):
    if text is None:
        return ""
    text = str(text).strip()
    text = re.sub(r"</?SAY>", "", text).strip()
    return text


# Rewriter 後設語言：private_state 應該是 agent 本人的思考，
# 出現這些詞代表模型在討論「被給定的資料」而不是入戲思考，直接丟棄。
META_LEAKAGE_RE = re.compile(
    r"tool_steps|reference_answer|maintain\s+fidelity|as\s+instructed|rewriter|patch\s+json|the\s+input\s+json",
    re.IGNORECASE,
)


def patch_meta_leakage(patch):
    fields = [patch.get("final_private_state", ""), patch.get("first_say", ""), patch.get("final_say", "")]
    for step in patch.get("steps") or []:
        if isinstance(step, dict):
            fields.append(step.get("pre_tool_private_state", ""))
            fields.append(step.get("post_tool_say", ""))
    return any(META_LEAKAGE_RE.search(str(f) or "") for f in fields)


def clean_private_state(text):
    if text is None:
        return ""
    text = str(text).strip()
    text = text.replace("[SOPR]", "").replace("[EOPR]", "")
    return text.strip()


def normalize_available_tools(tools):
    """
    將 OpenAI-style tools schema 壓成目標輸出格式:
    [{"name": "...", "description": "..."}]
    """
    tools = parse_json_maybe(tools, [])
    if not isinstance(tools, list):
        return []

    normalized = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function") if isinstance(tool.get("function"), dict) else tool
        name = fn.get("name") if isinstance(fn, dict) else None
        if not name:
            continue
        normalized.append({
            "name": name,
            "description": fn.get("description", "") if isinstance(fn, dict) else "",
        })
    return normalized


def normalize_tool_call_for_output(tool_call):
    """
    將各來源的 tool_call 統一成:
    {"function": {"name": "...", "arguments": {...}}}
    """
    if not isinstance(tool_call, dict):
        return {"function": {"name": "unknown", "arguments": {}}}

    fn = tool_call.get("function") if isinstance(tool_call.get("function"), dict) else tool_call
    name = fn.get("name") or tool_call.get("name") or "unknown"
    arguments = fn.get("arguments", tool_call.get("arguments", {}))
    arguments = parse_json_maybe(arguments, arguments)
    if arguments is None:
        arguments = {}
    return {"function": {"name": name, "arguments": arguments}}


def normalize_tool_result_for_output(tool_name, tool_result):
    """
    將 tool result 統一成 JSON object，避免輸出成裸字串。
    """
    parsed = parse_json_maybe(tool_result, None)
    if isinstance(parsed, dict):
        if "name" not in parsed:
            parsed = {"name": tool_name, **parsed}
        return parsed
    return {"name": tool_name, "response": "" if tool_result is None else str(tool_result)}


def tool_call_line(tool_call):
    return f"<TOOL_CALL>{json.dumps(tool_call, ensure_ascii=False)}</TOOL_CALL>"


def tool_result_line(tool_result):
    return f"<TOOL_RESULT>{json.dumps(tool_result, ensure_ascii=False)}</TOOL_RESULT>"


def as_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def pop_pending_tool_call(pending_tool_calls, pending_tool_calls_by_id, tool_call_id=None):
    # Prefer explicit tool_call_id when present; otherwise consume calls in order.
    if tool_call_id and tool_call_id in pending_tool_calls_by_id:
        tool_call = pending_tool_calls_by_id.pop(tool_call_id)
        pending_tool_calls = [
            tc for tc in pending_tool_calls
            if not (isinstance(tc, dict) and tc.get("id") == tool_call_id)
        ]
        return tool_call, pending_tool_calls, pending_tool_calls_by_id

    if pending_tool_calls:
        tool_call = pending_tool_calls.pop(0)
        if isinstance(tool_call, dict) and tool_call.get("id"):
            pending_tool_calls_by_id.pop(tool_call.get("id"), None)
        return tool_call, pending_tool_calls, pending_tool_calls_by_id

    return {}, pending_tool_calls, pending_tool_calls_by_id


def get_tool_response_call_id(tool_response):
    if not isinstance(tool_response, dict):
        return None
    return (
        tool_response.get("tool_call_id")
        or tool_response.get("call_id")
        or tool_response.get("id")
    )


def append_tool_step(canonical, step_id, tool_call, tool_result):
    output_tool_call = normalize_tool_call_for_output(tool_call)
    tool_name = get_tool_name(output_tool_call)
    output_tool_result = normalize_tool_result_for_output(tool_name, tool_result)
    call_line = tool_call_line(output_tool_call)
    result_line = tool_result_line(output_tool_result)

    canonical["steps_full"].append({
        "step_id": step_id,
        "tool_call": output_tool_call,
        "tool_result": output_tool_result
    })

    canonical["tool_steps"].append({
        "order": step_id + 1,
        "tool_call_line": call_line,
        "tool_result_line": result_line,
    })

    canonical["steps_compact"].append({
        "step_id": step_id,
        "tool_name": tool_name,
        "tool_call_brief": json.dumps(output_tool_call, ensure_ascii=False)[:300],
        "observation_brief": compact_observation(json.dumps(output_tool_result, ensure_ascii=False))
    })


def canonicalize_row(row):
    """
    將 voidful/agent-sft 格式的原始資料統一轉換為 Canonical Row 格式。
    
    格式如下。巢狀欄位在回傳前會序列化成 JSON 字串，避免 Ray/Arrow
    對不同 tool arguments schema 做 merge 時失敗。
    
    原始 canonical 格式:
    {
      "id": str,
      "source": str,
      "user_request": str,
      "steps_full": list,     # 完整步驟，包含完整 observation，組裝用 (不進 Prompt)
      "steps_compact": list,  # 壓縮步驟，只包含簡短說明，推論用 (進 Prompt)
      "available_tools": list,
      "tool_steps": list,
      "final_answer_hint": str
    }
    """
    rec_id = row.get("id") or "unknown_id"
    source = row.get("source") or "unknown_source"
    available_tools = normalize_available_tools(row.get("tools"))
    
    msgs = row.get("messages")
    if isinstance(msgs, str):
        try:
            msgs = json.loads(msgs) if msgs else []
        except:
            msgs = []
    elif msgs is None:
        msgs = []
            
    canonical = {
        "id": rec_id,
        "source": source,
        "user_request": "",
        "steps_full": [],
        "steps_compact": [],
        "available_tools": available_tools,
        "tool_steps": [],
        "final_answer_hint": "",
        "context": [],
    }

    step_id = 0
    pending_tool_calls = []
    pending_tool_calls_by_id = {}

    for msg in msgs:
        role = msg.get("role")
        raw_content = msg.get("content", "")
        content = normalize_text_content(raw_content)

        if role == "user":
            # 多輪對話：較早輪次的 user 訊息保留到 context，
            # user_request 永遠是最後一則 user 訊息。
            if canonical["user_request"]:
                canonical["context"].append(
                    {"role": "user", "content": compact_observation(canonical["user_request"], 800)}
                )
                if canonical["final_answer_hint"]:
                    canonical["context"].append(
                        {"role": "assistant", "content": compact_observation(canonical["final_answer_hint"], 800)}
                    )
                    canonical["final_answer_hint"] = ""
            canonical["user_request"] = content
        elif role == "assistant":
            # 取得 tool_calls
            tool_calls = msg.get("tool_calls")
            if tool_calls:
                tool_calls = as_list(tool_calls)
                pending_tool_calls = list(tool_calls)
                pending_tool_calls_by_id = {
                    tc.get("id"): tc
                    for tc in pending_tool_calls
                    if isinstance(tc, dict) and tc.get("id")
                }
                tool_responses = msg.get("tool_responses")
                if tool_responses:
                    for tool_call, tool_response in zip(tool_calls, as_list(tool_responses)):
                        append_tool_step(canonical, step_id, tool_call, tool_response)
                        step_id += 1
                    pending_tool_calls = pending_tool_calls[len(as_list(tool_responses)):]
                    pending_tool_calls_by_id = {
                        tc.get("id"): tc
                        for tc in pending_tool_calls
                        if isinstance(tc, dict) and tc.get("id")
                    }
            elif content:
                canonical["final_answer_hint"] = content
        elif role == "tool":
            # 工具結果。voidful/agent-sft 常把結果放在 tool_responses list，
            # 且 content 可能是 null。
            tool_responses = as_list(msg.get("tool_responses")) if msg.get("tool_responses") else [content]
            tool_call_id = msg.get("tool_call_id")

            for tool_result in tool_responses:
                response_tool_call_id = get_tool_response_call_id(tool_result) or tool_call_id
                # 用 tool_call_id 對齊對應的呼叫；若資料缺少 id，退回 FIFO。
                tool_call, pending_tool_calls, pending_tool_calls_by_id = pop_pending_tool_call(
                    pending_tool_calls,
                    pending_tool_calls_by_id,
                    response_tool_call_id,
                )
                append_tool_step(canonical, step_id, tool_call, tool_result)
                step_id += 1
            
    if msgs and msgs[-1].get("role") == "assistant" and not canonical["final_answer_hint"]:
        canonical["final_answer_hint"] = normalize_text_content(msgs[-1].get("content", ""))
        
    return {
        "id": canonical["id"],
        "source": canonical["source"],
        "user_request": canonical["user_request"],
        "steps_full_json": dumps_json_field(canonical["steps_full"]),
        "steps_compact_json": dumps_json_field(canonical["steps_compact"]),
        "available_tools_json": dumps_json_field(canonical["available_tools"]),
        "tool_steps_json": dumps_json_field(canonical["tool_steps"]),
        "final_answer_hint": canonical["final_answer_hint"],
        "language": canonical.get("language", "zh-TW"),
        "context_json": dumps_json_field(canonical.get("context", [])),
    }


def tool_reference_filter_failure(row):
    steps_full = loads_json_field(row.get("steps_full_json"), [])
    reference_answer = (row.get("final_answer_hint") or "").strip()
    reasons = []
    if not steps_full:
        reasons.append("missing_tool_call_tool_result")
    if not reference_answer:
        reasons.append("missing_reference_answer")
    if not reasons:
        return None
    return {
        "id": row.get("id"),
        "status": "dropped",
        "drop_reason": "+".join(reasons),
        "raw_patch": None,
        "error": "Filtered before inference because tool call/result or reference answer is missing.",
        "sft_data": None,
    }


def has_tool_call_tool_result_and_reference(row):
    return tool_reference_filter_failure(row) is None


def get_tool_name(tool_call):
    if not isinstance(tool_call, dict):
        return "unknown"
    fn = tool_call.get("function")
    if isinstance(fn, dict):
        return fn.get("name") or "unknown"
    return tool_call.get("name") or "unknown"

# ==========================================
# 3. Ray Data Batch Inference 處理器 (Processor)
# ==========================================

def build_user_payload(row):
    """
    從 canonical 欄位組裝給 Gemma 的 User Payload
    """
    available_tools = loads_json_field(row.get("available_tools_json"), [])
    steps_compact = loads_json_field(row.get("steps_compact_json"), [])
    payload = {
        "id": row["id"],
        "source": row["source"],
        "context": loads_json_field(row.get("context_json"), []),
        "user": row["user_request"],
        "available_tools": available_tools,
        # Prompt only needs compact observations to write SAY/SOPR patches.
        # Full tool call/result lines stay in tool_steps / steps_full for deterministic assembly.
        "tool_steps": steps_compact,
        "reference_answer": row.get("final_answer_hint", ""),
        "language": row.get("language", "zh-TW"),
    }
    return json.dumps(payload, ensure_ascii=False)


def estimate_prompt_tokens(text):
    # Conservative approximation for mixed English/Chinese/JSON text.
    # vLLM applies chat templating after this estimate, so keep this intentionally
    # stricter than a normal chars/token heuristic.
    return max(1, len(text) // 2)


def estimate_row_prompt_tokens(row):
    return (
        estimate_prompt_tokens(SYSTEM_PROMPT)
        + estimate_prompt_tokens(build_user_payload(row))
        + CHAT_TEMPLATE_TOKEN_OVERHEAD
    )


def prompt_fits_context(row):
    return estimate_row_prompt_tokens(row) <= PROMPT_TOKEN_BUDGET


def prompt_too_long_failure(row):
    prompt_tokens_estimate = estimate_row_prompt_tokens(row)
    return {
        "id": row["id"],
        "status": "dropped",
        "drop_reason": "prompt_too_long",
        "raw_patch": None,
        "error": (
            f"Estimated prompt tokens {prompt_tokens_estimate} exceed budget "
            f"{PROMPT_TOKEN_BUDGET}."
        ),
        "sft_data": None,
    }


def preprocess(row):
    """
    將資料轉換為 vLLM Engine 接受的 Input 格式。
    針對來回次數多（步驟多）的資料，動態調整 max_tokens，避免 JSON 被截斷。
    """
    steps_compact = loads_json_field(row.get("steps_compact_json"), [])
    num_steps = len(steps_compact)
    user_payload = build_user_payload(row)
    prompt_tokens_estimate = estimate_prompt_tokens(SYSTEM_PROMPT) + estimate_prompt_tokens(user_payload)
    prompt_tokens_estimate += CHAT_TEMPLATE_TOKEN_OVERHEAD
    # translated_user 會把整段 user 訊息翻譯回吐，長 user 訊息需要等比例的輸出預算，
    # 否則 JSON 會被 max_tokens 截斷。
    translation_tokens = estimate_prompt_tokens(row.get("user_request", ""))
    desired_max_tokens = min(MAX_OUTPUT_TOKENS, 1024 + num_steps * 384 + translation_tokens)
    available_max_tokens = MAX_MODEL_LEN - prompt_tokens_estimate - PROMPT_TOKEN_SAFETY_MARGIN
    estimated_max_tokens = max(MIN_OUTPUT_TOKENS, min(desired_max_tokens, available_max_tokens))
    
    return {
        "id": row["id"],
        "source": row["source"],
        "user_request": row["user_request"],
        "steps_full_json": row.get("steps_full_json", "[]"),
        "steps_compact_json": row.get("steps_compact_json", "[]"),
        "available_tools_json": row.get("available_tools_json", "[]"),
        "tool_steps_json": row.get("tool_steps_json", "[]"),
        "final_answer_hint": row.get("final_answer_hint", ""),
        "language": row.get("language", "zh-TW"),
        "context_json": row.get("context_json", "[]"),
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_payload},
        ],
        "sampling_params": {
            # 0.2 made every bridge SAY collapse into the same stock sentence across
            # the dataset; the eval judge penalizes that as template-like speech.
            "temperature": float(os.environ.get("GENERATE_TEMPERATURE", "0.6")),
            "top_p": 0.95,
            "max_tokens": estimated_max_tokens,
            "stop": ["\n\n<END_PATCH>"],
        },
    }


def postprocess(row):
    """
    擷取 LLM 生成的 Patch 文字，並保留 canonical 欄位。
    這樣後處理可以直接組裝，不需要再用 Ray Data join。
    """
    return {
        "id": row["id"],
        "source": row["source"],
        "user_request": row["user_request"],
        "steps_full_json": row.get("steps_full_json", "[]"),
        "steps_compact_json": row.get("steps_compact_json", "[]"),
        "available_tools_json": row.get("available_tools_json", "[]"),
        "tool_steps_json": row.get("tool_steps_json", "[]"),
        "final_answer_hint": row.get("final_answer_hint", ""),
        "language": row.get("language", "zh-TW"),
        "context_json": row.get("context_json", "[]"),
        "raw_patch": row.get("generated_text") or row.get("response") or "",
    }

# ==========================================
# 4. Deterministic Assembler (組裝最終 SFT target)
# ==========================================

def assemble_agent_stitch_s(canonical_row, raw_patch_str):
    """
    將 Gemma 生成的 Patch 與原始 Trajectory 進行確定性組裝 (Deterministic Assembly)。
    """
    try:
        patch = parse_patch_json(raw_patch_str)
    except Exception as e:
        # JSON 解析失敗，返回空或標記錯誤
        return {
            "error": f"JSON parse error: {str(e)}",
            "drop_reason": "patch_json_parse_failed",
            "linear_target": None,
        }
    
    if patch.get("drop_reason"):
        return {
            "drop_reason": patch["drop_reason"],
            "linear_target": None,
        }

    if patch_meta_leakage(patch):
        return {
            "drop_reason": "rewriter_meta_leakage",
            "linear_target": None,
        }
    
    parts = []
    
    # 1. first_say
    first_say = clean_say_text(patch.get("first_say", ""))
    parts.append(f"<SAY>{first_say}</SAY>")
    
    # 2. 中間步驟
    steps_patch = patch.get("steps", [])
    steps_full = canonical_row.get("steps_full", [])
    
    # 確保步驟數量對齊，若不對齊則捨棄
    if len(steps_patch) != len(steps_full):
        return {
            "drop_reason": f"step_count_mismatch: patch={len(steps_patch)}, full={len(steps_full)}",
            "linear_target": None,
        }
    
    for step_patch, step in zip(steps_patch, steps_full):
        pre_tool_private = clean_private_state(step_patch.get("pre_tool_private_state", ""))
        post_tool_say = clean_say_text(step_patch.get("post_tool_say", ""))
        
        # 寫入私有思維狀態 [SOPR]...[EOPR]
        parts.append(f"[SOPR]{pre_tool_private}[EOPR]")
        
        # 插入原始 Tool Call，接著放工具執行期間說給使用者聽的 bridge SAY。
        tool_call_str = json.dumps(step["tool_call"], ensure_ascii=False)
        parts.append(f"<TOOL_CALL>{tool_call_str}</TOOL_CALL>")
        parts.append(f"<SAY>{post_tool_say}</SAY>")

        # 最後插入原始 Tool Result。這樣順序會對齊目標格式:
        # TOOL_CALL -> SAY while tool is running -> TOOL_RESULT。
        tool_result_str = json.dumps(step["tool_result"], ensure_ascii=False)
        parts.append(f"<TOOL_RESULT>{tool_result_str}</TOOL_RESULT>")
        
    # 3. final_private_state
    final_private = clean_private_state(patch.get("final_private_state", ""))
    parts.append(f"[SOPR]{final_private}[EOPR]\n[EOR]")
    
    # 4. final_say
    final_say = clean_say_text(patch.get("final_say", ""))
    parts.append(f"<SAY>{final_say}</SAY>")
    
    linear_target = "\n\n".join(parts)
    
    return {
        "drop_reason": None,
        "linear_target": linear_target,
        "translated_user": patch.get("translated_user")
    }


def build_output_input(canonical_row):
    return {
        "language": canonical_row.get("language", "zh-TW"),
        "context": canonical_row.get("context", []),
        "user": canonical_row["user_request"],
        "available_tools": canonical_row.get("available_tools", []),
        "tool_steps": canonical_row.get("tool_steps", []),
        "reference_answer": canonical_row.get("final_answer_hint", ""),
    }


def generate_sft_row(canonical_row, assembled_result):
    """
    產生最後用於訓練的 SFT 樣本結構
    """
    return {
        "id": canonical_row["id"],
        "source": canonical_row["source"],
        "user": assembled_result.get("translated_user") or canonical_row["user_request"],
        "msg": assembled_result["linear_target"],
        "input": build_output_input(canonical_row)
    }


def make_run_timestamps():
    started_at = datetime.now().astimezone()
    return started_at, started_at.strftime("%Y%m%d_%H%M%S")


def take_random_preview(ds, max_rows):
    """
    從 Ray Dataset 抽少量隨機 preview。避免 full dataset 使用 take_all()
    或 random_shuffle() 造成 driver/cluster 額外負擔。
    """
    if max_rows <= 0:
        return []

    try:
        total_rows = ds.count()
        if total_rows <= max_rows:
            return ds.take(max_rows)

        sample_fraction = min(1.0, max(0.001, (max_rows * 5) / total_rows))
        rows = ds.random_sample(sample_fraction).take(max_rows)
        if len(rows) >= max_rows:
            return rows

        # Very small samples can undershoot by chance; retry with a wider sample.
        sample_fraction = min(1.0, max(0.001, (max_rows * 20) / total_rows))
        return ds.random_sample(sample_fraction).take(max_rows)
    except Exception as e:
        print(f"Random preview sampling failed; falling back to take({max_rows}): {e}")
        return ds.take(max_rows)


def write_success_preview(success_rows, preview_dir):
    """
    將成功樣本輸出成容易人工預覽的 JSONL / CSV。
    Full pipeline 只收集固定筆數隨機 preview，避免把全部資料拉回 driver。
    """
    os.makedirs(preview_dir, exist_ok=True)

    jsonl_path = os.path.join(preview_dir, "success.jsonl")
    csv_path = os.path.join(preview_dir, "success.csv")

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for row in success_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    fieldnames = ["id", "source", "user", "msg", "input_json"]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in success_rows:
            writer.writerow({
                "id": row.get("id"),
                "source": row.get("source"),
                "user": row.get("user"),
                "msg": row.get("msg"),
                "input_json": json.dumps(row.get("input"), ensure_ascii=False),
            })

    return jsonl_path, csv_path


def write_failed_preview(failed_rows, preview_dir):
    """
    將失敗樣本輸出成容易人工預覽的 JSONL / CSV。
    """
    os.makedirs(preview_dir, exist_ok=True)

    jsonl_path = os.path.join(preview_dir, "failed.jsonl")
    csv_path = os.path.join(preview_dir, "failed.csv")

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for row in failed_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    fieldnames = ["id", "status", "drop_reason", "error", "raw_patch", "sft_data"]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in failed_rows:
            writer.writerow({key: row.get(key) for key in fieldnames})

    return jsonl_path, csv_path

# ==========================================
# 5. 主程式流水線 (Main Production Pipeline)
# ==========================================

def load_only_ids(path):
    if not path:
        return None

    ids = set()
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            if line_no == 1 and line.lower() == "id":
                continue
            if line.startswith("{"):
                row = json.loads(line)
                row_id = row.get("id")
            else:
                row_id = line.split(",", 1)[0]
                if line_no == 1 and row_id == "id":
                    continue
            if row_id:
                ids.add(row_id)
    return ids


def run_pipeline(input_parquet_dir, output_sft_dir, only_ids_path=None):
    run_started_at, run_id = make_run_timestamps()
    print(f"Run timestamp: {run_started_at.isoformat()} (run_id={run_id})")

    # 初始化 Ray
    try:
        ray.init(address=os.environ.get("RAY_ADDRESS", "auto"))
    except Exception:
        print("Could not connect to existing Ray cluster. Starting a local Ray instance...")
        ray.init()
    
    # 1. 讀取並標準化完整資料
    print("Reading and canonicalizing input dataset...")
    is_hf_dataset = (
        input_parquet_dir.startswith("hf://")
        or ("/" in input_parquet_dir and not os.path.exists(input_parquet_dir))
    )
    if is_hf_dataset:
        dataset_name = input_parquet_dir.replace("hf://datasets/", "").replace("hf://", "")
        if "/" in dataset_name:
            parts = dataset_name.split("/")
            if len(parts) >= 2:
                dataset_name = f"{parts[0]}/{parts[1]}"
        print(f"Loading dataset from Hugging Face: {dataset_name}")
        import datasets
        hf_ds = datasets.load_dataset(dataset_name)
        if isinstance(hf_ds, datasets.DatasetDict):
            split_name = list(hf_ds.keys())[0]
            hf_ds = hf_ds[split_name]
        print(f"Using full Hugging Face split with {len(hf_ds)} rows")
        raw_ds = ray.data.from_huggingface(hf_ds)
    else:
        print(f"Reading parquet from local/cloud path: {input_parquet_dir}")
        raw_ds = ray.data.read_parquet(input_parquet_dir)
    
    only_ids = load_only_ids(only_ids_path)
    if only_ids is not None:
        print(f"Filtering input to {len(only_ids)} requested id(s) from {only_ids_path}")
        raw_ds = raw_ds.filter(lambda r: (r.get("id") or "unknown_id") in only_ids)

    print(f"Repartitioning raw input to {RAW_INPUT_BLOCKS} blocks")
    raw_ds = raw_ds.repartition(RAW_INPUT_BLOCKS, shuffle=False)

    canonical_ds = raw_ds.map(canonicalize_row).materialize()
    missing_tool_reference_ds = canonical_ds.filter(
        lambda r: not has_tool_call_tool_result_and_reference(r)
    ).map(tool_reference_filter_failure)
    eligible_ds = canonical_ds.filter(has_tool_call_tool_result_and_reference)

    llm_input_ds = eligible_ds.filter(prompt_fits_context)
    print(f"Repartitioning LLM input to {LLM_INPUT_BLOCKS} blocks")
    llm_input_ds = llm_input_ds.repartition(LLM_INPUT_BLOCKS, shuffle=False)
    prompt_too_long_ds = eligible_ds.filter(lambda r: not prompt_fits_context(r)).map(prompt_too_long_failure)
    
    # 2. vLLM 推論引擎配置
    print(
        "vLLM config: "
        f"tensor_parallel_size={TENSOR_PARALLEL_SIZE}, "
        f"concurrency={VLLM_CONCURRENCY}, "
        f"batch_size={VLLM_BATCH_SIZE}, "
        f"max_concurrent_batches={MAX_CONCURRENT_BATCHES}, "
        f"llm_input_blocks={LLM_INPUT_BLOCKS}"
    )
    config = vLLMEngineProcessorConfig(
        model_source=MODEL_ID,
        engine_kwargs={
            "tensor_parallel_size": TENSOR_PARALLEL_SIZE,
            "max_model_len": MAX_MODEL_LEN,  # 提高上限以容納多次來回的 Context
            "enable_chunked_prefill": True,
            "enable_prefix_caching": True,
            "kv_cache_dtype": "fp8",
            "gpu_memory_utilization": 0.90,
            "max_num_seqs": 256,
            "max_num_batched_tokens": 32768,
            "trust_remote_code": True,
        },
        concurrency=(VLLM_CONCURRENCY, VLLM_CONCURRENCY),
        batch_size=VLLM_BATCH_SIZE,
        should_continue_on_error=True,
        max_concurrent_batches=MAX_CONCURRENT_BATCHES,
        experimental={"max_tasks_in_flight_per_actor": 16},
    )
    
    processor = build_processor(
        config,
        preprocess=preprocess,
        postprocess=postprocess,
    )
    
    # 3. 執行推論產生 Patch
    print("Running vLLM engine inference...")
    patch_ds = processor(llm_input_ds)
    
    # 4. 組裝與後處理
    print("Assembling final SFT targets...")
    
    def process_patch_row(row):
        # 當 should_continue_on_error=True 時，engine 失敗的 row 會帶著
        # __inference_error__ 直接繞過 postprocess，因此沒有 raw_patch 欄位。
        inference_error = str(row.get("__inference_error__") or "")
        if inference_error or "raw_patch" not in row:
            return {
                "id": row.get("id"),
                "status": "dropped",
                "drop_reason": "inference_error",
                "raw_patch": None,
                "error": inference_error or "Row bypassed postprocess without raw_patch.",
                "sft_data": None,
            }
        canonical_row = {
            "id": row["id"],
            "source": row["source"],
            "user_request": row["user_request"],
            "steps_full": loads_json_field(row.get("steps_full_json"), []),
            "available_tools": loads_json_field(row.get("available_tools_json"), []),
            "tool_steps": loads_json_field(row.get("tool_steps_json"), []),
            "final_answer_hint": row.get("final_answer_hint", ""),
            "language": row.get("language", "zh-TW"),
            "context": loads_json_field(row.get("context_json"), []),
        }
        raw_patch_str = row["raw_patch"]
        
        assembled = assemble_agent_stitch_s(canonical_row, raw_patch_str)
        
        if assembled["drop_reason"] is not None:
            # 標記需要捨棄或進入 repair pass 的樣本
            return {
                "id": row["id"],
                "status": "dropped",
                "drop_reason": assembled["drop_reason"],
                "raw_patch": raw_patch_str,
                "error": assembled.get("error"),
                "sft_data": None
            }
        
        sft_row = generate_sft_row(canonical_row, assembled)
        return {
            "id": row["id"],
            "status": "success",
            "drop_reason": None,
            "raw_patch": None,
            "error": None,
            "sft_data": json.dumps(sft_row, ensure_ascii=False)
        }
        
    final_ds = patch_ds.map(process_patch_row).union(prompt_too_long_ds).union(missing_tool_reference_ds).materialize()
    
    # 5. 分流輸出：成功樣本與失敗樣本
    success_ds = final_ds.filter(lambda r: r["status"] == "success").map(lambda r: json.loads(r["sft_data"]))
    failed_ds = final_ds.filter(lambda r: r["status"] != "success")
    
    # 寫入最後的 Parquet。每次 run 使用獨立目錄，避免多次輸出混在一起。
    run_output_dir = os.path.join(output_sft_dir, "runs", run_id)
    success_output_path = os.path.join(run_output_dir, "success")
    failed_output_path = os.path.join(run_output_dir, "failed")
    preview_output_path = os.path.join(run_output_dir, "preview")
    
    success_ds = success_ds.materialize()
    success_ds.write_parquet(success_output_path)
    success_rows = take_random_preview(success_ds, PREVIEW_ROWS)
    preview_jsonl_path, preview_csv_path = write_success_preview(success_rows, preview_output_path)

    failed_ds = failed_ds.materialize()
    failed_ds.write_parquet(failed_output_path)
    failed_rows = take_random_preview(failed_ds, PREVIEW_ROWS)
    failed_preview_jsonl_path, failed_preview_csv_path = write_failed_preview(failed_rows, preview_output_path)

    print(f"Pipeline finished. SFT datasets written to {success_output_path}")
    print(f"Run output directory: {run_output_dir}")
    print(f"Success preview rows: {len(success_rows)}")
    print(f"Failed preview rows: {len(failed_rows)}")
    print(f"Success preview JSONL written to {preview_jsonl_path}")
    print(f"Success preview CSV written to {preview_csv_path}")
    print(f"Failed preview JSONL written to {failed_preview_jsonl_path}")
    print(f"Failed preview CSV written to {failed_preview_csv_path}")
