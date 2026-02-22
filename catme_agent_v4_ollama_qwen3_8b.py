#!/usr/bin/env python3
"""
CATME Peer Feedback LLM Classifier Agent
Uses Ollama (local) to classify feedback into CATME dimensions.
Run: ollama run qwen3:8b   (in a separate terminal to pre-load model)
"""

import json
import time
import re
import sys
from datetime import datetime
import pandas as pd
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── Configuration ──────────────────────────────────────────────────────────────
OLLAMA_HOST  = "http://192.168.4.214:11434"
OLLAMA_MODEL = "qwen3:8b"
OLLAMA_URL   = f"{OLLAMA_HOST}/api/chat"   # chat endpoint (supports messages + system)

INPUT_FILE = "CATME_DATA_Gates1_6_Fall2019_reduced_rows_cols.xlsx"
_RUN_STAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
OUTPUT_FILE = f"CATME_LLM_Classified_{_RUN_STAMP}.xlsx"
JSONL_FILE  = f"CATME_LLM_Responses_{_RUN_STAMP}.jsonl"  # one JSON record per line, written immediately

MAX_RETRIES = 3
RETRY_DELAY = 2
BATCH_DELAY = 0.0   # no rate limits on local Ollama
MAX_WORKERS = 1     # sequential to respect rate limits; increase if quota allows

# ── Mode: "api" for live calls, "test" to process first N rows with mock ──────
# Set via command line: python catme_agent.py [test N] or [api]
# If the API key returns 403, the script will auto-fall back to test mode.
RUN_MODE = "api"       # "api" or "test"
TEST_ROWS = None       # None = all rows; set a number to limit

# ── System Prompt (from spec document) ─────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert at analyzing peer feedback from engineering capstone teams.
Classify each comment according to five CATME dimensions plus an "other" category.

## CATME Dimensions (from official rating scale)

1. **Contributing**: Task completion, fair share, deadlines, attendance, helping with tasks, work quality.
   - Score 5: Does more or higher-quality work than expected. Makes important contributions. Helps teammates having difficulty.
   - Score 3: Completes fair share with acceptable quality. Keeps commitments on time. Helps when easy or important.
   - Score 1: Does not do fair share. Sloppy/incomplete work. Misses deadlines. Late/absent. Does not assist. Quits if difficult.

2. **Interacting**: Communication, listening, sharing info, respect, participation, feedback, encouragement.
   - Score 5: Asks for and shows interest in teammates' ideas. Keeps teammates informed. Provides encouragement. Uses feedback to improve.
   - Score 3: Listens and respects contributions. Communicates clearly. Shares info. Participates fully. Responds to feedback.
   - Score 1: Interrupts, ignores, bosses, or mocks teammates. Acts without input. Doesn't share info. Complains. Defensive.

3. **Keeping Track**: Monitoring progress, noticing problems, alerting team, solutions, organizing.
   - Score 5: Watches conditions and monitors progress. Ensures appropriate progress. Gives specific, timely, constructive feedback.
   - Score 3: Notices changes. Knows what everyone should be doing. Alerts teammates or suggests solutions when threatened.
   - Score 1: Unaware of goals. Doesn't pay attention to progress. Avoids discussing problems.

4. **Quality**: Motivation, standards, excellence, belief in team, going above and beyond.
   - Score 5: Motivates team to do excellent work. Cares about outstanding work even without reward. Believes team can do excellent work.
   - Score 3: Encourages good work meeting requirements. Wants team to perform well. Believes team can meet responsibilities.
   - Score 1: Satisfied if team doesn't meet standards. Wants team to avoid work. Doubts team can meet requirements.

5. **KSA** (Knowledge, Skills, and Abilities): Technical skills, domain knowledge, expertise, learning ability.
   - Score 5: Demonstrates KSA to do excellent work. Acquires new knowledge to improve team. Can perform any team member's role.
   - Score 3: Sufficient KSA to contribute. Acquires knowledge as needed. Can perform some other members' tasks.
   - Score 1: Missing basic qualifications. Unable/unwilling to develop. Unable to perform others' duties.

6. **Other**: Leadership, creativity, group comments, or too vague to code.

## CODING RULES (MUST FOLLOW)

Rule 1: "Being quiet" -> flag Interacting
Rule 2: "Being organized" -> Keeping Track (team coordination) OR Quality (work standards) — use context
Rule 3: "Does good work" without specific project -> flag Other (too vague)
Rule 4: "Does good work on [specific task]" -> flag Contributing
Rule 5: KSA mentions -> ALSO check if Contributing applies (skills used for tasks = both)
Rule 6: Keeping Track mention -> ALSO consider Interacting and Contributing
Rule 7: Exact CATME verbiage -> flag that dimension
Rule 8: "Share work / adequate work" -> flag Contributing
Rule 9: Comments about FULL GROUP (not individual) -> flag Other, is_group_comment=true
Rule 10: Leadership, creativity alone -> flag Other

## Additional Guidelines
- Multi-label: A comment can match multiple dimensions
- Flag for BOTH positive AND negative mentions of a dimension
- Empty, "N/A", "None", or meaningless comments -> all zeros, confidence=1.0

Respond ONLY with valid JSON (no markdown, no backticks):
{"contributing": 0, "interacting": 0, "keeping_track": 0, "quality": 0, "ksa": 0, "other": 0, "confidence": 0.9, "is_group_comment": false, "reasoning": "brief explanation citing rules"}

You are in non-thinking mode. Do NOT output any <think> blocks. Respond ONLY with the JSON object.
"""

# ── Few-Shot Examples ──────────────────────────────────────────────────────────
# Ollama /api/chat uses OpenAI-style messages: role = "user" | "assistant"
FEW_SHOT_EXAMPLES = [
    # Rule 1: quiet -> Interacting
    {"role": "user",      "content": 'Classify: "Very quiet, but speaks up when needed."'},
    {"role": "assistant", "content": '{"contributing": 0, "interacting": 1, "keeping_track": 0, "quality": 0, "ksa": 0, "other": 0, "confidence": 0.95, "is_group_comment": false, "reasoning": "Rule 1: quiet relates to communication style -> Interacting"}'},

    # Rule 2a: organized + team context -> Keeping Track
    {"role": "user",      "content": 'Classify: "Great job organizing the team and keeping us on track."'},
    {"role": "assistant", "content": '{"contributing": 0, "interacting": 0, "keeping_track": 1, "quality": 0, "ksa": 0, "other": 0, "confidence": 0.95, "is_group_comment": false, "reasoning": "Rule 2: organized in team coordination context -> Keeping Track"}'},

    # Rule 2b: organized + work quality -> Quality + Contributing
    {"role": "user",      "content": 'Classify: "Very organized in how she structures her reports."'},
    {"role": "assistant", "content": '{"contributing": 1, "interacting": 0, "keeping_track": 0, "quality": 1, "ksa": 0, "other": 0, "confidence": 0.90, "is_group_comment": false, "reasoning": "Rule 2: organized for work quality -> Quality; report work -> Contributing"}'},

    # Rule 3: vague praise -> Other
    {"role": "user",      "content": 'Classify: "Does good work."'},
    {"role": "assistant", "content": '{"contributing": 0, "interacting": 0, "keeping_track": 0, "quality": 0, "ksa": 0, "other": 1, "confidence": 0.95, "is_group_comment": false, "reasoning": "Rule 3: good work without specific project -> too vague -> Other"}'},

    # Rule 4: specific task -> Contributing
    {"role": "user",      "content": 'Classify: "Did excellent work on the Gate 1 report."'},
    {"role": "assistant", "content": '{"contributing": 1, "interacting": 0, "keeping_track": 0, "quality": 0, "ksa": 0, "other": 0, "confidence": 0.95, "is_group_comment": false, "reasoning": "Rule 4: good work on specific task (Gate 1 report) -> Contributing"}'},

    # Rule 5: KSA + task -> both
    {"role": "user",      "content": 'Classify: "Very knowledgeable in CAD, completed all drawings."'},
    {"role": "assistant", "content": '{"contributing": 1, "interacting": 0, "keeping_track": 0, "quality": 0, "ksa": 1, "other": 0, "confidence": 0.95, "is_group_comment": false, "reasoning": "Rule 5: KSA (CAD knowledge) applied to tasks (completed drawings) -> both KSA and Contributing"}'},

    # Rule 6: Keeping Track + Interacting + Contributing
    {"role": "user",      "content": 'Classify: "Organized meetings, reminded everyone of deadlines, and pitched in on coding."'},
    {"role": "assistant", "content": '{"contributing": 1, "interacting": 1, "keeping_track": 1, "quality": 0, "ksa": 0, "other": 0, "confidence": 0.95, "is_group_comment": false, "reasoning": "Rule 6: Keeping Track (organized meetings, deadlines) -> also check Interacting (reminded everyone) and Contributing (pitched in on coding)"}'},

    # Rule 8: fair share -> Contributing
    {"role": "user",      "content": 'Classify: "Does their fair share of the work."'},
    {"role": "assistant", "content": '{"contributing": 1, "interacting": 0, "keeping_track": 0, "quality": 0, "ksa": 0, "other": 0, "confidence": 0.95, "is_group_comment": false, "reasoning": "Rule 8: fair share of work -> Contributing"}'},

    # Rule 9: group comment -> Other + group flag
    {"role": "user",      "content": 'Classify: "We all worked well together this semester."'},
    {"role": "assistant", "content": '{"contributing": 0, "interacting": 0, "keeping_track": 0, "quality": 0, "ksa": 0, "other": 1, "confidence": 0.95, "is_group_comment": true, "reasoning": "Rule 9: comment about full group, not individual -> Other, group comment"}'},

    # Rule 10: leadership + creativity -> Other
    {"role": "user",      "content": 'Classify: "Great leader with creative ideas."'},
    {"role": "assistant", "content": '{"contributing": 0, "interacting": 0, "keeping_track": 0, "quality": 0, "ksa": 0, "other": 1, "confidence": 0.90, "is_group_comment": false, "reasoning": "Rule 10: leadership and creativity alone -> Other"}'},

    # Empty/N/A
    {"role": "user",      "content": 'Classify: "N/A"'},
    {"role": "assistant", "content": '{"contributing": 0, "interacting": 0, "keeping_track": 0, "quality": 0, "ksa": 0, "other": 0, "confidence": 1.0, "is_group_comment": false, "reasoning": "Empty/N/A comment -> all zeros"}'},

    # Negative feedback
    {"role": "user",      "content": "Classify: \"He hasn't done anything for the team.\""},
    {"role": "assistant", "content": '{"contributing": 1, "interacting": 1, "keeping_track": 0, "quality": 0, "ksa": 0, "other": 0, "confidence": 0.90, "is_group_comment": false, "reasoning": "Negative: not doing work -> Contributing; not participating -> Interacting."}'},

    # Multi-dimension
    {"role": "user",      "content": 'Classify: "Good teammate, does work on time."'},
    {"role": "assistant", "content": '{"contributing": 1, "interacting": 0, "keeping_track": 0, "quality": 0, "ksa": 0, "other": 0, "confidence": 0.90, "is_group_comment": false, "reasoning": "Does work on time -> Contributing (deadlines, task completion)"}'},

    # Interacting positive
    {"role": "user",      "content": 'Classify: "He works well with teammates"'},
    {"role": "assistant", "content": '{"contributing": 0, "interacting": 1, "keeping_track": 0, "quality": 0, "ksa": 0, "other": 0, "confidence": 0.90, "is_group_comment": false, "reasoning": "Works well with teammates -> communication/participation -> Interacting"}'},

    # Quality example
    {"role": "user",      "content": 'Classify: "Always pushes the team to do better and holds everyone to high standards."'},
    {"role": "assistant", "content": '{"contributing": 0, "interacting": 0, "keeping_track": 0, "quality": 1, "ksa": 0, "other": 0, "confidence": 0.95, "is_group_comment": false, "reasoning": "Pushes team to do better, high standards -> Quality (motivation, standards, excellence)"}'},
]



def is_empty_comment(comment):
    """Check if a comment is empty or N/A."""
    if comment is None or pd.isna(comment):
        return True
    c = str(comment).strip().lower()
    return c in ("", "n/a", "na", "none", ".", "-", "--", "no comment", "no comments", "nothing")


EMPTY_RESULT = {
    "contributing": 0, "interacting": 0, "keeping_track": 0,
    "quality": 0, "ksa": 0, "other": 0,
    "confidence": 1.0, "is_group_comment": False, "reasoning": "Empty/N/A comment"
}


def parse_llm_response(text):
    """Extract JSON from LLM response, handling markdown fences, truncation, and malformed output."""

     # Strip Qwen3 <think>...</think> blocks if present
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    
    if not text or not text.strip():
        return None

    text = text.strip()
    # Strip markdown code fences
    text = re.sub(r'^```(?:json)?\s*', '', text)
    text = re.sub(r'\s*```$', '', text)
    text = text.strip()

    # Try direct parse first
    try:
        result = json.loads(text)
        return _validate_result(result)
    except json.JSONDecodeError:
        pass

    # Try to find JSON object in the text
    match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group())
            return _validate_result(result)
        except json.JSONDecodeError:
            pass

    # Try to fix common truncation issues — close unclosed strings and braces
    fixed = text
    if fixed.count('"') % 2 == 1:
        fixed += '"'
    if fixed.count('{') > fixed.count('}'):
        fixed += '}' * (fixed.count('{') - fixed.count('}'))
    try:
        result = json.loads(fixed)
        return _validate_result(result)
    except json.JSONDecodeError:
        pass

    # Try extracting individual fields with regex as last resort
    return _regex_extract(text)


def _validate_result(result):
    """Validate and normalize a parsed JSON result."""
    if not isinstance(result, dict):
        return None
    for dim in ["contributing", "interacting", "keeping_track", "quality", "ksa", "other"]:
        result[dim] = int(bool(result.get(dim, 0)))
    result["confidence"] = max(0.0, min(1.0, float(result.get("confidence", 0.5))))
    result["is_group_comment"] = bool(result.get("is_group_comment", False))
    result["reasoning"] = str(result.get("reasoning", ""))[:500]
    return result


def _regex_extract(text):
    """Last-resort: extract dimension flags from malformed text using regex."""
    result = {
        "contributing": 0, "interacting": 0, "keeping_track": 0,
        "quality": 0, "ksa": 0, "other": 0,
        "confidence": 0.5, "is_group_comment": False,
        "reasoning": "Parsed from malformed response"
    }
    for dim in ["contributing", "interacting", "keeping_track", "quality", "ksa", "other"]:
        m = re.search(rf'"{dim}"\s*:\s*(\d)', text)
        if m:
            result[dim] = int(bool(int(m.group(1))))
    m = re.search(r'"confidence"\s*:\s*([\d.]+)', text)
    if m:
        result["confidence"] = max(0.0, min(1.0, float(m.group(1))))
    m = re.search(r'"is_group_comment"\s*:\s*(true|false)', text, re.IGNORECASE)
    if m:
        result["is_group_comment"] = m.group(1).lower() == "true"
    m = re.search(r'"reasoning"\s*:\s*"([^"]*)', text)
    if m:
        result["reasoning"] = m.group(1)[:500]
    # Check if we actually found anything useful
    if any(result[d] for d in ["contributing", "interacting", "keeping_track", "quality", "ksa", "other"]):
        return result
    return None


def classify_comment(comment, row_idx, total):
    """Classify a single comment via Ollama /api/chat."""
    if is_empty_comment(comment):
        return EMPTY_RESULT

    comment_str = str(comment).strip()

    # ── Print comment being sent to model ──────────────────────────────────
    print(f"\n  ┌─ [Row {row_idx}/{total}] COMMENT SENT TO MODEL " + "─" * 28)
    print(f"  │  {comment_str}")
    print(f"  └─" + "─" * 50)

    # Build messages: system + few-shot history + new comment
    # /no_think disables Qwen3's chain-of-thought <think> blocks so response is pure JSON
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(FEW_SHOT_EXAMPLES)
    #messages.append({"role": "user", "content": f'Classify: "{comment_str}" /no_think'})
    messages.append({"role": "user", "content": f'Classify: "{comment_str}"'})

    payload = {
        "model":    OLLAMA_MODEL,
        "messages": messages,
        "stream":   False,
#        "format":   "json",          # Ollama native JSON mode
        "options": {
            "temperature": 0.7,      # Qwen3 non-thinking mode recommended setting
            "top_p":       0.8,      # Qwen3 non-thinking mode recommended setting
            "top_k":       20,       # Qwen3 non-thinking mode recommended setting
            "num_predict": 300,      # max tokens to generate
        }
    }

    last_error = ""
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(OLLAMA_URL, json=payload, timeout=120)

            if resp.status_code != 200:
                last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
                print(f"  [Row {row_idx}/{total}] HTTP {resp.status_code} (attempt {attempt+1})")
                time.sleep(RETRY_DELAY * (2 ** attempt))
                continue

            data = resp.json()
            text = data.get("message", {}).get("content", "").strip()

            if not text:
                last_error = "Empty response from Ollama"
                print(f"  [Row {row_idx}/{total}] Empty response (attempt {attempt+1})")
                time.sleep(RETRY_DELAY)
                continue

            # ── Print raw model response ────────────────────────────────
            print(f"  ┌─ [Row {row_idx}/{total}] RAW MODEL RESPONSE " + "─" * 29)
            print(f"  │  {text}")
            print(f"  └─" + "─" * 50)

            result = parse_llm_response(text)
            if result:
                return result

            last_error = f"Could not parse: {text[:100]}"
            print(f"  [Row {row_idx}/{total}] Parse failed (attempt {attempt+1}): {text[:120]}")
            time.sleep(RETRY_DELAY)

        except requests.exceptions.Timeout:
            last_error = "Timeout"
            print(f"  [Row {row_idx}/{total}] Timeout (attempt {attempt+1})")
            time.sleep(RETRY_DELAY * (2 ** attempt))
        except requests.exceptions.RequestException as e:
            last_error = str(e)
            print(f"  [Row {row_idx}/{total}] Connection error (attempt {attempt+1}): {e}")
            time.sleep(RETRY_DELAY * (2 ** attempt))
        except Exception as e:
            last_error = str(e)
            print(f"  [Row {row_idx}/{total}] Unexpected error (attempt {attempt+1}): {e}")
            time.sleep(RETRY_DELAY)

    return {**EMPTY_RESULT, "confidence": 0.0, "reasoning": f"FAILED: {last_error}"}


def verify_ollama():
    """Check that Ollama is running and the model is available. Returns True if ready."""
    # 1. Check Ollama is reachable
    try:
        resp = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
        if resp.status_code != 200:
            print(f"  ⚠ Ollama returned status {resp.status_code}")
            return False
        models = [m["name"] for m in resp.json().get("models", [])]
    except Exception as e:
        print(f"  ⚠ Cannot reach Ollama at {OLLAMA_HOST}: {e}")
        print("  → Is Ollama running?  Start it with:  ollama serve")
        return False

    # 2. Check the target model is pulled
    # Ollama stores models as "name:tag"; match loosely
    if not any(OLLAMA_MODEL in m for m in models):
        print(f"  ⚠ Model '{OLLAMA_MODEL}' not found in Ollama.")
        print(f"  → Pull it with:  ollama pull {OLLAMA_MODEL}")
        print(f"  → Available models: {models}")
        return False

    # 3. Quick inference test
    try:
        payload = {
            "model": OLLAMA_MODEL,
            "messages": [{"role": "user", "content": "Reply with exactly: OK /no_think"}],
            "stream": False,
            "options": {"num_predict": 10}
        }
        resp = requests.post(OLLAMA_URL, json=payload, timeout=30)
        if resp.status_code == 200:
            return True
        print(f"  ⚠ Test inference failed: HTTP {resp.status_code}")
        return False
    except Exception as e:
        print(f"  ⚠ Test inference error: {e}")
        return False


def main():
    global RUN_MODE, TEST_ROWS

    # Parse CLI args
    if len(sys.argv) > 1:
        if sys.argv[1] == "test":
            RUN_MODE = "test"
            TEST_ROWS = int(sys.argv[2]) if len(sys.argv) > 2 else 20
        elif sys.argv[1] == "api":
            RUN_MODE = "api"
            if len(sys.argv) > 2:
                TEST_ROWS = int(sys.argv[2])

    print("=" * 70)
    print("CATME Peer Feedback LLM Classifier")
    print(f"Model: {OLLAMA_MODEL}  (Ollama @ {OLLAMA_HOST})")
    print("=" * 70)

    # ── Verify API ─────────────────────────────────────────────────────────
    print("\n[0/3] Verifying Ollama connection and model...")
    api_ok = verify_ollama()
    if api_ok:
        print(f"  ✓ Ollama is running and '{OLLAMA_MODEL}' is ready")
    else:
        if RUN_MODE == "api":
            print("\n  Switching to TEST mode (first 20 rows only)")
            print(f"  → Start Ollama:      ollama serve")
            print(f"  → Pull the model:    ollama pull {OLLAMA_MODEL}")
            RUN_MODE = "test"
            TEST_ROWS = 20

    # ── Load Data ──────────────────────────────────────────────────────────
    print("\n[1/3] Loading dataset...")
    df = pd.read_excel(INPUT_FILE)
    print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")
    print(f"  Columns: {list(df.columns)}")

    comment_col = df.columns[6]  # Column G = Comment
    print(f"  Comment column: '{comment_col}'")

    non_empty = df[~df[comment_col].apply(is_empty_comment)]
    print(f"  Non-empty comments: {len(non_empty)} / {len(df)}")

    if TEST_ROWS:
        df = df.head(TEST_ROWS).copy()
        print(f"  ⚠ Limiting to first {TEST_ROWS} rows ({RUN_MODE} mode)")

    # ── Classify ───────────────────────────────────────────────────────────
    if RUN_MODE == "test" and not api_ok:
        print(f"\n[2/3] TEST MODE — skipping API calls for {len(df)} rows")
        print("  Using empty placeholders. Fix API key and re-run with: python catme_agent.py api")
        results = [{**EMPTY_RESULT, "confidence": 0.0, "reasoning": "TEST MODE - no API call"} for _ in range(len(df))]
    else:
        print(f"\n[2/3] Classifying {len(df)} comments via Ollama ({OLLAMA_MODEL})...")
        print(f"  Estimated time: ~{len(df) * 0.5 / 60:.0f}-{len(df) * 1.5 / 60:.0f} minutes")

        results = []
        cache = {}  # Comment cache for identical comments
        cache_hits = 0
        start_time = time.time()

        print(f"  Live responses saved to: {JSONL_FILE}")
        jsonl_fh = open(JSONL_FILE, "w", encoding="utf-8")

        try:
          for idx, row in df.iterrows():
            comment = row[comment_col]
            cache_key = str(comment).strip().lower() if not is_empty_comment(comment) else "__EMPTY__"

            if cache_key in cache:
                result = cache[cache_key]
                results.append(result)
                cache_hits += 1
            else:
                result = classify_comment(comment, idx + 1, len(df))
                cache[cache_key] = result
                results.append(result)
                time.sleep(BATCH_DELAY)

            # ── Write result to JSONL immediately ──────────────────────
            record = {"row": int(idx + 1), "comment": str(comment), **result}
            jsonl_fh.write(json.dumps(record) + "\n")
            jsonl_fh.flush()

            # Progress reporting
            if (idx + 1) % 50 == 0 or idx + 1 == len(df):
                elapsed = time.time() - start_time
                rate = (idx + 1) / elapsed if elapsed > 0 else 0
                eta = (len(df) - idx - 1) / rate if rate > 0 else 0
                print(f"  Processed {idx+1}/{len(df)} | "
                      f"Cache hits: {cache_hits} | "
                      f"Rate: {rate:.1f} rows/s | "
                      f"ETA: {eta/60:.1f} min")
        finally:
          jsonl_fh.close()
          print(f"  ✓ JSONL log closed: {JSONL_FILE}")

        elapsed_total = time.time() - start_time
        print(f"\n  Classification complete in {elapsed_total/60:.1f} minutes")
        print(f"  Unique API calls: {len(cache)} | Cache hits: {cache_hits}")

    # ── Build Output ───────────────────────────────────────────────────────
    print("\n[3/3] Building output Excel...")

    df["LLM_Contributing"] = [r["contributing"] for r in results]
    df["LLM_Interacting"] = [r["interacting"] for r in results]
    df["LLM_KeepingTrack"] = [r["keeping_track"] for r in results]
    df["LLM_Quality"] = [r["quality"] for r in results]
    df["LLM_KSA"] = [r["ksa"] for r in results]
    df["LLM_Other"] = [r["other"] for r in results]
    df["LLM_Confidence"] = [r["confidence"] for r in results]
    df["LLM_GroupComment"] = [r["is_group_comment"] for r in results]
    df["LLM_Reasoning"] = [r["reasoning"] for r in results]

    # ── Summary Stats ──────────────────────────────────────────────────────
    orig_cols = df.columns[7:12]  # H-L original flags
    llm_cols = ["LLM_Contributing", "LLM_Interacting", "LLM_KeepingTrack", "LLM_Quality", "LLM_KSA"]
    dim_names = ["Contributing", "Interacting", "Keeping Track", "Quality", "KSA"]

    print("\n  ┌─────────────────────┬────────────┬────────────┐")
    print("  │ Dimension           │ Original   │ LLM        │")
    print("  ├─────────────────────┼────────────┼────────────┤")
    for name, orig, llm in zip(dim_names, orig_cols, llm_cols):
        o_count = df[orig].sum()
        l_count = df[llm].sum()
        print(f"  │ {name:<19} │ {o_count:>6} {o_count/len(df)*100:>4.1f}% │ {l_count:>6} {l_count/len(df)*100:>4.1f}% │")
    print("  └─────────────────────┴────────────┴────────────┘")

    # Agreement rates
    print("\n  Agreement rates (Original vs LLM):")
    for name, orig, llm in zip(dim_names, orig_cols, llm_cols):
        agree = (df[orig] == df[llm]).sum()
        print(f"    {name}: {agree}/{len(df)} ({agree/len(df)*100:.1f}%)")

    avg_conf = df["LLM_Confidence"].mean()
    low_conf = (df["LLM_Confidence"] < 0.7).sum()
    print(f"\n  Avg confidence: {avg_conf:.3f}")
    print(f"  Low confidence (<0.7): {low_conf} rows")

    # ── Save ───────────────────────────────────────────────────────────────
    Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(OUTPUT_FILE, index=False, engine="openpyxl")
    print(f"\n  ✓ Saved to: {OUTPUT_FILE}")
    print("=" * 70)


if __name__ == "__main__":
    main()
