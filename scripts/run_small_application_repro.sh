#!/usr/bin/env bash
set -euo pipefail

# RAM-aware application profile for Apple Silicon laptops with 16 GB RAM.
# Generated artifacts are written under runs/ and ignored by Git.

MODEL_ID="${MODEL_ID:-google/gemma-3-1b-it}"
RUN_DIR="${RUN_DIR:-runs/small_gemma3_1b_application}"
GEN_DEVICE="${GEN_DEVICE:-cpu}"
ANALYSIS_DEVICE="${ANALYSIS_DEVICE:-cpu}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_LENGTH="${MAX_LENGTH:-384}"
START_TOKEN="${START_TOKEN:-30}"
N_PER_PROMPT="${N_PER_PROMPT:-3}"
LIMIT_EMOTIONS="${LIMIT_EMOTIONS:-}"
LIMIT_TOPICS="${LIMIT_TOPICS:-}"

EMOTIONS_FILE="${EMOTIONS_FILE:-data/seeds/emotions_16.txt}"
TOPICS_FILE="${TOPICS_FILE:-data/seeds/topics_12.txt}"
NEUTRAL_FILE="${NEUTRAL_FILE:-data/examples/neutral_tiny.jsonl}"
PROBES_FILE="${PROBES_FILE:-data/probes/probe_texts_16.jsonl}"

mkdir -p "$RUN_DIR/data" "$RUN_DIR/outputs"

stories_path="$RUN_DIR/data/stories.jsonl"
raw_path="$RUN_DIR/data/raw_generations.jsonl"
vectors_path="$RUN_DIR/outputs/emotion_vectors.npz"
scores_path="$RUN_DIR/outputs/probe_scores.jsonl"

generate_args=(
  scripts/generate_stories.py
  --model-id "$MODEL_ID"
  --emotions "$EMOTIONS_FILE"
  --topics "$TOPICS_FILE"
  --out "$stories_path"
  --raw-out "$raw_path"
  --n-per-prompt "$N_PER_PROMPT"
  --min-stories-per-prompt "$N_PER_PROMPT"
  --max-retries 2
  --device "$GEN_DEVICE"
  --max-new-tokens 650
  --temperature 0.8
  --top-p 0.9
)

if [[ -n "$LIMIT_EMOTIONS" ]]; then
  generate_args+=(--limit-emotions "$LIMIT_EMOTIONS")
fi
if [[ -n "$LIMIT_TOPICS" ]]; then
  generate_args+=(--limit-topics "$LIMIT_TOPICS")
fi

echo "==> Generating stories"
PYTHONPATH=src python3 "${generate_args[@]}"

echo "==> Validating story balance"
validate_args=(
  scripts/validate_stories.py
  --stories "$stories_path"
  --emotions "$EMOTIONS_FILE"
  --topics "$TOPICS_FILE"
  --expected-per-cell "$N_PER_PROMPT"
  --min-words 45
)
if [[ -n "$LIMIT_EMOTIONS" || -n "$LIMIT_TOPICS" ]]; then
  echo "Skipping full-grid validation because LIMIT_EMOTIONS or LIMIT_TOPICS is set."
else
  PYTHONPATH=src python3 "${validate_args[@]}"
fi

echo "==> Computing emotion vectors"
PYTHONPATH=src python3 scripts/compute_vectors.py \
  --model-id "$MODEL_ID" \
  --stories "$stories_path" \
  --neutral "$NEUTRAL_FILE" \
  --out "$vectors_path" \
  --device "$ANALYSIS_DEVICE" \
  --dtype float32 \
  --batch-size "$BATCH_SIZE" \
  --max-length "$MAX_LENGTH" \
  --start-token "$START_TOKEN"

echo "==> Scoring probe texts"
PYTHONPATH=src python3 scripts/score_texts.py \
  --model-id "$MODEL_ID" \
  --vectors "$vectors_path" \
  --input "$PROBES_FILE" \
  --output "$scores_path" \
  --device "$ANALYSIS_DEVICE" \
  --dtype float32 \
  --max-length "$MAX_LENGTH" \
  --start-token 0

cat <<EOF

Done.

Run directory:
  $RUN_DIR

Generated files:
  $stories_path
  $raw_path
  $vectors_path
  $scores_path

EOF
