#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: generate_casualty_snippets.sh --ls-url URL --ls-api-key TOKEN \
       --project PROJECT_ID --task TASK_ID --annotation ANNOTATION_ID \
       --summary SUMMARY_JSON [--person-id ID] [--min-frames N | --min-seconds S] \
       [--fps FPS] [--output-dir DIR]

Downloads the raw video from Label Studio and generates per-casualty MP4 snippets
from the summary JSON produced by export_interpolated_annotation.sh. Requires
curl, jq, and ffmpeg.
EOF
}

LS_URL=""
LS_API_KEY=""
PROJECT_ID=""
TASK_ID=""
ANNOTATION_ID=""
SUMMARY_PATH=""
PERSON_ID=""
MIN_FRAMES=""
MIN_SECONDS=""
FPS_TARGET=""
OUTPUT_DIR=""
APT_UPDATED=0
DEBUG=0

run_privileged() {
  if (( EUID == 0 )); then
    "$@"
  elif command -v sudo >/dev/null 2>&1; then
    sudo "$@"
  else
    echo "[error] Installing packages requires root privileges or sudo. Please rerun with sudo." >&2
    exit 1
  fi
}

install_package() {
  local pkg="$1"
  if command -v apt-get >/dev/null 2>&1; then
    if (( APT_UPDATED == 0 )); then
      echo "[info] Updating apt package index..."
      run_privileged apt-get update -y >/dev/null
      APT_UPDATED=1
    fi
    echo "[info] Installing $pkg via apt-get..."
    run_privileged apt-get install -y "$pkg"
  elif command -v dnf >/dev/null 2>&1; then
    echo "[info] Installing $pkg via dnf..."
    run_privileged dnf install -y "$pkg"
  elif command -v yum >/dev/null 2>&1; then
    echo "[info] Installing $pkg via yum..."
    run_privileged yum install -y "$pkg"
  else
    echo "[error] Could not find a supported package manager (apt-get, dnf, yum) to install $pkg." >&2
    exit 1
  fi
}

ensure_tool() {
  local bin="$1"
  local pkg="${2:-$1}"
  if ! command -v "$bin" >/dev/null 2>&1; then
    echo "[warn] $bin not found. Attempting to install $pkg..."
    install_package "$pkg"
    if ! command -v "$bin" >/dev/null 2>&1; then
      echo "[error] Failed to install $pkg. Please install it manually and rerun." >&2
      exit 1
    fi
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ls-url)
      LS_URL="$2"; shift 2;;
    --ls-api-key)
      LS_API_KEY="$2"; shift 2;;
    --project)
      PROJECT_ID="$2"; shift 2;;
    --task)
      TASK_ID="$2"; shift 2;;
    --annotation)
      ANNOTATION_ID="$2"; shift 2;;
    --summary)
      SUMMARY_PATH="$2"; shift 2;;
    --person-id)
      PERSON_ID="$2"; shift 2;;
    --min-frames)
      MIN_FRAMES="$2"; shift 2;;
    --min-seconds)
      MIN_SECONDS="$2"; shift 2;;
    --fps)
      FPS_TARGET="$2"; shift 2;;
    --output-dir)
      OUTPUT_DIR="$2"; shift 2;;
    --debug)
      DEBUG=1; shift;;
    -h|--help)
      usage; exit 0;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1;;
  esac
done

if [[ -z "$LS_URL" || -z "$LS_API_KEY" || -z "$PROJECT_ID" || -z "$TASK_ID" || -z "$ANNOTATION_ID" || -z "$SUMMARY_PATH" ]]; then
  echo "Missing required arguments." >&2
  usage
  exit 1
fi

if [[ -n "$MIN_FRAMES" && -n "$MIN_SECONDS" ]]; then
  echo "[error] --min-frames and --min-seconds are mutually exclusive." >&2
  exit 1
fi

if [[ ! -f "$SUMMARY_PATH" ]]; then
  echo "[error] Summary file not found: $SUMMARY_PATH" >&2
  exit 1
fi

if [[ "$DEBUG" -eq 1 ]]; then
  echo "[debug] Debug mode enabled" >&2
  set -x
fi

ensure_tool curl curl
ensure_tool jq jq
ensure_tool ffmpeg ffmpeg

BASE_URL="${LS_URL%/}"
AUTH_HEADER=("Authorization: Token $LS_API_KEY")
TMP_PARENT="$(pwd)"
TMP_DIR="$(mktemp -d -p "$TMP_PARENT" snippet_tmp.XXXXXX)"
trap 'rm -rf "$TMP_DIR"' EXIT

SUMMARY_FPS=$(jq -r '.fps // empty' "$SUMMARY_PATH")
VIDEO_URL=$(jq -r '.video_url // empty' "$SUMMARY_PATH")
if [[ -z "$SUMMARY_FPS" || -z "$VIDEO_URL" ]]; then
  echo "[error] Missing fps or video_url in summary JSON." >&2
  exit 1
fi

if [[ -z "$FPS_TARGET" ]]; then
  FPS_TARGET="$SUMMARY_FPS"
fi

if [[ "$VIDEO_URL" == http* ]]; then
  FULL_VIDEO_URL="$VIDEO_URL"
elif [[ "$VIDEO_URL" == /* ]]; then
  FULL_VIDEO_URL="$BASE_URL$VIDEO_URL"
else
  FULL_VIDEO_URL="$BASE_URL/$VIDEO_URL"
fi

VIDEO_PATH="$TMP_DIR/source_video"
VIDEO_EXT="${VIDEO_URL##*.}"
if [[ -n "$VIDEO_EXT" && "$VIDEO_EXT" != "$VIDEO_URL" ]]; then
  VIDEO_PATH="$VIDEO_PATH.$VIDEO_EXT"
else
  VIDEO_PATH="$VIDEO_PATH.mp4"
fi

echo "[info] Downloading video..." >&2
curl -sS -L -o "$VIDEO_PATH" "$FULL_VIDEO_URL" -H "${AUTH_HEADER[@]}"

MIN_FRAMES_JSON="null"
MIN_SECONDS_JSON="null"
if [[ -n "$MIN_FRAMES" ]]; then
  MIN_FRAMES_JSON="$MIN_FRAMES"
fi
if [[ -n "$MIN_SECONDS" ]]; then
  MIN_SECONDS_JSON="$MIN_SECONDS"
fi

RANGE_FILE="$TMP_DIR/ranges.tsv"
jq -r --arg person "$PERSON_ID" --argjson min_frames "$MIN_FRAMES_JSON" --argjson min_seconds "$MIN_SECONDS_JSON" '
  def ranges_for($id; $ranges):
    $ranges[]
    | {
        id: $id,
        start_frame: .start_frame,
        end_frame: .end_frame,
        start_time: .start_time,
        end_time: .end_time
      };

  (.casualties // {})
  | to_entries
  | (if $person != "" then map(select(.key == $person)) else . end)
  | .[]
  | .key as $id
  | (.value.ranges // [])
  | ranges_for($id; .)
  | select(($min_frames == null) or ((.end_frame - .start_frame + 1) >= $min_frames))
  | select(($min_seconds == null) or ((.end_time - .start_time) >= $min_seconds))
  | [.id, .start_frame, .end_frame, .start_time, .end_time]
  | @tsv
' "$SUMMARY_PATH" > "$RANGE_FILE"

TOTAL_SNIPPETS=$(wc -l < "$RANGE_FILE" | tr -d ' ')
if [[ "$TOTAL_SNIPPETS" -eq 0 ]]; then
  echo "[info] No ranges matched the filters. Nothing to do." >&2
  exit 0
fi

if [[ -z "$OUTPUT_DIR" ]]; then
  OUTPUT_DIR="$(pwd)/snippets_proj${PROJECT_ID}_task${TASK_ID}_ann${ANNOTATION_ID}_$(date -u +"%Y%m%dT%H%M%SZ")"
fi
mkdir -p "$OUTPUT_DIR"

FPS_NAME=$(awk -v fps="$FPS_TARGET" 'BEGIN{printf "%.0f", fps}')
FPS_SAME=$(awk -v a="$FPS_TARGET" -v b="$SUMMARY_FPS" 'BEGIN{diff=a-b; if (diff<0) diff=-diff; if (diff < 0.000001) print 1; else print 0}')

README_PATH="$OUTPUT_DIR/README.txt"
cat <<EOF > "$README_PATH"
Casualty snippet export

Generated at: $(date -u +"%Y-%m-%dT%H:%M:%SZ")
Project ID: $PROJECT_ID
Task ID: $TASK_ID
Annotation ID: $ANNOTATION_ID
Summary JSON: $SUMMARY_PATH
Video URL: $FULL_VIDEO_URL
Person ID filter: ${PERSON_ID:-<all>}
Min frames: ${MIN_FRAMES:-<none>}
Min seconds: ${MIN_SECONDS:-<none>}
Target FPS: $FPS_TARGET
Output directory: $OUTPUT_DIR
Total snippets: $TOTAL_SNIPPETS
Stream copy when FPS unchanged: true
EOF

echo "[info] Generating $TOTAL_SNIPPETS snippet(s)..." >&2
COUNTER=0
while IFS=$'\t' read -r casualty_id start_frame end_frame start_time end_time; do
  COUNTER=$((COUNTER + 1))
  OUTPUT_FILE="$OUTPUT_DIR/casualty_${casualty_id}_f${start_frame}-${end_frame}_fps${FPS_NAME}.mp4"

  if [[ "$FPS_SAME" -eq 1 ]]; then
    ffmpeg -nostdin -hide_banner -loglevel error -y \
      -ss "$start_time" -to "$end_time" -i "$VIDEO_PATH" \
      -c copy "$OUTPUT_FILE"
  else
    ffmpeg -nostdin -hide_banner -loglevel error -y \
      -ss "$start_time" -to "$end_time" -i "$VIDEO_PATH" \
      -r "$FPS_TARGET" -c:v libx264 -crf 18 -preset veryfast -c:a copy "$OUTPUT_FILE"
  fi

  PERCENT=$(awk -v c="$COUNTER" -v t="$TOTAL_SNIPPETS" 'BEGIN{printf "%.1f", (c/t)*100}')
  echo "[info] Progress: $COUNTER/$TOTAL_SNIPPETS (${PERCENT}%)"
done < "$RANGE_FILE"

echo "[info] Snippets saved to $OUTPUT_DIR"
