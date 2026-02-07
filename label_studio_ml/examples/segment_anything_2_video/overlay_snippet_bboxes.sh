#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: overlay_snippet_bboxes.sh --snippet PATH --bbox-json PATH [--output PATH] [--chunk-size N]

Overlays bounding boxes from a snippet bbox JSON onto an existing snippet video.
Requires ffmpeg, ffprobe, and jq.
EOF
}

SNIPPET_PATH=""
BBOX_JSON=""
OUTPUT_PATH=""
CHUNK_SIZE=1000
APT_UPDATED=0

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
    --snippet)
      SNIPPET_PATH="$2"; shift 2;;
    --bbox-json)
      BBOX_JSON="$2"; shift 2;;
    --output)
      OUTPUT_PATH="$2"; shift 2;;
    --chunk-size)
      CHUNK_SIZE="$2"; shift 2;;
    -h|--help)
      usage; exit 0;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1;;
  esac
done

if [[ -z "$SNIPPET_PATH" || -z "$BBOX_JSON" ]]; then
  echo "Missing required arguments." >&2
  usage
  exit 1
fi

if [[ ! -f "$SNIPPET_PATH" ]]; then
  echo "[error] Snippet not found: $SNIPPET_PATH" >&2
  exit 1
fi

if [[ ! -f "$BBOX_JSON" ]]; then
  echo "[error] BBox JSON not found: $BBOX_JSON" >&2
  exit 1
fi

ensure_tool jq jq
ensure_tool ffmpeg ffmpeg
ensure_tool ffprobe ffmpeg

if [[ -z "$OUTPUT_PATH" ]]; then
  base="${SNIPPET_PATH%.*}"
  OUTPUT_PATH="${base}_bbox_overlaid.mp4"
fi

DIMENSIONS=$(ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=p=0:s=x "$SNIPPET_PATH")
WIDTH="${DIMENSIONS%x*}"
HEIGHT="${DIMENSIONS#*x}"
FPS_RAW=$(ffprobe -v error -select_streams v:0 -show_entries stream=avg_frame_rate -of csv=p=0 "$SNIPPET_PATH")
if [[ -z "$FPS_RAW" ]]; then
  echo "[error] Unable to read FPS from snippet." >&2
  exit 1
fi
FPS_VALUE=$(awk -v fps="$FPS_RAW" 'BEGIN{split(fps,a,"/"); if (a[2] == "") print a[1]; else printf "%.6f", a[1]/a[2]}')

FRAME_COUNT=$(jq -r 'length' "$BBOX_JSON")
if [[ "$FRAME_COUNT" -eq 0 ]]; then
  echo "[error] Empty bbox JSON." >&2
  exit 1
fi

TMP_PARENT="$(pwd)"
TMP_DIR="$(mktemp -d -p "$TMP_PARENT" overlay_tmp.XXXXXX)"
trap 'rm -rf "$TMP_DIR"' EXIT

build_drawbox_filter() {
  local bbox_json="$1"
  local width="$2"
  local height="$3"
  local output_file="$4"
  local frame_offset="${5:-0}"

  jq -r --argjson width "$width" --argjson height "$height" --argjson offset "$frame_offset" '
    def clamp($v; $min; $max): if $v < $min then $min elif $v > $max then $max else $v end;
    .[]
    | {
        x: ((.x * $width / 100.0) | floor),
        y: ((.y * $height / 100.0) | floor),
        w: ((.width * $width / 100.0) | floor),
        h: ((.height * $height / 100.0) | floor),
        n: (.snippet_frame - 1 - $offset)
      }
    | "drawbox=x=" + (clamp(.x;0;$width)|tostring)
      + ":y=" + (clamp(.y;0;$height)|tostring)
      + ":w=" + (clamp(.w;1;$width)|tostring)
      + ":h=" + (clamp(.h;1;$height)|tostring)
      + ":color=red@0.6:t=2:enable=eq(n\\," + (.n|tostring) + ")"
  ' "$bbox_json" | paste -sd ',' - > "$output_file"
}

if [[ "$FRAME_COUNT" -le "$CHUNK_SIZE" ]]; then
  FILTER_FILE="$TMP_DIR/drawbox.txt"
  build_drawbox_filter "$BBOX_JSON" "$WIDTH" "$HEIGHT" "$FILTER_FILE" 0
  ffmpeg -nostdin -hide_banner -loglevel error -y \
    -i "$SNIPPET_PATH" \
    -filter_complex_script "$FILTER_FILE" \
    -c:v libx264 -crf 18 -preset veryfast -c:a copy "$OUTPUT_PATH"
else
  CHUNK_DIR="$TMP_DIR/chunks"
  mkdir -p "$CHUNK_DIR"
  LIST_FILE="$CHUNK_DIR/segments.txt"
  : > "$LIST_FILE"
  start_idx=0
  chunk_index=0
  while [[ "$start_idx" -lt "$FRAME_COUNT" ]]; do
    end_idx=$((start_idx + CHUNK_SIZE))
    if [[ "$end_idx" -gt "$FRAME_COUNT" ]]; then
      end_idx="$FRAME_COUNT"
    fi
    chunk_json="$CHUNK_DIR/chunk_${chunk_index}.json"
    jq -c --argjson start_idx "$start_idx" --argjson stop_idx "$end_idx" 'to_entries | map(select(.key >= $start_idx) | select(.key < $stop_idx)) | map(.value)' "$BBOX_JSON" > "$chunk_json"
    chunk_start_time=$(awk -v s="$start_idx" -v fps="$FPS_VALUE" 'BEGIN{printf "%.10f", s / fps}')
    chunk_end_time=$(awk -v e="$end_idx" -v fps="$FPS_VALUE" 'BEGIN{printf "%.10f", e / fps}')
    filter_file="$CHUNK_DIR/drawbox_${chunk_index}.txt"
    build_drawbox_filter "$chunk_json" "$WIDTH" "$HEIGHT" "$filter_file" "$start_idx"
    segment_file="$CHUNK_DIR/segment_${chunk_index}.mp4"
    ffmpeg -nostdin -hide_banner -loglevel error -y \
      -ss "$chunk_start_time" -to "$chunk_end_time" -i "$SNIPPET_PATH" \
      -filter_complex_script "$filter_file" \
      -c:v libx264 -crf 18 -preset veryfast -c:a copy "$segment_file"
    printf "file '%s'\n" "$(cd "$(dirname "$segment_file")" && pwd)/$(basename "$segment_file")" >> "$LIST_FILE"
    start_idx="$end_idx"
    chunk_index=$((chunk_index + 1))
  done
  ffmpeg -nostdin -hide_banner -loglevel error -y \
    -f concat -safe 0 -i "$LIST_FILE" -c copy "$OUTPUT_PATH"
fi

echo "[info] Wrote overlay snippet to $OUTPUT_PATH"
