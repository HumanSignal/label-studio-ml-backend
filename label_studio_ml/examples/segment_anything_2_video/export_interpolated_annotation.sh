#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: export_interpolated_annotation.sh --ls-url URL --ls-api-key TOKEN \
       --project PROJECT_ID --task TASK_ID --annotation ANNOTATION_ID [--output PATH]

Creates a Label Studio export snapshot with interpolated keyframes enabled,
waits for completion, downloads the JSON export, and saves only the specified
annotation (frame-wise) to disk. Requires curl, jq, and unzip (if the export is
returned as a ZIP archive).
EOF
}

LS_URL=""
LS_API_KEY=""
PROJECT_ID=""
TASK_ID=""
ANNOTATION_ID=""
OUTPUT_PATH=""
POLL_INTERVAL=5
TIMEOUT=300
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
    --output)
      OUTPUT_PATH="$2"; shift 2;;
    --poll-interval)
      POLL_INTERVAL="$2"; shift 2;;
    --timeout)
      TIMEOUT="$2"; shift 2;;
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

if [[ -z "$LS_URL" || -z "$LS_API_KEY" || -z "$PROJECT_ID" || -z "$TASK_ID" || -z "$ANNOTATION_ID" ]]; then
  echo "Missing required arguments." >&2
  usage
  exit 1
fi

if [[ "$DEBUG" -eq 1 ]]; then
  echo "[debug] Debug mode enabled" >&2
  set -x
fi

ensure_tool curl curl
ensure_tool jq jq
ensure_tool unzip unzip

BASE_URL="${LS_URL%/}"
AUTH_HEADER=("Authorization: Token $LS_API_KEY")
JSON_HEADER=("Content-Type: application/json" "Accept: application/json")
TMP_PARENT="$(pwd)"
TMP_DIR="$(mktemp -d -p "$TMP_PARENT" export_tmp.XXXXXX)"
trap 'rm -rf "$TMP_DIR"' EXIT
EXPORT_TITLE="Interpolated Export proj${PROJECT_ID}_task${TASK_ID}_ann${ANNOTATION_ID}_$(date -u +"%Y%m%dT%H%M%SZ")"

create_export() {
  echo "[info] Creating export snapshot with interpolated keyframes..." >&2
  local payload
  payload=$(jq -n --arg title "$EXPORT_TITLE" '{title: $title, serialization_options: {interpolate_key_frames: true}}')
  local response
  response=$(curl -sS -X POST "$BASE_URL/api/projects/$PROJECT_ID/exports" \
    -H "${AUTH_HEADER[@]}" -H "${JSON_HEADER[0]}" -H "${JSON_HEADER[1]}" \
    --data "$payload")
  local export_id
  export_id=$(echo "$response" | jq -re '.id // .pk') || {
    echo "[error] Failed to parse export id from response: $response" >&2
    exit 1
  }
  echo "$export_id"
}

wait_for_export() {
  local export_id="$1"
  local deadline=$((SECONDS + TIMEOUT))
  echo "[info] Waiting for export $export_id to complete..."
  while (( SECONDS < deadline )); do
    local status_json
    status_json=$(curl -sS "$BASE_URL/api/projects/$PROJECT_ID/exports/$export_id" -H "${AUTH_HEADER[@]}")
    local status
    status=$(echo "$status_json" | jq -r '.status // .state // "unknown"')
    if [[ "$status" == "completed" ]]; then
      echo "[info] Export completed"
      return
    elif [[ "$status" == "failed" || "$status" == "error" ]]; then
      echo "[error] Export failed: $status_json" >&2
      exit 1
    fi
    sleep "$POLL_INTERVAL"
  done
  echo "[error] Timed out waiting for export $export_id" >&2
  exit 1
}

download_export() {
  local export_id="$1"
  local body_file="$TMP_DIR/export.bin"
  local headers_file="$TMP_DIR/headers.txt"
  curl -sS -D "$headers_file" -o "$body_file" \
    "$BASE_URL/api/projects/$PROJECT_ID/exports/$export_id/download?exportType=JSON" \
    -H "${AUTH_HEADER[@]}"

  local content_type
  content_type=$(grep -i '^content-type:' "$headers_file" | tail -n1 | awk '{print tolower($2)}' | tr -d '\r')

  if [[ "$content_type" == application/json* ]]; then
    echo "$body_file"
  elif [[ "$content_type" == application/zip* ]]; then
    local zip_json="$TMP_DIR/export.json"
    local first_entry
    first_entry=$(unzip -Z1 "$body_file" | head -n1)
    unzip -p "$body_file" "$first_entry" > "$zip_json"
    echo "$zip_json"
  else
    echo "[error] Unsupported content-type: $content_type" >&2
    exit 1
  fi
}

extract_annotation() {
  local json_file="$1"
  local output_file="$2"
  jq -e --arg project "$PROJECT_ID" --arg task "$TASK_ID" --arg entry_id "$ANNOTATION_ID" '
    def task_array:
      if type == "array" then .
      elif has("tasks") then .tasks
      elif has("data") then .data
      elif has("results") then .results
      else error("Unsupported export structure")
      end;

    def normalize_task:
      select(type == "object");

    def choose_task:
      task_array
        | map(normalize_task | select(((.id? // .task_id?) | tostring) == $task))
        | first;

    def normalize_entry:
      if type == "object" then
        { key: ((.id? // .annotation_id? // .prediction_id?) | tostring? ), value: . }
      elif (type == "number" or type == "string") then
        { key: (tostring), value: . }
      else empty end;

    def choose_entry($list):
      ($list // [])
      | (if type == "array" then . else [.] end)
      | map(normalize_entry)
      | map(select(.key != null))
      | map(select(.key == $entry_id))
      | first;

    (choose_task) as $task_obj
    | if $task_obj == null then error("Task not found in export") else . end
    | ($task_obj.data.video // $task_obj.data.video_url // $task_obj.data.videoUrl // $task_obj.data.video_path // $task_obj.data.videoPath // $task_obj.data.source // $task_obj.data.video_source) as $video_url
    | ($task_obj.data.fps) as $fps
    | (choose_entry($task_obj.annotations) as $ann
       | choose_entry($task_obj.predictions) as $pred
       | if $ann != null then
        {
          project_id: ($project|tonumber? // $project),
          task_id: ($task|tonumber? // $task),
          annotation_id: ($ann.key|tonumber? // $ann.key // ($entry_id|tonumber? // $entry_id)),
          source_type: "annotation",
          exported_at: (now | strftime("%Y-%m-%dT%H:%M:%SZ")),
          video_url: $video_url,
          fps: $fps,
          annotation: $ann.value
        }
      elif $pred != null then
        {
          project_id: ($project|tonumber? // $project),
          task_id: ($task|tonumber? // $task),
          prediction_id: ($pred.key|tonumber? // $pred.key // ($entry_id|tonumber? // $entry_id)),
          source_type: "prediction",
          exported_at: (now | strftime("%Y-%m-%dT%H:%M:%SZ")),
          video_url: $video_url,
          fps: $fps,
          prediction: $pred.value
        }
      else
        error("Annotation or prediction not found within task")
      end)
  ' "$json_file" > "$output_file"
}

generate_summary() {
  local source_file="$1"
  local summary_file="$2"
  jq -e '
    def parse_id:
      (.meta.text // [])
      | map(select(type == "string" and test("id:[0-9]+")))
      | map(match("id:([0-9]+)").captures[0].string)
      | first;

    def seq_ranges($seq):
      ($seq | sort_by(.frame)) as $sorted
      | if ($sorted | length) == 0 then [] else
          reduce $sorted[] as $item (
            {ranges: [], current: null};
            if .current == null then
              .current = {
                start_frame: $item.frame,
                end_frame: $item.frame,
                start_time: ($item.time // null),
                end_time: ($item.time // null)
              }
            else
              if ($item.frame == (.current.end_frame + 1)) then
                .current.end_frame = $item.frame
                | .current.end_time = ($item.time // .current.end_time)
              else
                .ranges += [.current]
                | .current = {
                  start_frame: $item.frame,
                  end_frame: $item.frame,
                  start_time: ($item.time // null),
                  end_time: ($item.time // null)
                }
              end
            end
          )
          | (.ranges + (if .current == null then [] else [.current] end))
        end;

    def result_items:
      if .source_type == "annotation" then
        .annotation.result // []
      elif .source_type == "prediction" then
        .prediction.result // []
      else [] end;

    {
      project_id,
      task_id,
      annotation_id,
      prediction_id,
      source_type,
      video_url,
      fps,
      casualties: (
        reduce (result_items | map(select(.type == "videorectangle")) | map(. + {casualty_id: (parse_id)}))[] as $item (
          {};
          if ($item.casualty_id == null) then
            .
          else
            .[$item.casualty_id] = (
              (.[$item.casualty_id] // []) + (seq_ranges($item.value.sequence // []))
            )
          end
        )
        | with_entries({key: .key, value: {ranges: .value}})
      )
    }
  ' "$source_file" > "$summary_file"
}

log_export_summary() {
  local json_file="$1"
  local target_task="$2"
  local target_entry="$3"
  echo "[debug] Export root structure:" >&2
  jq -r '
    def describe(obj):
      "type=" + (obj|type) +
      (if obj|type == "array" then ", length=" + ((obj|length)|tostring)
       else ", keys=" + ((obj|keys)|join(",")) end);
    describe(.)
  ' "$json_file" >&2 || true
  echo "[debug] First task-like entry:" >&2
  jq -r '
    def task_array:
      if type == "array" then .
      elif has("tasks") then .tasks
      elif has("data") then .data
      elif has("results") then .results
      else [] end;
    task_array | .[0] // {} | {id: .id, task_id: .task_id, keys: (keys // [])}
  ' "$json_file" >&2 || true

  echo "[debug] Sample annotations/predictions for target task $target_task:" >&2
  jq -r --arg task "$target_task" --arg entry "$target_entry" '
    def task_array:
      if type == "array" then .
      elif has("tasks") then .tasks
      elif has("data") then .data
      elif has("results") then .results
      else [] end;
    task_array
      | map(select((.id // .task_id | tostring) == $task))
      | first // {}
      | {
          task_id: (.id // .task_id),
          annotations_type: ((.annotations|type)?),
          first_annotation: ((.annotations // [])[0]),
          predictions_type: ((.predictions|type)?),
          first_prediction: ((.predictions // [])[0])
        }
  ' "$json_file" >&2 || true
}

main() {
  local export_id
  export_id=$(create_export)
  wait_for_export "$export_id"
  local download_path
  download_path=$(download_export "$export_id")
  if [[ "$DEBUG" -eq 1 ]]; then
    log_export_summary "$download_path" "$TASK_ID" "$ANNOTATION_ID"
  fi

  if [[ -z "$OUTPUT_PATH" ]]; then
    OUTPUT_PATH="$(pwd)/project${PROJECT_ID}_task${TASK_ID}_ann${ANNOTATION_ID}.json"
  fi
  mkdir -p "$(dirname "$OUTPUT_PATH")"
  extract_annotation "$download_path" "$OUTPUT_PATH"
  echo "[info] Saved filtered annotation to $OUTPUT_PATH"

  local summary_path
  if [[ "$OUTPUT_PATH" == *.json ]]; then
    summary_path="${OUTPUT_PATH%.json}.summary.json"
  else
    summary_path="${OUTPUT_PATH}.summary.json"
  fi
  generate_summary "$OUTPUT_PATH" "$summary_path"
  echo "[info] Saved summary to $summary_path"
}

main "$@"
