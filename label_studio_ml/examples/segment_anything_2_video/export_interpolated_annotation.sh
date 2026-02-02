#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: export_interpolated_annotation.sh --ls-url URL --ls-api-key TOKEN \
       --project PROJECT_ID --task TASK_ID --annotation ANNOTATION_ID [--output PATH] \
       [--snippets] [--snippets-dir DIR] [--person-id ID] [--min-frames N | --min-seconds S] \
       [--fps FPS]

Creates a Label Studio export snapshot with interpolated keyframes enabled,
waits for completion, downloads the JSON export, and saves only the specified
annotation (frame-wise) to disk. Requires curl, jq, and unzip (if the export is
returned as a ZIP archive). Use --snippets to also generate per-casualty video
snippets with matching bbox JSON outputs.
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
SNIPPETS=0
SNIPPETS_DIR=""
PERSON_ID=""
MIN_FRAMES=""
MIN_SECONDS=""
FPS_TARGET=""

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
    --snippets)
      SNIPPETS=1; shift;;
    --snippets-dir)
      SNIPPETS_DIR="$2"; shift 2;;
    --person-id)
      PERSON_ID="$2"; shift 2;;
    --min-frames)
      MIN_FRAMES="$2"; shift 2;;
    --min-seconds)
      MIN_SECONDS="$2"; shift 2;;
    --fps)
      FPS_TARGET="$2"; shift 2;;
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

if [[ -n "$MIN_FRAMES" && -n "$MIN_SECONDS" ]]; then
  echo "[error] --min-frames and --min-seconds are mutually exclusive." >&2
  exit 1
fi

if [[ "$DEBUG" -eq 1 ]]; then
  echo "[debug] Debug mode enabled" >&2
  set -x
fi

ensure_tool curl curl
ensure_tool jq jq
ensure_tool unzip unzip
if [[ "$SNIPPETS" -eq 1 ]]; then
  ensure_tool ffmpeg ffmpeg
  if [[ "$DEBUG" -eq 1 ]]; then
    ensure_tool ffprobe ffmpeg
  fi
fi

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

build_keep_frames_json() {
  local start_frame="$1"
  local end_frame="$2"
  local start_time="$3"
  local end_time="$4"
  local source_fps="$5"
  local target_fps="$6"
  local fps_same
  local frame_list

  fps_same=$(awk -v a="$target_fps" -v b="$source_fps" 'BEGIN{diff=a-b; if (diff<0) diff=-diff; if (diff < 0.000001) print 1; else print 0}')
  if [[ "$fps_same" -eq 1 ]]; then
    frame_list=$(awk -v s="$start_frame" -v e="$end_frame" 'BEGIN{for(i=s;i<=e;i++) print i}')
  else
    frame_list=$(awk -v s="$start_time" -v e="$end_time" -v fps_t="$target_fps" -v fps_s="$source_fps" -v sf="$start_frame" -v ef="$end_frame" '
      BEGIN{
        step=1/fps_t;
        for(t=s; t<=e+1e-9; t+=step){
          frame=int(t*fps_s+0.5);
          if(frame<sf) frame=sf;
          if(frame>ef) frame=ef;
          print frame;
        }
      }
    ')
  fi

  if [[ -z "$frame_list" ]]; then
    echo "[]"
    return
  fi

  printf '%s\n' "$frame_list" | jq -R -s 'split("\n") | map(select(length>0)) | map(tonumber)'
}

write_bbox_json() {
  local annotation_file="$1"
  local casualty_id="$2"
  local start_frame="$3"
  local end_frame="$4"
  local keep_frames_json="$5"
  local source_fps="$6"
  local output_file="$7"

  jq -e --arg casualty "$casualty_id" \
    --argjson start_frame "$start_frame" \
    --argjson end_frame "$end_frame" \
    --argjson keep_frames "$keep_frames_json" \
    --argjson fps "$source_fps" '
    def parse_id:
      (.meta.text // [])
      | map(select(type == "string" and test("id:[0-9]+")))
      | map(match("id:([0-9]+)").captures[0].string)
      | first;

    def result_items:
      if .source_type == "annotation" then
        .annotation.result // []
      elif .source_type == "prediction" then
        .prediction.result // []
      else [] end;

    def seq_entries($id):
      (result_items
        | map(select(.type == "videorectangle"))
        | map(select((parse_id) == $id))
        | map(.value.sequence // [] )
        | add) // [];

    (seq_entries($casualty)
      | sort_by(.frame)
      | map(select(.frame >= $start_frame and .frame <= $end_frame))) as $seq
    | ($seq | reduce .[] as $item ({}; . + {($item.frame|tostring): $item})) as $by_frame
    | ($keep_frames | map(tostring) | map($by_frame[.] // empty)) as $selected
    | reduce $selected[] as $item ({frames: [], idx: 1};
        .frames += [{
          original_frame: $item.frame,
          snippet_frame: .idx,
          time: (if $item.time != null then $item.time else ($item.frame / $fps) end),
          x: $item.x,
          y: $item.y,
          width: $item.width,
          height: $item.height,
          rotation: $item.rotation,
          score: $item.score,
          enabled: $item.enabled,
          auto: $item.auto
        }]
        | .idx += 1)
    | .frames
  ' "$annotation_file" > "$output_file"
}

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

generate_snippets() {
  local annotation_file="$1"
  local summary_file="$2"
  local summary_fps
  local video_url
  local full_video_url

  summary_fps=$(jq -r '.fps // empty' "$summary_file")
  video_url=$(jq -r '.video_url // empty' "$summary_file")
  if [[ -z "$summary_fps" || -z "$video_url" ]]; then
    echo "[error] Missing fps or video_url in summary JSON." >&2
    exit 1
  fi

  if [[ -z "$FPS_TARGET" ]]; then
    FPS_TARGET="$summary_fps"
  fi

  if [[ "$video_url" == http* ]]; then
    full_video_url="$video_url"
  elif [[ "$video_url" == /* ]]; then
    full_video_url="$BASE_URL$video_url"
  else
    full_video_url="$BASE_URL/$video_url"
  fi

  local video_path="$TMP_DIR/source_video"
  local video_ext="${video_url##*.}"
  if [[ -n "$video_ext" && "$video_ext" != "$video_url" ]]; then
    video_path="$video_path.$video_ext"
  else
    video_path="$video_path.mp4"
  fi

  echo "[info] Downloading video for snippets..." >&2
  curl -sS -L -o "$video_path" "$full_video_url" -H "${AUTH_HEADER[@]}"

  local min_frames_json="null"
  local min_seconds_json="null"
  if [[ -n "$MIN_FRAMES" ]]; then
    min_frames_json="$MIN_FRAMES"
  fi
  if [[ -n "$MIN_SECONDS" ]]; then
    min_seconds_json="$MIN_SECONDS"
  fi

  local range_file="$TMP_DIR/snippet_ranges.tsv"
  jq -r --arg person "$PERSON_ID" --argjson min_frames "$min_frames_json" --argjson min_seconds "$min_seconds_json" '
    (.casualties // {})
    | to_entries
    | (if $person != "" then map(select(.key == $person)) else . end)
    | .[]
    | .key as $id
    | (.value.ranges // [])
    | .[]
    | select(($min_frames == null) or ((.end_frame - .start_frame + 1) >= $min_frames))
    | select(($min_seconds == null) or ((.end_time - .start_time) >= $min_seconds))
    | [$id, .start_frame, .end_frame, .start_time, .end_time]
    | @tsv
  ' "$summary_file" > "$range_file"

  local total_snippets
  total_snippets=$(wc -l < "$range_file" | tr -d ' ')
  if [[ "$total_snippets" -eq 0 ]]; then
    echo "[info] No snippet ranges matched the filters." >&2
    return
  fi

  if [[ -z "$SNIPPETS_DIR" ]]; then
    SNIPPETS_DIR="$(pwd)/snippets_proj${PROJECT_ID}_task${TASK_ID}_ann${ANNOTATION_ID}_$(date -u +"%Y%m%dT%H%M%SZ")"
  fi
  mkdir -p "$SNIPPETS_DIR"

  local fps_name
  fps_name=$(awk -v fps="$FPS_TARGET" 'BEGIN{printf "%.0f", fps}')
  local fps_same
  fps_same=$(awk -v a="$FPS_TARGET" -v b="$summary_fps" 'BEGIN{diff=a-b; if (diff<0) diff=-diff; if (diff < 0.000001) print 1; else print 0}')

  cat <<EOF > "$SNIPPETS_DIR/README.txt"
Casualty snippet export

Generated at: $(date -u +"%Y-%m-%dT%H:%M:%SZ")
Project ID: $PROJECT_ID
Task ID: $TASK_ID
Annotation ID: $ANNOTATION_ID
Summary JSON: $summary_file
Video URL: $full_video_url
Person ID filter: ${PERSON_ID:-<all>}
Min frames: ${MIN_FRAMES:-<none>}
Min seconds: ${MIN_SECONDS:-<none>}
Target FPS: $FPS_TARGET
Output directory: $SNIPPETS_DIR
Total snippets: $total_snippets
Stream copy when FPS unchanged: true
EOF

  echo "[info] Generating $total_snippets snippet(s)..." >&2
  local counter=0
  while IFS=$'\t' read -r casualty_id start_frame end_frame start_time end_time; do
    counter=$((counter + 1))
    local output_video="$SNIPPETS_DIR/casualty_${casualty_id}_f${start_frame}-${end_frame}_fps${fps_name}.mp4"
    local output_json="$SNIPPETS_DIR/casualty_${casualty_id}_f${start_frame}-${end_frame}_fps${fps_name}.json"

    local keep_frames_json
    keep_frames_json=$(build_keep_frames_json "$start_frame" "$end_frame" "$start_time" "$end_time" "$summary_fps" "$FPS_TARGET")
    write_bbox_json "$annotation_file" "$casualty_id" "$start_frame" "$end_frame" "$keep_frames_json" "$summary_fps" "$output_json"

    if [[ "$DEBUG" -eq 1 ]]; then
      local dims
      dims=$(ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=p=0:s=x "$video_path")
      local width="${dims%x*}"
      local height="${dims#*x}"
      local frame_count
      frame_count=$(jq -r 'length' "$output_json")
      local chunk_size=1000
      if [[ "$frame_count" -le "$chunk_size" ]]; then
        local filter_file="$TMP_DIR/drawbox_${casualty_id}_${start_frame}_${end_frame}.txt"
        build_drawbox_filter "$output_json" "$width" "$height" "$filter_file" 0
        ffmpeg -nostdin -hide_banner -loglevel error -y \
          -ss "$start_time" -to "$end_time" -i "$video_path" \
          -r "$FPS_TARGET" -filter_complex_script "$filter_file" \
          -c:v libx264 -crf 18 -preset veryfast -c:a copy "$output_video"
      else
        local chunk_dir="$TMP_DIR/chunks_${casualty_id}_${start_frame}_${end_frame}"
        mkdir -p "$chunk_dir"
        local list_file="$chunk_dir/segments.txt"
        : > "$list_file"
        local start_idx=0
        local chunk_index=0
        while [[ "$start_idx" -lt "$frame_count" ]]; do
          local end_idx=$((start_idx + chunk_size))
          if [[ "$end_idx" -gt "$frame_count" ]]; then
            end_idx="$frame_count"
          fi
          local chunk_json="$chunk_dir/chunk_${chunk_index}.json"
          jq -c --argjson start_idx "$start_idx" --argjson stop_idx "$end_idx" 'to_entries | map(select(.key >= $start_idx) | select(.key < $stop_idx)) | map(.value)' "$output_json" > "$chunk_json"
          local chunk_start_time
          chunk_start_time=$(jq -r '.[0].time' "$chunk_json")
          local chunk_end_time
          chunk_end_time=$(jq -r '.[-1].time' "$chunk_json")
          local chunk_end_with_pad
          chunk_end_with_pad=$(awk -v e="$chunk_end_time" -v fps="$FPS_TARGET" 'BEGIN{printf "%.10f", e + (1/fps)}')
          local filter_file="$chunk_dir/drawbox_${chunk_index}.txt"
          build_drawbox_filter "$chunk_json" "$width" "$height" "$filter_file" "$start_idx"
          local segment_file="$chunk_dir/segment_${chunk_index}.mp4"
          ffmpeg -nostdin -hide_banner -loglevel error -y \
            -ss "$chunk_start_time" -to "$chunk_end_with_pad" -i "$video_path" \
            -r "$FPS_TARGET" -filter_complex_script "$filter_file" \
            -c:v libx264 -crf 18 -preset veryfast -c:a copy "$segment_file"
          printf "file '%s'\n" "$(cd "$(dirname "$segment_file")" && pwd)/$(basename "$segment_file")" >> "$list_file"
          start_idx="$end_idx"
          chunk_index=$((chunk_index + 1))
        done
        ffmpeg -nostdin -hide_banner -loglevel error -y \
          -f concat -safe 0 -i "$list_file" -c copy "$output_video"
      fi
    else
      if [[ "$fps_same" -eq 1 ]]; then
        ffmpeg -nostdin -hide_banner -loglevel error -y \
          -ss "$start_time" -to "$end_time" -i "$video_path" \
          -c copy "$output_video"
      else
        ffmpeg -nostdin -hide_banner -loglevel error -y \
          -ss "$start_time" -to "$end_time" -i "$video_path" \
          -r "$FPS_TARGET" -c:v libx264 -crf 18 -preset veryfast -c:a copy "$output_video"
      fi
    fi

    local percent
    percent=$(awk -v c="$counter" -v t="$total_snippets" 'BEGIN{printf "%.1f", (c/t)*100}')
    echo "[info] Progress: $counter/$total_snippets (${percent}%)"
  done < "$range_file"
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

  if [[ "$SNIPPETS" -eq 1 ]]; then
    generate_snippets "$OUTPUT_PATH" "$summary_path"
    echo "[info] Saved snippets to $SNIPPETS_DIR"
  fi
}

main "$@"
