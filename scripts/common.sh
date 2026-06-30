#!/usr/bin/env bash
## @file common.sh
## @brief Shared shell helpers for host-wrapper and board-local workflows.
## @details This file centralizes runtime selection, validated vendor320
## workaround guards, camera auto-detection, and remote staging utilities so
## the individual scripts stay thin and consistent.

banana_demo_repo_root() {
  local source_path="${BASH_SOURCE[0]}"
  while [ -L "${source_path}" ]; do
    source_path="$(readlink "${source_path}")"
  done
  local script_dir
  script_dir="$(cd "$(dirname "${source_path}")" && pwd)"
  cd "${script_dir}/.." && pwd
}

## @brief Return whether the current process is running directly on the Banana.
banana_demo_is_board_mode() {
  if [[ "${BANANA_DEMO_EXEC_MODE:-}" == "board" ]]; then
    return 0
  fi
  [[ "$(uname -m)" == "riscv64" ]]
}

## @brief Resolve the repository root that board-local scripts should use.
banana_demo_board_root() {
  if [[ -n "${BOARD_DIR:-}" ]]; then
    printf '%s\n' "${BOARD_DIR}"
    return 0
  fi
  banana_demo_repo_root
}

banana_demo_host_target() {
  printf '%s\n' "${BANANA_SSH_TARGET:-svt@banana}"
}

banana_demo_host_board_dir() {
  printf '%s\n' "${BOARD_DIR:-/home/svt/banana-yolo11-spacemit-demo}"
}

## @brief Return the default visual model path for user-facing demos.
banana_demo_default_visual_model() {
  local repo_root="$1"
  printf '%s\n' "${repo_root}/models/generated/xquant_640/yolov11n_640x640.dynamic_int8.onnx"
}

banana_demo_default_visual_input_size() {
  printf '640\n'
}

## @brief Return the official vendor320 benchmark model path.
banana_demo_default_benchmark_model() {
  local repo_root="$1"
  printf '%s\n' "${repo_root}/models/vendor/yolo11/yolov11n_320x320.q.onnx"
}

## @brief Return the trusted fast-live model path used for responsive camera demos.
banana_demo_default_fast_live_model() {
  local repo_root="$1"
  printf '%s\n' "${repo_root}/models/vendor/yolo11/yolov11n_320x320.q.onnx"
}

## @brief Return the default square input size for the fast-live camera profile.
banana_demo_default_fast_live_input_size() {
  printf '320\n'
}

## @brief Return the preferred fast-live camera width in pixels.
banana_demo_default_fast_live_camera_width() {
  printf '640\n'
}

## @brief Return the preferred fast-live camera height in pixels.
banana_demo_default_fast_live_camera_height() {
  printf '480\n'
}

## @brief Return the preferred fast-live camera FPS request.
banana_demo_default_fast_live_camera_fps() {
  printf '60\n'
}

## @brief Return whether a model path refers to the public vendor320 bundle.
banana_demo_is_vendor320_model() {
  local model_path="${1:-}"
  [[ "$(basename "${model_path}")" == "yolov11n_320x320.q.onnx" ]]
}

## @brief Compute the SHA256 of a file using the best available local tool.
banana_demo_sha256_file() {
  local file_path="${1:-}"
  [[ -f "${file_path}" ]] || return 1
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "${file_path}" | awk '{print $1}'
    return 0
  fi
  if command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "${file_path}" | awk '{print $1}'
    return 0
  fi
  return 1
}

## @brief Check whether a hash matches the validated public vendor320 bundle.
banana_demo_is_validated_vendor320_model_hash() {
  local model_hash="${1:-}"
  case "${model_hash}" in
    558011431ba1cd26269af3694abc2ee2fc2d467d7fe043e10df78ed7449d9edc)
      return 0
      ;;
  esac
  return 1
}

## @brief Compute the official vendor320 model hash if the path matches the bundle name.
banana_demo_vendor320_model_hash() {
  local model_path="${1:-}"
  banana_demo_is_vendor320_model "${model_path}" || return 1
  banana_demo_sha256_file "${model_path}"
}

## @brief Return whether the model path matches the validated official vendor320 bundle.
banana_demo_is_validated_vendor320_model() {
  local model_path="${1:-}"
  local model_hash
  model_hash="$(banana_demo_vendor320_model_hash "${model_path}" 2>/dev/null || true)"
  [[ -n "${model_hash}" ]] || return 1
  banana_demo_is_validated_vendor320_model_hash "${model_hash}"
}

banana_demo_runtime_dir() {
  local repo_root="$1"
  local runtime_tag="$2"
  printf '%s\n' "${repo_root}/runtime/${runtime_tag}"
}

## @brief Map one runtime tag to the staged vendor runtime directory name.
banana_demo_runtime_vendor_dir_name() {
  local runtime_tag="$1"
  case "${runtime_tag}" in
    rt123)
      printf 'spacemit-ort.riscv64.1.2.3\n'
      ;;
    rt201)
      printf 'spacemit-ort.riscv64.2.0.1\n'
      ;;
    rt202b1)
      printf 'spacemit-ort.riscv64.2.0.2+beta1\n'
      ;;
    rt202)
      printf 'spacemit-ort.riscv64.2.0.2\n'
      ;;
    rt204)
      printf 'spacemit-ort.riscv64.2.0.4\n'
      ;;
    *)
      echo "ERROR: unsupported runtime tag=${runtime_tag}; use rt123|rt201|rt202b1|rt202|rt204" >&2
      return 2
      ;;
  esac
}

## @brief Return the host-side staged vendor runtime root for one runtime tag.
banana_demo_runtime_vendor_root() {
  local repo_root="$1"
  local runtime_tag="$2"
  local dir_name
  dir_name="$(banana_demo_runtime_vendor_dir_name "${runtime_tag}")"
  printf '%s\n' "${repo_root}/third_party/vendor/${dir_name}"
}

banana_demo_binary_path() {
  local repo_root="$1"
  local runtime_tag="$2"
  local candidate="${repo_root}/bin/banana_yolo11_demo_${runtime_tag}"
  if [[ -x "${candidate}" ]]; then
    printf '%s\n' "${candidate}"
    return 0
  fi
  printf '%s\n' "${repo_root}/bin/banana_yolo11_demo"
}

banana_demo_perf_test_path() {
  local repo_root="$1"
  local runtime_tag="$2"
  printf '%s\n' "${repo_root}/runtime/${runtime_tag}/bin/onnxruntime_perf_test"
}

banana_demo_runtime_tag_from_override() {
  local override="${BANANA_DEMO_RUNTIME_TAG:-auto}"
  case "${override}" in
    auto|"")
      return 1
      ;;
    rt123|1.2.3)
      printf 'rt123\n'
      return 0
      ;;
    rt201|2.0.1)
      printf 'rt201\n'
      return 0
      ;;
    rt202b1|2.0.2+beta1)
      printf 'rt202b1\n'
      return 0
      ;;
    rt202|2.0.2)
      printf 'rt202\n'
      return 0
      ;;
    rt204|2.0.4)
      printf 'rt204\n'
      return 0
      ;;
    *)
      echo "ERROR: unsupported BANANA_DEMO_RUNTIME_TAG=${override}; use auto|rt123|rt201|rt202b1|rt202|rt204" >&2
      return 2
      ;;
  esac
}

## @brief Resolve the runtime tag for the requested model and usage profile.
banana_demo_resolve_runtime_tag() {
  local model_path="$1"
  local runtime_profile="${2:-visual}"
  local override
  if override="$(banana_demo_runtime_tag_from_override)"; then
    printf '%s\n' "${override}"
    return 0
  fi

  if [[ "${runtime_profile}" == "perf" ]]; then
    printf 'rt201\n'
    return 0
  fi

  if banana_demo_is_validated_vendor320_model "${model_path}"; then
    printf 'rt123\n'
    return 0
  fi

  printf 'rt201\n'
}

banana_demo_join_colon_paths() {
  local out=""
  local item
  for item in "$@"; do
    [[ -n "${item}" ]] || continue
    [[ -d "${item}" ]] || continue
    if [[ -n "${out}" ]]; then
      out="${out}:${item}"
    else
      out="${item}"
    fi
  done
  printf '%s\n' "${out}"
}

## @brief Export runtime library paths for the selected staged runtime tree.
banana_demo_export_runtime_env() {
  local repo_root="$1"
  local runtime_tag="$2"
  local libs
  libs="$(banana_demo_join_colon_paths "$(banana_demo_runtime_dir "${repo_root}" "${runtime_tag}")/lib" "${repo_root}/opencv/lib")"
  if [[ -n "${libs}" ]]; then
    export LD_LIBRARY_PATH="${libs}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  fi
}

## @brief Apply the validated public rt201 visual workaround for vendor320.
## @details The workaround is intentionally SHA256-guarded so it only applies to
## the exact official vendor320 bundle that was validated during the forensic
## passes.
banana_demo_apply_vendor320_rt201_visual_fix() {
  local model_path="$1"
  local runtime_tag="$2"
  local runtime_profile="${3:-visual}"
  local mode="${BANANA_DEMO_VENDOR320_RT201_VISUAL_FIX:-auto}"
  local model_hash=""
  export BANANA_DEMO_VENDOR320_RT201_VISUAL_FIX_APPLIED=0

  banana_demo_is_vendor320_model "${model_path}" || return 0
  [[ "${runtime_tag}" == "rt201" ]] || return 0
  model_hash="$(banana_demo_vendor320_model_hash "${model_path}" 2>/dev/null || true)"

  case "${mode}" in
    0|off|false|disable|disabled)
      return 0
      ;;
    auto)
      [[ "${runtime_profile}" == "visual" ]] || return 0
      ;;
    1|on|true|enable|enabled)
      ;;
    *)
      echo "ERROR: unsupported BANANA_DEMO_VENDOR320_RT201_VISUAL_FIX=${mode}; use auto|0|1" >&2
      return 2
      ;;
  esac

  if ! banana_demo_is_validated_vendor320_model_hash "${model_hash}"; then
    if [[ "${mode}" == "auto" ]]; then
      echo "WARN: skipping vendor320 rt201 visual workaround because model identity is not the validated official bundle (sha256=${model_hash:-unknown})." >&2
      return 0
    fi
    echo "ERROR: refusing to apply vendor320 rt201 visual workaround to an unvalidated model (sha256=${model_hash:-unknown})." >&2
    return 2
  fi

  export SPACEMIT_EP_DISABLE_FLOAT16_EPILOGUE=1
  export SPACEMIT_EP_DISABLE_OP_NAME_FILTER="/model.23/Slice;/model.23/Slice_1;/model.23/Add_1;/model.23/Add_2;/model.23/Sub;/model.23/Sub_1"
  export BANANA_DEMO_VENDOR320_RT201_VISUAL_FIX_APPLIED=1
  echo "INFO: enabling vendor320 rt201 visual workaround for validated official model (sha256=${model_hash}) (disable float16 epilogue; keep /model.23 tail Slice/Add/Sub ops on CPU)." >&2
}

## @brief Restore local GUI session variables when display output is requested.
banana_demo_prepare_display_env() {
  local display_flag="${1:-0}"
  [[ "${display_flag}" == "1" ]] || return 0
  export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
  if [[ -z "${WAYLAND_DISPLAY:-}" && -S "${XDG_RUNTIME_DIR}/wayland-0" ]]; then
    export WAYLAND_DISPLAY="wayland-0"
  fi
  export DISPLAY="${DISPLAY:-:0}"
  if [[ -z "${XAUTHORITY:-}" ]]; then
    local auth
    for auth in "${XDG_RUNTIME_DIR}"/.mutter-Xwaylandauth.*; do
      [[ -f "${auth}" ]] || continue
      export XAUTHORITY="${auth}"
      break
    done
  fi
}

banana_demo_gui_env_available() {
  [[ -n "${DISPLAY:-}" || -n "${WAYLAND_DISPLAY:-}" ]]
}

banana_demo_gui_socket_available() {
  local runtime_dir="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
  [[ -S "${runtime_dir}/wayland-0" || -S /tmp/.X11-unix/X0 ]]
}

## @brief Resolve `display=auto|0|1` to a concrete display flag plus reason.
banana_demo_resolve_display_flag() {
  local requested="${1:-auto}"
  case "${requested}" in
    ""|auto)
      if banana_demo_gui_env_available; then
        printf '1\tgui-env\n'
      elif banana_demo_gui_socket_available; then
        printf '1\tgui-socket\n'
      else
        printf '0\tno-gui-env-or-socket\n'
      fi
      ;;
    1|on|true|enable|enabled)
      printf '1\tuser-forced-on\n'
      ;;
    0|off|false|disable|disabled)
      printf '0\tuser-forced-off\n'
      ;;
    *)
      echo "ERROR: unsupported DISPLAY_FLAG=${requested}; use auto|0|1" >&2
      return 2
      ;;
  esac
}

## @brief Resolve `headless=auto|0|1` based on the final display decision.
banana_demo_resolve_headless_flag() {
  local requested="${1:-auto}"
  local display_flag="${2:-0}"
  case "${requested}" in
    ""|auto)
      printf '%s\tauto-from-display\n' "$((display_flag == 1 ? 0 : 1))"
      ;;
    1|on|true|enable|enabled)
      printf '1\tuser-forced-on\n'
      ;;
    0|off|false|disable|disabled)
      printf '0\tuser-forced-off\n'
      ;;
    *)
      echo "ERROR: unsupported HEADLESS_FLAG=${requested}; use auto|0|1" >&2
      return 2
      ;;
  esac
}

## @brief Parse `/dev/videoN` into a numeric camera index.
banana_demo_parse_video_index() {
  local candidate="${1:-}"
  if [[ "${candidate}" =~ ^/dev/video([0-9]+)$ ]]; then
    printf '%s\n' "${BASH_REMATCH[1]}"
    return 0
  fi
  return 1
}

## @brief Resolve a camera argument to a concrete `/dev/videoN` path when possible.
banana_demo_resolve_camera_path() {
  local candidate="${1:-}"
  if [[ -z "${candidate}" || "${candidate}" == "auto" ]]; then
    return 1
  fi

  if [[ "${candidate}" =~ ^[0-9]+$ ]]; then
    printf '/dev/video%s\n' "${candidate}"
    return 0
  fi

  if [[ -e "${candidate}" ]]; then
    readlink -f "${candidate}"
    return 0
  fi

  printf '%s\n' "${candidate}"
}

## @brief Prefer stable V4L capture symlinks and then fall back to USB video nodes.
banana_demo_default_camera_path() {
  local candidate

  for candidate in /dev/v4l/by-id/*video-index0; do
    [[ -e "${candidate}" ]] || continue
    printf '%s\n' "${candidate}"
    return 0
  done

  for candidate in /dev/v4l/by-path/*video-index0; do
    [[ -e "${candidate}" ]] || continue
    printf '%s\n' "${candidate}"
    return 0
  done

  if command -v v4l2-ctl >/dev/null 2>&1; then
    local chosen
    chosen="$(
      v4l2-ctl --list-devices 2>/dev/null | awk '
        /^[^ \t].*:$/ {header=$0; next}
        /^[ \t]+\/dev\/video[0-9]+$/ {
          node=$1
          if (header ~ /(usb|USB)/ && chosen == "") {
            chosen=node
          }
        }
        END {
          if (chosen != "")
            print chosen
        }'
    )"
    if [[ -n "${chosen}" ]]; then
      printf '%s\n' "${chosen}"
      return 0
    fi
  fi

  for candidate in /dev/video*; do
    [[ -e "${candidate}" ]] || continue
    printf '%s\n' "${candidate}"
    return 0
  done

  return 1
}

## @brief Choose the best sane pixel format for the requested camera mode.
banana_demo_choose_camera_pixfmt() {
  local camera_path="${1:-}"
  local camera_width="${2:-1280}"
  local camera_height="${3:-720}"

  if ! command -v v4l2-ctl >/dev/null 2>&1; then
    printf 'auto\n'
    return 0
  fi

  local resolved
  resolved="$(banana_demo_resolve_camera_path "${camera_path}")" || {
    printf 'auto\n'
    return 0
  }

  local formats
  formats="$(v4l2-ctl -d "${resolved}" --list-formats-ext 2>/dev/null || true)"
  [[ -n "${formats}" ]] || {
    printf 'auto\n'
    return 0
  }

  local exact_size="${camera_width}x${camera_height}"
  if printf '%s\n' "${formats}" | awk -v fmt="MJPG" -v size="${exact_size}" '
      $0 ~ "\\047" fmt "\\047" {in_fmt=1; next}
      /^[ \t]*\\[[0-9]+\\]:/ {in_fmt=0}
      in_fmt && index($0, "Size: Discrete " size) {found=1}
      END {exit(found ? 0 : 1)}'; then
    printf 'mjpg\n'
    return 0
  fi
  if printf '%s\n' "${formats}" | awk -v fmt="YUYV" -v size="${exact_size}" '
      $0 ~ "\\047" fmt "\\047" {in_fmt=1; next}
      /^[ \t]*\\[[0-9]+\\]:/ {in_fmt=0}
      in_fmt && index($0, "Size: Discrete " size) {found=1}
      END {exit(found ? 0 : 1)}'; then
    printf 'yuyv\n'
    return 0
  fi
  if printf '%s\n' "${formats}" | grep -q "'MJPG'"; then
    printf 'mjpg\n'
    return 0
  fi
  if printf '%s\n' "${formats}" | grep -q "'YUYV'"; then
    printf 'yuyv\n'
    return 0
  fi
  printf 'auto\n'
}

## @brief Clear OMP/GOMP variables so strict benchmark runs stay reproducible.
banana_demo_unset_parallel_env() {
  unset OMP_NUM_THREADS OMP_PROC_BIND OMP_PLACES OMP_SCHEDULE OMP_MAX_ACTIVE_LEVELS OMP_NESTED OMP_STACKSIZE OMP_CANCELLATION OMP_DISPLAY_ENV
  unset GOMP_CPU_AFFINITY GOMP_STACKSIZE GOMP_SPINCOUNT
}

## @brief Resolve the canonical default input image from staged or board-local paths.
banana_demo_resolve_default_image() {
  local repo_root="$1"
  local candidates=(
    "${IMAGE_PATH:-}"
    "${BANANA_DEMO_IMAGE:-}"
    "${repo_root}/inputs/photo_2024-10-11_10-04-04.jpg"
    "/home/svt/ncnn-k1x-int8-smoke/models/photo_2024-10-11_10-04-04.jpg"
    "${HOME}/ncnn-k1x-int8-smoke/models/photo_2024-10-11_10-04-04.jpg"
  )
  local candidate
  for candidate in "${candidates[@]}"; do
    [[ -n "${candidate}" ]] || continue
    if [[ -f "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  return 1
}

## @brief Stage a local file into the deployed board tree when needed.
banana_demo_stage_remote_file() {
  local target="$1"
  local board_dir="$2"
  local local_path="$3"
  local remote_subdir="$4"
  if [[ -f "${local_path}" ]]; then
    ssh "${target}" "mkdir -p '${board_dir}/${remote_subdir}'"
    rsync -av "${local_path}" "${target}:${board_dir}/${remote_subdir}/" >&2
    printf '%s/%s/%s\n' "${board_dir}" "${remote_subdir}" "$(basename "${local_path}")"
    return 0
  fi
  printf '%s\n' "${local_path}"
}
