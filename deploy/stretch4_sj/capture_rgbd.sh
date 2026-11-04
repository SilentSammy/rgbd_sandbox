#!/usr/bin/env bash
set -e
sj_capture_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
case " $* " in
    *" --help "*|*" -h "*) ;;
    *)
        sj_calibration="${HELLO_FLEET_PATH:-}/${HELLO_FLEET_ID:-}/calibration_cameras/camera_extrinsics.yaml"
        if [[ ! -f "$sj_calibration" ]]; then
            printf 'Cannot capture RGB-D: missing factory calibration %s\nAsk the owner to restore calibration for this robot.\n' "$sj_calibration" >&2
            exit 2
        fi
        ;;
esac
cd "$sj_capture_root/stretch4_rgbd"
exec "$sj_capture_root/python.sh" scripts/capture_emulated_rgbd.py "$@"
