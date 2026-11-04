# Source this file: source /home/hello-robot/stretch_clean_ws/sj/activate.sh
sj_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# The old terminal workaround selects an OpenCV too old for the RGB-D API.
# Preserve the ROS overlays while removing only that manually added path.
IFS=: read -r -a sj_paths <<< "${PYTHONPATH:-}"
sj_pythonpath=""
for sj_path in "${sj_paths[@]}"; do
    if [[ -n "$sj_path" && "$sj_path" != /usr/lib/python3/dist-packages ]]; then
        sj_pythonpath="${sj_pythonpath:+$sj_pythonpath:}$sj_path"
    fi
done
export PYTHONPATH="$sj_pythonpath"
source "$sj_root/venv/bin/activate"
unset sj_root sj_paths sj_pythonpath sj_path
