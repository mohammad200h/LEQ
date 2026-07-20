SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

sudo xhost +local:root

sudo docker run --runtime=nvidia -it --name jepa_leq \
  --env-file "${SCRIPT_DIR}/.env" \
  -v "${SCRIPT_DIR}/..:/workspace/LEQ" \
  --net=host --privileged jepa_leq

