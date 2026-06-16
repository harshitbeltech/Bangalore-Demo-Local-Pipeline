#!/usr/bin/env bash
# Restart the Triton server in its docker container with ALL model repos the
# bangalore_demo needs (yolo_infer, feature_extractor, number_yolo, ocr_model_vit,
# seatbelt/helmet/phone/hsrp/side_view/uncovered, ...).
#
# Needs sudo (docker requires it on this box). Run it yourself:
#     bash start_triton.sh            # will prompt for your sudo password
#
# Reconstructed from cv_pipeline/run_pipeline_new.py:start_triton().
set -uo pipefail

IMAGE="bd3db511ae81"                         # tritonserver:24.09-py3-pillow
MODELS="/home/cv-gpu-2/triton_model/models"
NAME="triton_bangalore"

echo ">> removing any stale container named ${NAME}..."
sudo docker rm -f "${NAME}" 2>/dev/null || true
sleep 2

echo ">> starting container (detached, sleep infinity)..."
CID=$(sudo docker run --gpus=all -d --name "${NAME}" \
  --shm-size=5g \
  --security-opt seccomp=unconfined --security-opt apparmor=unconfined \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v "${MODELS}:/models" \
  "${IMAGE}" sleep infinity) || { echo "docker run failed"; exit 1; }
echo "   container: ${CID}"

echo ">> installing opencv-python-headless in container (as the original does)..."
sudo docker exec "${CID}" pip install -q opencv-python-headless || true

echo ">> launching tritonserver (logs -> container:/tmp/triton.log)..."
sudo docker exec -d "${CID}" bash -c \
  "tritonserver \
   --model-repository=/models/yolo_v11/ \
   --model-repository=/models/ds/ \
   --model-repository=/models/LPR_2/ \
   --model-repository=/models/models_v2/conversion_scripts/triton_repo_v2/ \
   --model-repository=/models/hsrp/ \
   >/tmp/triton.log 2>&1"

echo ">> waiting for Triton HTTP health (up to 120s)..."
for i in $(seq 1 60); do
  if curl -sf http://localhost:8000/v2/health/ready >/dev/null 2>&1; then
    echo "   Triton READY ✅"
    exit 0
  fi
  sleep 2
done
echo "   Triton NOT ready after 120s. Inspect with:"
echo "     sudo docker exec ${NAME} tail -50 /tmp/triton.log"
exit 1
