export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export CKPT_PATH=logs/JOJO
export DEVICE_IDS=0
CUDA_VISIBLE_DEVICES=$DEVICE_IDS python inference.py \
--delta_ckpt "${CKPT_PATH}/delta.bin" \
--ckpt $MODEL_NAME \
--prompts "a manga face of <new1> in the style of van Gogh" \
"a manga face of Taylor Swift in the style of <new4>"