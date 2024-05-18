# DreamAnime: Learning Style-Identity Textual Disentanglement for Anime and Beyond
### Official implementation of "[DreamAnime: Learning Style-Identity Textual Disentanglement for Anime and Beyond](https://ieeexplore.ieee.org/abstract/document/10521816)" at IEEE TVCG.
#### Chenshu Xu, Yangyang Xu, Huaidong Zhang, Xuemiao Xu, and Shengfeng He
<div>
<p align="center">
<img src='assets/teaser.png' align="center" width=800>
</p>

## Getting Started
```
git clone https://github.com/chnshx/DreamAnime.git
cd DreamAnime
```
This implementation uses the requirements below and may also support future versions.
```
pip install accelerate==0.16.0
pip install modelcards
pip install transformers==4.26.1
pip install deepspeed
pip install diffusers==0.11.0
```
Finally,
```
accelerate config
```
## Style-Identity Joint Training
We suggest training with 750 × n steps, where n denotes the number of distinct identities. 
```
bash launch.sh
```
## Inference
```
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export CKPT_PATH=logs/JOJO
export DEVICE_IDS=0
CUDA_VISIBLE_DEVICES=$DEVICE_IDS python inference.py \
--delta_ckpt "${CKPT_PATH}/delta.bin" \
--ckpt $MODEL_NAME \
--prompts "a manga face of <new1> in the style of van Gogh" \
"a manga face of Taylor Swift in the style of <new4>"
```
