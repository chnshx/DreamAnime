# This is for training with one GPUs
# export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export CKPT_PATH=./logs/JOJO
export DEVICE_IDS=0
CUDA_VISIBLE_DEVICES=$DEVICE_IDS accelerate launch \
joint_training.py \
          --pretrained_model_name_or_path=$MODEL_NAME \
          --id_list=./templates/id_list_jojo.json \
          --template=./templates/template.json \
          --output_dir=$CKPT_PATH \
          --resolution=512 \
          --train_batch_size=1 \
          --learning_rate=1e-5 \
          --lr_warmup_steps=0 \
          --max_train_steps=2250 \
          --scale_lr --hflip \
          --modifier_token "<new1>+<new2>+<new3>+<new4>" \
          --disentangle \
          --divided_steps \
          --id_rate 0.5 \
          --style_aug 