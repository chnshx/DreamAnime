import argparse
import os
import numpy as np
import torch
from diffusers import StableDiffusionPipeline
import joint_training 
from PIL import Image
import json

def sample(ckpt, delta_ckpt, from_file, prompts, compress, batch_size, freeze_model):
    model_id = ckpt
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")

    if delta_ckpt is not None:
        joint_training.load_model(pipe.text_encoder, pipe.tokenizer, pipe.unet, delta_ckpt, compress, freeze_model)
        outdir = os.path.dirname(delta_ckpt)
    
    if prompts is not None:
        for prompt in prompts:
            all_images = []
            print(prompt)
            for i in range(2):
                if args.negp != None:
                    images = pipe([prompt]*batch_size, num_inference_steps=200, guidance_scale=6., negative_prompt=[args.negp]*batch_size, eta=1.).images
                else:
                    images = pipe([prompt]*batch_size, num_inference_steps=200, guidance_scale=6., eta=1.).images
                all_images += images
            images = np.hstack([np.array(x) for x in images])
            images = Image.fromarray(images)
            name = '-'.join(prompt[:50].split())
            name = name.replace('<', '').replace('>', '')
            images.save(f'{outdir}/{name}.png')
            os.makedirs(f'{outdir}/{outdir}', exist_ok=True)
            os.makedirs(f'{outdir}/{outdir}/'+name, exist_ok=True)
            for i, im in enumerate(all_images):
                im.save(f'{outdir}/{outdir}/samples/{i}.jpg'.replace('samples', name))
    else:
        print(f"reading prompts from {from_file}")
        with open(from_file, "r") as f:
            prompts = json.load(f)

        for prompt in prompts:
            all_images = []
            print(prompt)
            for i in range(2):
                if args.negp != None:
                    images = pipe([prompt]*batch_size, num_inference_steps=200, guidance_scale=6., negative_prompt=[args.negp]*batch_size, eta=1.).images
                else:
                    images = pipe([prompt]*batch_size, num_inference_steps=200, guidance_scale=6., eta=1.).images
                all_images += images
            images = np.hstack([np.array(x) for x in images])
            images = Image.fromarray(images)
            # takes only first 50 characters of prompt to name the image file
            name = '-'.join(prompt[:50].split())
            name = name.replace('<', '').replace('>', '')
            images.save(f'{outdir}/{name}.png')
            os.makedirs(f'{outdir}/{outdir}', exist_ok=True)
            os.makedirs(f'{outdir}/{outdir}/'+name, exist_ok=True)
            for i, im in enumerate(all_images):
                im.save(f'{outdir}/{outdir}/samples/{i}.jpg'.replace('samples', name))


def parse_args():
    parser = argparse.ArgumentParser('', add_help=False)
    parser.add_argument('--ckpt', help='target string for query',
                        type=str)
    parser.add_argument('--delta_ckpt', help='target string for query', default=None,
                        type=str)
    parser.add_argument('--from_file', help='path to prompt file', default=None,
                        type=str)
    parser.add_argument('-p','--prompts', nargs='+', help='A prompt list', default=None)
    parser.add_argument("--compress", action='store_true')
    parser.add_argument("--batch_size", default=10, type=int)
    parser.add_argument('--freeze_model', help='crossattn or crossattn_kv', default='crossattn_kv',
                        type=str)
    parser.add_argument('--negp', type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    sample(args.ckpt, args.delta_ckpt, args.from_file, args.prompts, args.compress, args.batch_size, args.freeze_model)
