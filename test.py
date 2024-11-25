'''
 * HoliSDiP: Image Super-Resolution via Holistic Semantics and Diffusion Prior 
 * Modified from SeeSR (https://github.com/cswry/SeeSR)
'''
import os
import sys
sys.path.append(os.getcwd())
import cv2
import glob
import argparse
import numpy as np
from PIL import Image

import torch
import torch.utils.checkpoint

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from transformers import CLIPTextModel, CLIPTokenizer, CLIPImageProcessor

from pipelines.pipeline_holisdip import StableDiffusionControlNetPipeline
from utils.misc import load_dreambooth_lora
from utils.wavelet_color_fix import wavelet_color_fix, adain_color_fix

from ram.models.ram_lora import ram
from ram import inference_ram as inference
from ram import get_transform

from models.gfm import SCM_encoder

from typing import Mapping, Any
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

############## import Mask2Former model ##############
import detectron2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
from detectron2.projects.deeplab import add_deeplab_config
ade20k_metadata = MetadataCatalog.get("ade20k_sem_seg_val")

from Mask2Former.mask2former import add_maskformer2_config
from utils.seg_class import ADE20K_150_CATEGORIES

cfg = get_cfg()
add_deeplab_config(cfg)
add_maskformer2_config(cfg)
cfg.merge_from_file("./preset/models/mask2former/config/ade20k-maskformer2_swin_large_IN21k_384_bs16_160k_res640.yaml")
cfg.MODEL.WEIGHTS = "./preset/models/mask2former/model_final_6b4a3a.pkl"
cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = True
predictor = DefaultPredictor(cfg)

ADE20k_COLORS = [k["color"] for k in ADE20K_150_CATEGORIES]
ADE20k_NAMES = [k["name"] for k in ADE20K_150_CATEGORIES]
######################################################

logger = get_logger(__name__, log_level="INFO")


tensor_transforms = transforms.Compose([
                transforms.ToTensor(),
            ])

ram_transforms = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

def load_state_dict_diffbirSwinIR(model: nn.Module, state_dict: Mapping[str, Any], strict: bool=False) -> None:
    state_dict = state_dict.get("state_dict", state_dict)
    
    is_model_key_starts_with_module = list(model.state_dict().keys())[0].startswith("module.")
    is_state_dict_key_starts_with_module = list(state_dict.keys())[0].startswith("module.")
    
    if (
        is_model_key_starts_with_module and
        (not is_state_dict_key_starts_with_module)
    ):
        state_dict = {f"module.{key}": value for key, value in state_dict.items()}
    if (
        (not is_model_key_starts_with_module) and
        is_state_dict_key_starts_with_module
    ):
        state_dict = {key[len("module."):]: value for key, value in state_dict.items()}
    
    model.load_state_dict(state_dict, strict=strict)


def load_pipeline(args, accelerator, enable_xformers_memory_efficient_attention):
    
    from models.controlnet import ControlNetModel
    from models.unet_2d_condition import UNet2DConditionModel

    # Load scheduler, tokenizer and models.
    scheduler = DDPMScheduler.from_pretrained(args.sd_model_path, subfolder="scheduler")
    text_encoder = CLIPTextModel.from_pretrained(args.sd_model_path, subfolder="text_encoder")
    tokenizer = CLIPTokenizer.from_pretrained(args.sd_model_path, subfolder="tokenizer")
    vae = AutoencoderKL.from_pretrained(args.sd_model_path, subfolder="vae")
    feature_extractor = CLIPImageProcessor.from_pretrained(f"{args.sd_model_path}/feature_extractor")
    unet = UNet2DConditionModel.from_pretrained(args.holisdip_model_path, subfolder="unet")
    controlnet = ControlNetModel.from_pretrained(args.holisdip_model_path, subfolder="controlnet")
    
    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    controlnet.requires_grad_(False)

    if enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
            controlnet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # Get the validation pipeline
    validation_pipeline = StableDiffusionControlNetPipeline(
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, feature_extractor=feature_extractor, 
        unet=unet, controlnet=controlnet, scheduler=scheduler, safety_checker=None, requires_safety_checker=False,
    )
    
    validation_pipeline._init_tiled_vae(encoder_tile_size=args.vae_encoder_tiled_size, decoder_tile_size=args.vae_decoder_tiled_size)

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move text_encode and vae to gpu and cast to weight_dtype
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    controlnet.to(accelerator.device, dtype=weight_dtype)

    return validation_pipeline

def load_tag_model(args, device='cuda'):
    
    model = ram(pretrained='preset/models/ram_swin_large_14m.pth',
                pretrained_condition=args.ram_ft_path,
                image_size=384,
                vit='swin_l')
    model.eval()
    model.to(device)
    
    return model
    
def get_validation_prompt(args, image, model, device='cuda'):
    validation_prompt = ""
 
    lq = tensor_transforms(image).unsqueeze(0).to(device)
    lq = ram_transforms(lq)
    res = inference(lq, model)
    ram_encoder_hidden_states = model.generate_image_embeds(lq)

    validation_prompt = f"{res[0]}, {args.prompt},"

    return validation_prompt, ram_encoder_hidden_states

def main(args, enable_xformers_memory_efficient_attention=True,):
    txt_path = os.path.join(args.output_dir, 'prompts')
    os.makedirs(txt_path, exist_ok=True)

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
    )

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the output folder creation
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(f'{args.output_dir}/masks/', exist_ok=True)
        os.makedirs(f'{args.output_dir}/masks_meta/', exist_ok=True)
        os.makedirs(f'{args.output_dir}/samples/', exist_ok=True)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("HoliSDiP")

    pipeline = load_pipeline(args, accelerator, enable_xformers_memory_efficient_attention)
    model = load_tag_model(args, accelerator.device)

    # load scm_encoder model
    scm_dim = 1024
    scm_encoder = SCM_encoder(input_nc=scm_dim).to(accelerator.device)
    scm_encoder.load_state_dict(torch.load(os.path.join(args.holisdip_model_path, "scm_encoder.pth")))
    scm_encoder.eval()

    # build the CLIP embedding of ADE20k categories
    max_length = 1 # the max length of the clip embedding of the category
    scm_dim = max_length*1024
    scm_list = torch.zeros(len(ADE20k_NAMES), scm_dim, device=accelerator.device)
    for i, name in enumerate(ADE20k_NAMES):
        class_token = pipeline.tokenizer(name, return_tensors="pt")
        class_token.input_ids = class_token.input_ids[0][1].unsqueeze(0) # only take the first token

        scm_list[i] = pipeline.text_encoder(class_token.input_ids.to(accelerator.device))[0].squeeze(0).view(-1)
    print(f"Finished building CLIP embeddings for ADE20k categories")
 
    if accelerator.is_main_process:
        generator = torch.Generator(device=accelerator.device)
        if args.seed is not None:
            generator.manual_seed(args.seed)

        if os.path.isdir(args.image_path):
            image_names = sorted(glob.glob(f'{args.image_path}/*.*'))
        else:
            image_names = [args.image_path]

        for image_idx, image_name in enumerate(image_names[:]):
            print(f'================== process {image_idx} imgs... ===================')
            validation_image = Image.open(image_name).convert("RGB")

            _, ram_encoder_hidden_states = get_validation_prompt(args, validation_image, model)

            ori_width, ori_height = validation_image.size
            resize_flag = False
            rscale = args.upscale
            if ori_width < args.process_size//rscale or ori_height < args.process_size//rscale:
                scale = (args.process_size//rscale)/min(ori_width, ori_height)
                tmp_image = validation_image.resize((int(scale*ori_width), int(scale*ori_height)))

                validation_image = tmp_image
                resize_flag = True

            validation_image = validation_image.resize((validation_image.size[0]*rscale, validation_image.size[1]*rscale))
            validation_image = validation_image.resize((validation_image.size[0]//8*8, validation_image.size[1]//8*8))
            width, height = validation_image.size
            resize_flag = True #

            print(f'input size: {height}x{width}')

            # load image for Mask2Former, which should be BGR format
            validation_image_cv2 = cv2.imread(image_name)
            # resize to 512x512
            validation_image_cv2 = cv2.resize(validation_image_cv2, (args.process_size, args.process_size))
            validation_prompt = ""
            
            # get mask from Mask2Former and save it along with the image
            with torch.no_grad():
                outputs = predictor(validation_image_cv2)
                label = outputs["sem_seg"].argmax(dim=0).to("cpu")
                v = Visualizer(validation_image_cv2[:, :, ::-1], ade20k_metadata, scale=1.2, instance_mode=ColorMode.IMAGE_BW)
                semantic_result = v.draw_sem_seg(label).get_image()
                semantic_result = cv2.cvtColor(semantic_result, cv2.COLOR_BGR2RGB)
                semantic_result = Image.fromarray(semantic_result)
                semantic_result.save(f'{args.output_dir}/masks_meta/{os.path.basename(image_name)}')

                scm = torch.zeros((args.process_size, args.process_size, scm_dim)).to(accelerator.device)
                seg_mask = torch.zeros((3, args.process_size, args.process_size))
                for i in torch.unique(label):
                    # build seg mask
                    color = ADE20k_COLORS[i]
                    seg_mask[0][label == i] = color[0]
                    seg_mask[1][label == i] = color[1]
                    seg_mask[2][label == i] = color[2]

                    # build scm 
                    scm[label == i] = scm_list[i]

                    # build text prompts
                    name = ADE20k_NAMES[i]
                    validation_prompt += f"{name}, "

                scm = scm.permute(2, 0, 1).unsqueeze(0)
                scm = scm_encoder(scm).to(accelerator.device)
                
                seg_mask = seg_mask.permute(1, 2, 0).numpy().astype(np.uint8)
                seg_mask = Image.fromarray(seg_mask)
                seg_mask.save(f'{args.output_dir}/masks/{os.path.basename(image_name)}')
                seg_mask = tensor_transforms(seg_mask).unsqueeze(0).to(accelerator.device)

            if args.added_prompt == "":
                validation_prompt = validation_prompt[:-2] # remove the last comma and space
            else:
                validation_prompt = validation_prompt
            validation_prompt += args.added_prompt # clean, extremely detailed, best quality, sharp, clean
            negative_prompt = args.negative_prompt #dirty, messy, low quality, frames, deformed, 

            if args.save_prompts:
                txt_save_path = f"{txt_path}/{os.path.basename(image_name).split('.')[0]}.txt"
                file = open(txt_save_path, "w")
                file.write(validation_prompt)
                file.close()
            print(f'{validation_prompt}')

            with torch.autocast("cuda"):
                image = pipeline(
                        validation_prompt, validation_image, seg_mask=seg_mask, scm=scm, num_inference_steps=args.num_inference_steps, generator=generator, height=height, width=width,
                        guidance_scale=args.guidance_scale, negative_prompt=negative_prompt, conditioning_scale=args.conditioning_scale,
                        start_point=args.start_point, ram_encoder_hidden_states=ram_encoder_hidden_states,
                        latent_tiled_size=args.latent_tiled_size, latent_tiled_overlap=args.latent_tiled_overlap,
                        args=args,
                    ).images[0]
            
            if args.align_method == 'nofix':
                image = image
            else:
                if args.align_method == 'wavelet':
                    image = wavelet_color_fix(image, validation_image)
                elif args.align_method == 'adain':
                    image = adain_color_fix(image, validation_image)

            if resize_flag: 
                image = image.resize((ori_width*rscale, ori_height*rscale))
                
            name, ext = os.path.splitext(os.path.basename(image_name))
            
            image.save(f'{args.output_dir}/samples/{name}.png')
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--holisdip_model_path", type=str, default=None)
    parser.add_argument("--ram_ft_path", type=str, default='preset/models/DAPE.pth')
    parser.add_argument("--sd_model_path", type=str, default='preset/models/stable-diffusion-2-base')
    parser.add_argument("--prompt", type=str, default="") # user can add self-prompt to improve the results
    parser.add_argument("--added_prompt", type=str, default="")
    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument("--image_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--mixed_precision", type=str, default="fp16") # no/fp16/bf16
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--conditioning_scale", type=float, default=1.0)
    parser.add_argument("--blending_alpha", type=float, default=1.0)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--process_size", type=int, default=512)
    parser.add_argument("--vae_decoder_tiled_size", type=int, default=224) # latent size, for 24G
    parser.add_argument("--vae_encoder_tiled_size", type=int, default=1024) # image size, for 13G
    parser.add_argument("--latent_tiled_size", type=int, default=96) 
    parser.add_argument("--latent_tiled_overlap", type=int, default=32) 
    parser.add_argument("--upscale", type=int, default=4)
    parser.add_argument("--seed", type=int, default=6666)
    parser.add_argument("--align_method", type=str, choices=['wavelet', 'adain', 'nofix'], default='adain')
    parser.add_argument("--start_steps", type=int, default=999) # defaults set to 999.
    parser.add_argument("--start_point", type=str, choices=['lr', 'noise'], default='lr') # LR Embedding Strategy, choose 'lr latent + 999 steps noise' as diffusion start point. 
    parser.add_argument("--save_prompts", action='store_true')
    args = parser.parse_args()
    main(args)



