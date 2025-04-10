import os
import torch
import numpy as np
import re
from PIL import Image
from typing import Literal
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from comfy.model_management import get_torch_device
import folder_paths

from uno.flux.model import Flux
from uno.flux.modules.conditioner import HFEmbedder
from uno.flux.pipeline import UNOPipeline, preprocess_ref
from uno.flux.util import configs, load_ae, load_flow_model, set_lora, get_lora_rank
from uno.flux.modules.layers import DoubleStreamBlockLoraProcessor, SingleStreamBlockLoraProcessor, DoubleStreamBlockProcessor, SingleStreamBlockProcessor


def custom_set_lora(
    model: Flux,
    lora_rank: int,
    double_blocks_indices: list[int] | None = None,
    single_blocks_indices: list[int] | None = None,
    device: str | torch.device = "cpu",
) -> Flux:
    double_blocks_indices = list(range(model.params.depth)) if double_blocks_indices is None else double_blocks_indices
    single_blocks_indices = list(range(model.params.depth_single_blocks)) if single_blocks_indices is None \
                            else single_blocks_indices
    
    lora_attn_procs = {}
    with torch.device(device):
        for name, attn_processor in  model.attn_processors.items():
            match = re.search(r'\.(\d+)\.', name)
            if match:
                layer_index = int(match.group(1))

            if name.startswith("double_blocks") and layer_index in double_blocks_indices:
                lora_attn_procs[name] = DoubleStreamBlockLoraProcessor(dim=model.params.hidden_size, rank=lora_rank)
            elif name.startswith("single_blocks") and layer_index in single_blocks_indices:
                lora_attn_procs[name] = SingleStreamBlockLoraProcessor(dim=model.params.hidden_size, rank=lora_rank)
            else:
                lora_attn_procs[name] = attn_processor
    model.set_attn_processor(lora_attn_procs)
    return model

# 添加自定义加载模型的函数
def custom_load_flux_model(model_path, device, model_type="flux-dev", lora_rank=512, lora_path=None):
    """
    从指定路径加载 Flux 模型
    """
    from uno.flux.model import Flux
    from uno.flux.util import load_model
    
    # 获取对应模型类型的参数
    params = configs[model_type].params
    
    # 初始化模型
    with torch.device("meta" if model_path is not None else device):
        model = Flux(params)
    
    # 如果有lora，设置 LoRA 层
    if os.path.exists(lora_path):
        print(f"Using only_lora mode with rank: {lora_rank}")
        model = set_lora(model, lora_rank, device="meta" if model_path is not None else device)
    
    # 加载模型权重
    if model_path is not None:
        print(f"Loading Flux model from {model_path}")
        if model_path.endswith('safetensors'):
            from safetensors.torch import load_file as load_sft
            sd = load_sft(model_path, device=str(device))
        else:
            sd = load_model(model_path, device=str(device))
        
        # 检查是否有单独的 LoRA 文件
        if os.path.exists(lora_path):
            print(f"Found LoRA weights at {lora_path}, loading...")
            if lora_path.endswith('safetensors'):
                from safetensors.torch import load_file as load_sft
                lora_sd = load_sft(lora_path, device=str(device))
            else:
                lora_sd = torch.load(lora_path, map_location='cpu')
            # 合并 LoRA 权重
            sd.update(lora_sd)
        
        missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
        if len(missing) > 0:
            print(f"Missing keys: {len(missing)}")
        if len(unexpected) > 0:
            print(f"Unexpected keys: {len(unexpected)}")
        
        # 转移到目标设备
        model = model.to(str(device)).to(torch.bfloat16)
    return model

def custom_load_ae(ae_path, device, model_type="flux-dev"):
    """
    从指定路径加载自编码器
    """
    from uno.flux.modules.autoencoder import AutoEncoder
    from uno.flux.util import load_model
    
    # 获取对应模型类型的自编码器参数
    ae_params = configs[model_type].ae_params
    
    # 初始化自编码器
    with torch.device("meta" if ae_path is not None else device):
        ae = AutoEncoder(ae_params)
    
    # 加载自编码器权重
    if ae_path is not None:
        print(f"Loading AutoEncoder from {ae_path}")
        if ae_path.endswith('safetensors'):
            from safetensors.torch import load_file as load_sft
            sd = load_sft(ae_path, device=str(device))
        else:
            sd = torch.load(ae_path, map_location=str(device))
        missing, unexpected = ae.load_state_dict(sd, strict=False, assign=True)
        if len(missing) > 0:
            print(f"Missing keys: {len(missing)}")
        if len(unexpected) > 0:
            print(f"Unexpected keys: {len(unexpected)}")
        
        # 转移到目标设备
        ae = ae.to(str(device))
    return ae

def custom_load_t5(device: str | torch.device = "cuda", max_length: int = 512) -> HFEmbedder:
    # max length 64, 128, 256 and 512 should work (if your sequence is short enough)
    version = "xlabs-ai/xflux_text_encoders"
    cache_dir = folder_paths.get_folder_paths("clip")[0]
    return HFEmbedder(version, max_length=max_length, torch_dtype=torch.bfloat16).to(device)

def custom_load_clip(device: str | torch.device = "cuda") -> HFEmbedder:
    version = "openai/clip-vit-large-patch14"
    cache_dir = folder_paths.get_folder_paths("clip")[0]
    return HFEmbedder(version, max_length=77, torch_dtype=torch.bfloat16, cache_dir=cache_dir).to(device)



class UNOModelLoader:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "UNO_MODEL"
        self.loaded_model = None

    @classmethod
    def INPUT_TYPES(cls):
        # 获取 unet 模型列表和 vae 模型列表
        model_paths = folder_paths.get_filename_list("unet")
        vae_paths = folder_paths.get_filename_list("vae")
        
        # 增加 LoRA 模型选项
        lora_paths = folder_paths.get_filename_list("loras")
        
        return {
            "required": {
                "flux_model": (model_paths, ),
                "ae_model": (vae_paths, ),
                "model_type": (["flux-dev", "flux-dev-fp8", "flux-schnell"], {"default": "flux-dev"}),
                "offload": ("BOOLEAN", {"default": False}),
                "lora_model": (["None"] + lora_paths, ),
            }
        }

    RETURN_TYPES = ("UNO_MODEL",)
    RETURN_NAMES = ("uno_model",)
    FUNCTION = "load_model"
    CATEGORY = "UNO"

    def load_model(self, flux_model, ae_model, model_type, offload, lora_model=None):
        device = get_torch_device()
        
        try:
            # 获取模型文件的完整路径
            flux_model_path = folder_paths.get_full_path("unet", flux_model)
            ae_model_path = folder_paths.get_full_path("vae", ae_model)
            
            # 获取LoRA模型路径（如果有）
            lora_model_path = None
            if lora_model is not None and lora_model != "None":
                lora_model_path = folder_paths.get_full_path("loras", lora_model)
            
            print(f"Loading Flux model from: {flux_model_path}")
            print(f"Loading AE model from: {ae_model_path}")
            lora_rank = 512
            if lora_model_path:
                print(f"Loading LoRA model from: {lora_model_path}")
            
            # 创建自定义 UNO Pipeline
            class CustomUNOPipeline(UNOPipeline):
                def __init__(self, model_type, device, flux_path, ae_path, offload=False, 
                            lora_rank=512, lora_path=None):
                    self.device = device
                    self.offload = offload
                    self.model_type = model_type
                    
                    # 加载 CLIP 和 T5 编码器
                    self.clip = custom_load_clip(device="cpu" if offload else self.device)
                    self.t5 = custom_load_t5(device="cpu" if offload else self.device, max_length=512)
                    
                    # 加载自定义模型
                    self.ae = custom_load_ae(ae_path, device="cpu" if offload else self.device, model_type=model_type)
                    self.model = custom_load_flux_model(
                        flux_path, 
                        device="cpu" if offload else self.device, 
                        model_type=model_type,
                        lora_rank=lora_rank,
                        lora_path=lora_path
                    )
                    
            # 创建自定义 pipeline
            model = CustomUNOPipeline(
                model_type=model_type,
                device=device,
                flux_path=flux_model_path,
                ae_path=ae_model_path,
                offload=offload,
                lora_rank=lora_rank,
                lora_path=lora_model_path,
            )
            
            self.loaded_model = model
            print(f"UNO model loaded successfully with custom models.")
            return (model,)
        except Exception as e:
            print(f"Error loading UNO model: {e}")
            import traceback
            traceback.print_exc()
            raise e


class UNOGenerate:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        os.makedirs(self.output_dir, exist_ok=True)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "uno_model": ("UNO_MODEL",),
                "prompt": ("STRING", {"multiline": True}),
                "width": ("INT", {"default": 512, "min": 256, "max": 2048, "step": 16}),
                "height": ("INT", {"default": 512, "min": 256, "max": 2048, "step": 16}),
                "guidance": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "num_steps": ("INT", {"default": 25, "min": 1, "max": 100}),
                "seed": ("INT", {"default": 3407}),
                "pe": (["d", "h", "w", "o"], {"default": "d"}),
            },
            "optional": {
                "reference_image_1": ("IMAGE",),
                "reference_image_2": ("IMAGE",),
                "reference_image_3": ("IMAGE",),
                "reference_image_4": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "UNO"

    def generate(self, uno_model, prompt, width, height, guidance, num_steps, seed, pe, 
                reference_image_1=None, reference_image_2=None, reference_image_3=None, reference_image_4=None):
        # Make sure width and height are multiples of 16
        width = (width // 16) * 16
        height = (height // 16) * 16
        
        # Process reference images if provided
        ref_imgs = []
        ref_tensors = [reference_image_1, reference_image_2, reference_image_3, reference_image_4]
        for ref_tensor in ref_tensors:
            if ref_tensor is not None:
                # Convert from tensor to PIL
                if isinstance(ref_tensor, torch.Tensor):
                    # Handle batch of images
                    if ref_tensor.dim() == 4:  # [batch, height, width, channels]
                        for i in range(ref_tensor.shape[0]):
                            img = ref_tensor[i].cpu().numpy()
                            # Assume ComfyUI range is [-1, 1], convert to [0, 1]
                            img = (img + 1.0) / 2.0
                            ref_image_pil = Image.fromarray((img * 255).astype(np.uint8))
                            # Determine reference size based on number of reference images
                            ref_size = 512 if len([t for t in ref_tensors if t is not None]) <= 1 else 320
                            ref_image_pil = preprocess_ref(ref_image_pil, ref_size)
                            ref_imgs.append(ref_image_pil)
                    else:  # [height, width, channels]
                        img = ref_tensor.cpu().numpy()
                        # Assume ComfyUI range is [-1, 1], convert to [0, 1]
                        img = (img + 1.0) / 2.0
                        ref_image_pil = Image.fromarray((img * 255).astype(np.uint8))
                        # Determine reference size based on number of reference images
                        ref_size = 512 if len([t for t in ref_tensors if t is not None]) <= 1 else 320
                        ref_image_pil = preprocess_ref(ref_image_pil, ref_size)
                        ref_imgs.append(ref_image_pil)
                elif isinstance(ref_tensor, np.ndarray):
                    # Assume ComfyUI range is [-1, 1], convert to [0, 1]
                    img = (ref_tensor + 1.0) / 2.0
                    ref_image_pil = Image.fromarray((img * 255).astype(np.uint8))
                    # Determine reference size based on number of reference images
                    ref_size = 512 if len([t for t in ref_tensors if t is not None]) <= 1 else 320
                    ref_image_pil = preprocess_ref(ref_image_pil, ref_size)
                    ref_imgs.append(ref_image_pil)
        
        try:
            # Generate image
            output_img = uno_model(
                prompt=prompt,
                width=width,
                height=height,
                guidance=guidance,
                num_steps=num_steps,
                seed=seed,
                ref_imgs=ref_imgs,
                pe=pe
            )
            
            # Save the generated image
            output_filename = f"uno_{seed}_{prompt[:20].replace(' ', '_')}.png"
            output_path = os.path.join(self.output_dir, output_filename)
            
            # Convert to ComfyUI-compatible tensor
            if hasattr(output_img, 'images') and len(output_img.images) > 0:
                # Handle FluxPipelineOutput
                output_img.images[0].save(output_path)
                print(f"Saved UNO generated image to {output_path}")
                image = np.array(output_img.images[0]) / 255.0  # Convert to [0, 1]
            else:
                # Handle PIL Image
                output_img.save(output_path)
                print(f"Saved UNO generated image to {output_path}")
                image = np.array(output_img) / 255.0  # Convert to [0, 1]
            
            # Convert numpy array to torch.Tensor
            image = torch.from_numpy(image).float()
            
            # Make sure it's in ComfyUI format [batch, height, width, channels]
            if image.dim() == 3:  # [height, width, channels]
                image = image.unsqueeze(0)  # Add batch dimension to make it [1, height, width, channels]
            
            # Scale to [-1, 1] range as expected by ComfyUI
            image = image * 2.0 - 1.0
            
            return (image,)
        except Exception as e:
            print(f"Error generating image with UNO: {e}")
            raise e


# Register our nodes to be used in ComfyUI
NODE_CLASS_MAPPINGS = {
    "UNOModelLoader": UNOModelLoader,
    "UNOGenerate": UNOGenerate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UNOModelLoader": "UNO Model Loader",
    "UNOGenerate": "UNO Generate",
}
