import torch
from diffusers import StableDiffusionPipeline, AutoencoderKL, UNet2DConditionModel, PNDMScheduler
from safetensors import safe_open
from safetensors.torch import load_file
import os
from PIL import Image
import numpy as np
from huggingface_hub import hf_hub_download
import re
import time

# 设置模型缓存路径
CACHE_DIR = os.path.expanduser('~/.cache/huggingface')
os.environ['HF_HOME'] = CACHE_DIR
os.environ['HF_HUB_CACHE'] = os.path.join(CACHE_DIR, 'hub')

# 确保缓存目录存在
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(os.path.join(CACHE_DIR, 'hub'), exist_ok=True)

MODEL_PATH = "/Users/hanxiang1/work/github_dev/pic_image_script/lora/cute/"
# 基础模型ID
BASE_MODEL_ID = "runwayml/stable-diffusion-v1-5"
# 基础模型本地路径，如果设置了该路径，将优先使用本地模型
BASE_MODEL_PATH = ''

def parse_lora_models(prompt):
    """从提示词中解析LoRA模型和权重"""
    pattern = r'<lora:([^:>]+):([0-9.]+)>'
    return re.findall(pattern, prompt)

def load_model(lora_models=None, model_path=None):
    """
    加载模型
    Args:
        lora_models: List of tuples containing (lora_name, lora_weight)
        model_path: 自定义基础模型路径，如果为None则使用BASE_MODEL_ID
    """
    # 确定模型来源
    model_source = model_path if model_path else (BASE_MODEL_PATH if BASE_MODEL_PATH else BASE_MODEL_ID)
    
    # 检查是否是本地模型文件路径
    is_local_model = os.path.exists(model_source)
    
    # 加载基础模型
    if is_local_model:
        print(f"正在加载本地模型: {model_source}")
        pipe = StableDiffusionPipeline.from_single_file(
            model_source,
            torch_dtype=torch.float32,
            load_safety_checker=False,
            local_files_only=True,
            custom_pipeline="stable_diffusion",
            use_safetensors=True,
            extract_ema=False,
            load_connected_pipeline=False,  # 防止下载额外的模型
            model_type="StableDiffusion"
        )
    else:
        print(f"正在从Hugging Face加载模型: {model_source}")
        pipe = StableDiffusionPipeline.from_pretrained(
            BASE_MODEL_ID,
            torch_dtype=torch.float32,
            safety_checker=None,
        requires_safety_checker=False,
        cache_dir='/Users/hanxiang1/.cache/huggingface/hub'
    )

    # 如果有指定的LORA模型，则加载
    if lora_models:
        for lora_name, lora_weight in lora_models:
            lora_file = f"{lora_name}.safetensors"
            if not os.path.exists(os.path.join(MODEL_PATH, lora_file)):
                print(f"警告: LORA模型 {lora_file} 未找到")
                continue
            
            pipe.load_lora_weights(MODEL_PATH, weight_name=lora_file)
            pipe.fuse_lora(lora_scale=float(lora_weight))
    
    return pipe

def generate_image(prompt, negative_prompt="", output_path="output.png", num_inference_steps=50, guidance_scale=7.5, model_path=None):
    """
    生成图片
    Args:
        prompt: 正向提示词
        negative_prompt: 负向提示词
        output_path: 输出图片路径
        num_inference_steps: 推理步数  速度优先：20~30 步（适合快速测试）。 质量优先：50~80 步（平衡质量与速度）。 超高精度：100+ 步（边际收益递减，耗时显著增加）
        guidance_scale: 提示词引导强度
        model_path: 自定义基础模型路径
    Returns:
        生成的图片路径
    """
    # 从提示词中解析LoRA模型
    lora_models = parse_lora_models(prompt)
    print(f"解析到LoRA模型: {lora_models}")
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print("正在加载模型...")
    try:
        pipe = load_model(lora_models, model_path)
        pipe = pipe.to("cpu")  # 如果有 GPU 可以改为 "cuda"
        
        # 移除LoRA标签后的提示词
        clean_prompt = re.sub(r'<lora:[^>]+>', '', prompt).strip()
        print(f"使用正向提示词: {clean_prompt}")
        if negative_prompt:
            print(f"使用负向提示词: {negative_prompt}")
        
        # 生成图片
        print("正在生成图片...")
        with torch.no_grad():
            image = pipe(
                clean_prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
            ).images[0]
        
        # 保存图片
        image.save(output_path)
        print(f"图片已保存到: {output_path}")
        
    except Exception as e:
        print(f"生成图片时出错: {str(e)}")
        raise
    
    return output_path

def generate_story_image(prompt, negative_prompt="", num_inference_steps=20, guidance_scale=7.5, model_path=None):
    """
    生成故事风格的图片API
    Args:
        prompt: 正向提示词
        negative_prompt: 负向提示词
        num_inference_steps: 推理步数
        guidance_scale: 提示词引导强度
        model_path: 自定义基础模型路径
    Returns:
        生成的图片路径
    """
    output_path = f'output/generated_image_{time.time()}.png'
    return generate_image(
        prompt=prompt + ",highly detailed,8k resolution,photorealistic,realistic,absurdres,background light,extremely detailed，8k",
        negative_prompt=negative_prompt,
        output_path=output_path,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        model_path=model_path
    )

if __name__ == "__main__":
    # 测试样例
    test_prompt = "(masterpiece:1.2), best quality,PIXIV, fairy tale style, 1girl, fish, eyes, long hair, turtle, smile, open mouth, shirt, solo, window<lora:fairy tale style-000016:0.7>"
    test_negative_prompt = "EasyNegative, badhandsv5-neg,Subtitles,word"
    
    try:
        generated_image_path = generate_story_image(
            prompt=test_prompt,
            negative_prompt=test_negative_prompt
        )
        print(f"图片已生成: {generated_image_path}")
    except Exception as e:
        print(f"发生错误: {str(e)}")
        raise