import torch
from diffusers import StableDiffusionPipeline
from safetensors import safe_open
import os
from PIL import Image
import numpy as np
from huggingface_hub import hf_hub_download
import re
import time

# 设置模型缓存路径
os.environ['HF_HOME'] = '/Users/hanxiang1/.cache/huggingface'
os.environ['HF_HUB_CACHE'] = '/Users/hanxiang1/.cache/huggingface/hub'

MODEL_PATH = "/Users/hanxiang1/work/github_dev/pic_image_script/lora/cute/"
# 基础模型ID
BASE_MODEL_ID = "runwayml/stable-diffusion-v1-5"

def parse_lora_models(prompt):
    """从提示词中解析LoRA模型和权重"""
    pattern = r'<lora:([^:>]+):([0-9.]+)>'
    return re.findall(pattern, prompt)

def load_model(lora_models=None):
    """
    加载模型
    Args:
        lora_models: List of tuples containing (lora_name, lora_weight)
    """
    # 加载基础模型
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

def generate_image(prompt, negative_prompt="", output_path="output.png", num_inference_steps=50, guidance_scale=7.5):
    """
    生成图片
    Args:
        prompt: 正向提示词
        negative_prompt: 负向提示词
        output_path: 输出图片路径
        num_inference_steps: 推理步数  速度优先：20~30 步（适合快速测试）。 质量优先：50~80 步（平衡质量与速度）。 超高精度：100+ 步（边际收益递减，耗时显著增加）
        guidance_scale: 提示词引导强度
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
        pipe = load_model(lora_models)
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

def generate_story_image(prompt, negative_prompt="", num_inference_steps=50, guidance_scale=7.5):
    """
    生成故事风格的图片API
    Args:
        prompt: 正向提示词
        negative_prompt: 负向提示词
        num_inference_steps: 推理步数
        guidance_scale: 提示词引导强度
    Returns:
        生成的图片路径
    """
    output_path = f'outputs/generated_image_{time.time()}.png'
    return generate_image(
        prompt=prompt + ",highly detailed,8k resolution,photorealistic,realistic,absurdres,background light,extremely detailed，8k",
        negative_prompt=negative_prompt,
        output_path=output_path,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale
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