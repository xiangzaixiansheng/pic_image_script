import torch
from diffusers import StableDiffusionPipeline
from safetensors import safe_open
import os
from PIL import Image
import numpy as np
from huggingface_hub import hf_hub_download

# 设置模型缓存路径
os.environ['HF_HOME'] = '/Users/hanxiang1/.cache/huggingface'
os.environ['HF_HUB_CACHE'] = '/Users/hanxiang1/.cache/huggingface/hub'

MODEL_PATH = "/Users/hanxiang1/work/github_dev/pic_image_script/lora/cute/"
BASE_MODEL_ID = "runwayml/stable-diffusion-v1-5"

def load_model(model_file=None, lora_weight=0.5):
    """
    加载模型
    Args:
        model_file: LoRA模型文件路径，如果为None则使用默认路径
        lora_weight: LoRA模型权重，范围0-1，默认0.75
    """
    # 加载基础模型
    pipe = StableDiffusionPipeline.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.float32,
        safety_checker=None,
        requires_safety_checker=False,
        cache_dir='/Users/hanxiang1/.cache/huggingface/hub'
    )

    # 设置LoRA文件夹为模型文件夹
    pipe.load_lora_weights(MODEL_PATH)
    
    return pipe

def generate_image(prompt, output_path="output.png", num_inference_steps=20, guidance_scale=7.5):
    """
    生成图片
    Args:
        prompt: 文本提示词
        output_path: 输出图片路径
        num_inference_steps: 推理步数
        guidance_scale: 提示词引导强度
    Returns:
        生成的图片路径
    """
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print("正在加载模型...")
    try:
        pipe = load_model()
        pipe = pipe.to("cpu")  # 如果有 GPU 可以改为 "cuda"
        
        # 不再添加额外的触发词
        print(f"使用提示词: {prompt}")
        
        # 生成图片
        print("正在生成图片...")
        with torch.no_grad():
            image = pipe(
                prompt,
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

def main():
    # 测试样例
    test_prompt = "(masterpiece:1.2), best quality,PIXIV, fairy tale style, one girl, plant, rabbit, cloud, outdoors, mountain, solo, food, basket, grass, fruit, closed eyes, day, smile, sky, tree, dress, leaf, scenery <lora:fairy tale style-000016:0.75> "
    output_path = "outputs/generated_image.png"
    
    print(f"正在处理提示词: {test_prompt}")
    try:
        generated_image_path = generate_image(test_prompt, output_path)
    except Exception as e:
        print(f"发生错误: {str(e)}")
        raise
    
if __name__ == "__main__":
    main()