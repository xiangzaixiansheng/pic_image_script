import os
import sys
import glob
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torchvision import transforms

# 必须在所有 import 之前设置环境变量
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cache_dir = os.path.join(project_root, ".cache", "huggingface")
os.environ["HF_HOME"] = cache_dir
os.environ["TRANSFORMERS_CACHE"] = os.path.join(cache_dir, "transformers")
os.environ["HF_DATASETS_CACHE"] = os.path.join(cache_dir, "datasets")

from modelscope import AutoModelForImageSegmentation
import datetime

import glob

def add_local_transformers_modules_to_sys_path():
    """
    prod 环境下：
    1. 检查 /mnt/workspace/.cache/huggingface/modules/transformers_modules 是否存在，不存在则将项目 .cache 拷贝过去（适配 Docker 环境）。
    """
    if os.getenv("ENV") == "prod":
        import shutil
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        local_cache_dir = os.path.join(project_root, ".cache")
        docker_cache_root = "/mnt/workspace/.cache"
        if not os.path.exists(docker_cache_root, "huggingface"):
            try:
                shutil.copytree(local_cache_dir, docker_cache_root, dirs_exist_ok=True)
                print(f"已自动将项目 .cache 拷贝到 /mnt/workspace/.cache 适配 Docker 环境")
            except Exception as e:
                print(f"拷贝 .cache 到 Docker 失败: {e}")

add_local_transformers_modules_to_sys_path()


def remove_background(input_image_path):
    if os.getenv("ENV") == "prod":
        # Docker prod 环境下，模型路径指向 /mnt/workspace/.cache/modelscope/damo/RMBG-2.0
        model_path = "/mnt/workspace/.cache/modelscope/maple775885/RMBG-2.0"
    else:
        model_path = "maple775885/RMBG-2.0"
    model = AutoModelForImageSegmentation.from_pretrained(model_path, trust_remote_code=True)
    torch.set_float32_matmul_precision(['high', 'highest'][0])
    model.to('cpu')
    model.eval()

    # Data settings
    image_size = (1024, 1024)
    transform_image = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image = Image.open(input_image_path).convert('RGB')
    input_images = transform_image(image).unsqueeze(0).to('cpu')

    # Prediction
    with torch.no_grad():
        preds = model(input_images)[-1].sigmoid().cpu()
    pred = preds[0].squeeze()
    pred_pil = transforms.ToPILImage()(pred)
    mask = pred_pil.resize(image.size)
    image.putalpha(mask)

    # Save output
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, '../output')
    os.makedirs(output_dir, exist_ok=True)
    filename = f'result_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
    output_path = os.path.join(output_dir, filename)
    image.save(output_path)

    return output_path

if __name__ == '__main__':
    input_image_path = "input/5a8a069aee5041a4b470671460fbc03f.png"
    output_path = remove_background(input_image_path)
    print(f'Saved result to: {output_path}')