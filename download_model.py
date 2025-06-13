# 必须在所有 import 之前设置环境变量
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cache_dir = os.path.join(project_root, ".cache", "huggingface")
os.environ["HF_HOME"] = cache_dir
os.environ["TRANSFORMERS_CACHE"] = os.path.join(cache_dir, "transformers")
os.environ["HF_DATASETS_CACHE"] = os.path.join(cache_dir, "datasets")

from modelscope.hub.snapshot_download import snapshot_download

# 指定本地保存目录
project_root = os.path.dirname(os.path.abspath(__file__))
save_dir = os.path.join(project_root, "model")

# 下载模型到指定目录
path = snapshot_download('maple775885/RMBG-2.0', cache_dir=save_dir)
print("模型已下载到：", path)

print(path)