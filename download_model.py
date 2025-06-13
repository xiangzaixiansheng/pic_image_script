from modelscope.hub.snapshot_download import snapshot_download
import os

# 指定本地保存目录
project_root = os.path.dirname(os.path.abspath(__file__))
save_dir = os.path.join(project_root, "model")

# 下载模型到指定目录
path = snapshot_download('maple775885/RMBG-2.0', cache_dir=save_dir)
print("模型已下载到：", path)

print(path)