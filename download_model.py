from modelscope.hub.snapshot_download import snapshot_download
# 下载模型
path = snapshot_download('damo/cv_unet_image-matting')
print(path)