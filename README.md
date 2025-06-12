



## 一、项目说明

### 1.项目脚本主要

https://modelscope.cn/models
里面的模型scope的库和工具

### 2.mac环境安装命令

```shell
conda create -n modelscope_new python=3.10
conda activate modelscope_new

pip3 install torch torchvision torchaudio -i https://pypi.tuna.tsinghua.edu.cn/simple


pip3 install modelscope -i https://mirrors.aliyun.com/pypi/simple/
pip3 install "modelscope[multi-modal]" -i https://mirrors.aliyun.com/pypi/simple/

#下面这个可以不安装也可以，用于后续使用torchvision
pip3 install "modelscope[nlp]" -i https://mirrors.aliyun.com/pypi/simple/

# 下面也可以不安装，如果需要在安装
pip install numpy==1.22.0
pip install scikit-image
```

报错内容：

如果出现ImportError: numpy.core.multiarray failed to import

```shell
pip uninstall numpy pandas modelscope

pip3 install pandas==1.5.3
pip3 install numpy==1.22.0
pip3 install modelscope==1.6.1
```

# 升级最新版的
pip install -U modelscope -i https://pypi.tuna.tsinghua.edu.cn/simple

pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

---

## Gradio 功能接口文档

本项目通过 Gradio 提供了以下两个主要功能：

### 1. 一键抠图（背景移除）

- **功能说明**：上传图片，自动去除图片背景，输出为透明背景 PNG。
- **输入**：图片文件（JPG/PNG）
- **输出**：去除背景后的 PNG 图片
- **前端操作**：在 Gradio 页面上传图片，点击“一键抠图”按钮获得结果。
- **API 调用**（如需程序化调用）：
    - `POST /api/predict/0`
    - 请求体：multipart/form-data，字段名为 data，内容为图片
    - 返回：去背景后的图片

### 2. 人像增强

- **功能说明**：上传人像图片，自动美化、增强细节。
- **输入**：人像图片文件（JPG/PNG）
- **输出**：增强后的人像图片
- **前端操作**：在 Gradio 页面上传图片，点击“人像增强”按钮获得结果。
- **API 调用**：
    - `POST /api/predict/1`
    - 请求体：multipart/form-data，字段名为 data，内容为图片
    - 返回：增强后的人像图片

#### 示例（API 调用）

```bash
# 一键抠图
curl -X POST -F 'data=@your_image.jpg' http://localhost:7860/api/predict/0

# 人像增强
curl -X POST -F 'data=@your_image.jpg' http://localhost:7860/api/predict/1
```

---
