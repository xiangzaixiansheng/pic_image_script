



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

