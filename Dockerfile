FROM registry.cn-hangzhou.aliyuncs.com/hanxiang/ubuntu22.04-py310-torch2.1.0-tf2.14.0-1.10.0-nodejs16.14:1.0.0

ENV XDG_CACHE_HOME /mnt/workspace/.cache

COPY . /home/pic_img_script
WORKDIR /home/pic_img_script

RUN mkdir -p /mnt/workspace/.cache/modelscope/maple775885 && \
    cp -r ./model/maple775885/. /mnt/workspace/.cache/modelscope/maple775885/ && \
    mkdir -p /mnt/workspace/.cache/modelscope/iic && \
    cp -r ./model/iic/. /mnt/workspace/.cache/modelscope/iic/
    
RUN pip3 install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/

ENV ENV=prod

# 启动命令（如入口为app.py，需根据实际调整）
ENTRYPOINT ["uwsgi", "--ini", "img_uwsgi.ini"]