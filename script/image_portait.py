# 图像增强处理, 分辨率优化
import cv2
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.outputs import OutputKeys
import datetime
import os

portrait_enhancement = pipeline(Tasks.image_portrait_enhancement, model='iic/cv_gpen_image-portrait-enhancement')
result = portrait_enhancement('input/20031743473117_.pic_hd.jpg')
# 修改文件名
filename = f'result_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.png'

script_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(script_dir, '../output')
print(f'{os.path.join(output_path, filename)}')

cv2.imwrite(os.path.join(output_path, filename), result[OutputKeys.OUTPUT_IMG])