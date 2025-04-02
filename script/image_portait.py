# 图像增强处理, 分辨率优化
import cv2
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.outputs import OutputKeys
import datetime
import os

def enhance_portrait(input_image_path):
    """
    对输入的人像图片进行增强处理
    Args:
        input_image_path: 输入图片的路径
    Returns:
        str: 处理后图片的保存路径
    """
    # 创建pipeline（只需创建一次）
    portrait_enhancement = pipeline(Tasks.image_portrait_enhancement, model='iic/cv_gpen_image-portrait-enhancement')
    
    # 处理图片
    result = portrait_enhancement(input_image_path)
    
    # 生成输出文件名
    filename = f'result_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
    
    # 设置输出路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, '../output')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    
    # 保存结果
    cv2.imwrite(output_path, result[OutputKeys.OUTPUT_IMG])
    
    return output_path

if __name__ == "__main__":
    # 测试代码
    test_image = 'input/20031743473117_.pic_hd.jpg'
    result_path = enhance_portrait(test_image)
    print(f'Enhanced image saved to: {result_path}')