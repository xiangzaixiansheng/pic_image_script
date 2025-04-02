import os
import sys
import gradio as gr
from PIL import Image
import numpy as np

# Add the parent directory to sys.path to import from script folder
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from script.image_no_background import remove_background

def process_image(image):
    if image is None:
        return [None, None]
    
    # Save the input image temporarily
    temp_input_path = "temp_input.png"
    if isinstance(image, str):
        # If image is already a path
        temp_input_path = image
        original_image = Image.open(temp_input_path)
    else:
        # If image is a numpy array or PIL Image
        if not isinstance(image, Image.Image):
            original_image = Image.fromarray(image)
        else:
            original_image = image
        original_image.save(temp_input_path)
    
    # Process the image
    output_path = remove_background(temp_input_path)
    
    # Clean up temp files if needed
    if temp_input_path != image:
        os.remove(temp_input_path)
    
    return [output_path, original_image]

def create_image_processing_interface():
    with gr.Blocks() as demo:
        gr.Markdown("# 图像处理工具")
        
        with gr.Tab("一键抠图"):
            with gr.Row():
                with gr.Column():
                    input_image = gr.Image(label="上传图片", type="filepath")
                with gr.Column():
                    with gr.Row():
                        original_view = gr.Image(label="原图", interactive=False)
                        result_view = gr.Image(label="抠图结果", interactive=False)
            
            process_btn = gr.Button("开始处理")
            process_btn.click(
                fn=process_image,
                inputs=[input_image],
                outputs=[result_view, original_view]
            )
    
    return demo

if __name__ == "__main__":
    demo = create_image_processing_interface()
    demo.launch(share=False)
