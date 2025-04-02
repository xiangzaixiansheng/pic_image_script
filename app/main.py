import os
import sys
import gradio as gr
from PIL import Image
import numpy as np

# Add the parent directory to sys.path to import from script folder
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from script.image_no_background import remove_background
from script.image_portait import enhance_portrait

def process_background_removal(image):
    if image is None:
        return None
    
    # Save the input image temporarily
    temp_input_path = "temp_input.png"
    if isinstance(image, str):
        # If image is already a path
        temp_input_path = image
    else:
        # If image is a numpy array or PIL Image
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        image.save(temp_input_path)
    
    # Process the image
    output_path = remove_background(temp_input_path)
    
    # Clean up temp files if needed
    if temp_input_path != image:
        os.remove(temp_input_path)
    
    return output_path

def process_portrait_enhancement(image):
    if image is None:
        return None
    
    # Save the input image temporarily
    temp_input_path = "temp_input.png"
    if isinstance(image, str):
        # If image is already a path
        temp_input_path = image
    else:
        # If image is a numpy array or PIL Image
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        image.save(temp_input_path)
    
    # Process the image
    output_path = enhance_portrait(temp_input_path)
    
    # Clean up temp files if needed
    if temp_input_path != image:
        os.remove(temp_input_path)
    
    return output_path

def create_image_processing_interface():
    with gr.Blocks() as demo:
        gr.Markdown("# 图像处理工具")
        
        with gr.Tab("一键抠图"):
            with gr.Row():
                with gr.Column():
                    bg_input_image = gr.Image(label="上传图片", type="filepath")
                with gr.Column():
                    bg_result_view = gr.Image(label="抠图结果", interactive=False)
            
            bg_process_btn = gr.Button("开始抠图")
            bg_process_btn.click(
                fn=process_background_removal,
                inputs=[bg_input_image],
                outputs=[bg_result_view]
            )
            
        with gr.Tab("人像增强"):
            with gr.Row():
                with gr.Column():
                    portrait_input_image = gr.Image(label="上传图片", type="filepath")
                with gr.Column():
                    portrait_result_view = gr.Image(label="增强结果", interactive=False)
            
            portrait_process_btn = gr.Button("开始增强")
            portrait_process_btn.click(
                fn=process_portrait_enhancement,
                inputs=[portrait_input_image],
                outputs=[portrait_result_view]
            )
    
    return demo

if __name__ == "__main__":
    demo = create_image_processing_interface()
    demo.launch(share=False)
