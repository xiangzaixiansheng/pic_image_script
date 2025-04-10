import os
import sys
import gradio as gr
from PIL import Image
import numpy as np

# Add the parent directory to sys.path to import from script folder
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from script.image_no_background import remove_background
from script.image_portait import enhance_portrait
from script.image_use_story import generate_story_image

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

def process_story_generation(prompt, negative_prompt="", num_steps=50, guidance_scale=7.5):
    """处理故事风格图片生成"""
    try:
        output_path = generate_story_image(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=int(num_steps),
            guidance_scale=float(guidance_scale)
        )
        return output_path
    except Exception as e:
        raise gr.Error(f"生成图片时出错: {str(e)}")

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

        with gr.Tab("故事风格生成"):
            with gr.Row():
                with gr.Column():
                    story_prompt = gr.Textbox(
                        label="正向提示词",
                        placeholder="输入正向提示词，例如: (masterpiece:1.2), best quality, fairy tale style..."
                    )
                    story_negative_prompt = gr.Textbox(
                        label="负向提示词",
                        placeholder="输入负向提示词，例如: EasyNegative, badhandsv5-neg..."
                    )
                    with gr.Row():
                        num_steps = gr.Slider(
                            minimum=20,
                            maximum=100,
                            value=50,
                            step=1,
                            label="推理步数"
                        )
                        guidance_scale = gr.Slider(
                            minimum=1.0,
                            maximum=20.0,
                            value=7.5,
                            step=0.1,
                            label="提示词引导强度"
                        )
                with gr.Column():
                    story_result_view = gr.Image(label="生成结果", interactive=False)
            
            story_process_btn = gr.Button("开始生成")
            story_process_btn.click(
                fn=process_story_generation,
                inputs=[story_prompt, story_negative_prompt, num_steps, guidance_scale],
                outputs=[story_result_view]
            )
    
    return demo

if __name__ == "__main__":
    demo = create_image_processing_interface()
    demo.launch(share=False)
