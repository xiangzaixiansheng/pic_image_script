from PIL import Image
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from modelscope import AutoModelForImageSegmentation
import os
import datetime

def remove_background(input_image_path):
    model = AutoModelForImageSegmentation.from_pretrained('maple775885/RMBG-2.0', trust_remote_code=True)
    torch.set_float32_matmul_precision(['high', 'highest'][0])
    model.to('cpu')
    model.eval()

    # Data settings
    image_size = (1024, 1024)
    transform_image = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image = Image.open(input_image_path).convert('RGB')
    input_images = transform_image(image).unsqueeze(0).to('cpu')

    # Prediction
    with torch.no_grad():
        preds = model(input_images)[-1].sigmoid().cpu()
    pred = preds[0].squeeze()
    pred_pil = transforms.ToPILImage()(pred)
    mask = pred_pil.resize(image.size)
    image.putalpha(mask)

    # Save output
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, '../output')
    os.makedirs(output_dir, exist_ok=True)
    filename = f'result_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
    output_path = os.path.join(output_dir, filename)
    image.save(output_path)

    return output_path

if __name__ == '__main__':
    input_image_path = "input/5a8a069aee5041a4b470671460fbc03f.png"
    output_path = remove_background(input_image_path)
    print(f'Saved result to: {output_path}')