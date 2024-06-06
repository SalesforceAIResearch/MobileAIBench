import yaml
import base64
from io import BytesIO


def load_config(model_name, file_path):
    with open(file_path, 'r') as file:
        configs = yaml.safe_load(file)
    if model_name in configs:
        return configs[model_name]
    else:
        raise ValueError(f"No model config for model {model_name}")
    

def image_to_base64_data_uri(file_path):
    with open(file_path, "rb") as img_file:
        base64_data = base64.b64encode(img_file.read()).decode('utf-8')
        return f"data:image/png;base64,{base64_data}"


def pil_to_base64_data_uri(pil_obj):
    img_buffer = BytesIO()
    if pil_obj.mode == 'CMYK':
        pil_obj = pil_obj.convert('RGB')
    pil_obj.save(img_buffer, format='PNG')
    byte_data = img_buffer.getvalue()
    base64_data = base64.b64encode(byte_data).decode('utf-8')
    return f"data:image/png;base64,{base64_data}"