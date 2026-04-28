from torchvision import transforms
import io
from PIL import Image
import base64

class_names = ['Blight', 'Common_Rust', 'Gray_Leaf_Spot', 'Healthy', 'Pest_Damage']

tensor_to_pil = transforms.ToPILImage(mode='RGB')

def encode_img_base64(img: Image):
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG')

    return f"data:image/jpeg;base64,{base64.b64encode(buffer.getvalue()).decode('utf-8')}"
