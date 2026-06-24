from torchvision import transforms
import io
from PIL import Image
import base64

class_names = ['Hawar Daun', 'Karat Daun', 'Bercak Daun', 'Sehat', 'Kerusakan Hama']

tensor_to_pil = transforms.ToPILImage(mode='RGB')

def encode_img_base64(img: Image):
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG')

    return f"data:image/jpeg;base64,{base64.b64encode(buffer.getvalue()).decode('utf-8')}"
