import torch
from torch import nn
from torch.types import FileLike
import torchvision
from torchvision import transforms

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

import numpy as np
from PIL import Image

from model.ml_output import InferenceResult
from utils import tensor_to_pil, encode_img_base64

convnext_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.PILToTensor(),
    transforms.ConvertImageDtype(torch.float),
    transforms.Normalize(std=[0.229, 0.224, 0.225], mean=[0.485, 0.456, 0.406]),
])

convnext_display_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.PILToTensor(),
    transforms.ConvertImageDtype(torch.float),
])

maxvit_transform = transforms.Compose([
    transforms.Resize(224, interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.PILToTensor(),
    transforms.ConvertImageDtype(torch.float),
    transforms.Normalize(std=[0.229, 0.224, 0.225], mean=[0.485, 0.456, 0.406]),
])

maxvit_display_transform = transforms.Compose([
    transforms.Resize(224, interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.PILToTensor(),
    transforms.ConvertImageDtype(torch.float),
])

def load_convnext(weight_path: FileLike, num_classes: int, device: str="cpu"):
    model = torchvision.models.convnext_tiny().to(device)

    # for params in model.features.parameters():
    #   params.requires_grad = False

    new_layer = nn.Sequential(
        torchvision.models.convnext.LayerNorm2d((768, ), eps=1e-06, elementwise_affine=True),
        nn.Flatten(start_dim=1, end_dim=-1),
        nn.Linear(in_features=768, out_features=num_classes, bias=True),
    )

    new_layer = new_layer.to(device)

    model.classifier = new_layer

    model.load_state_dict(torch.load(weight_path, weights_only=True, map_location=torch.device(device)))

    model.eval()

    print("ConvNeXt model loaded")

    return model

def load_maxvit(weight_path: FileLike, num_classes: int, device: str="cpu"):
    model = torchvision.models.maxvit_t().to(device)

    # for params in model.parameters():
    #   params.requires_grad = False

    new_layer = nn.Sequential(
        nn.AdaptiveAvgPool2d(output_size=1),
        nn.Flatten(start_dim=1, end_dim=-1),
        nn.LayerNorm((512, ), eps=1e-05, elementwise_affine=True),
        nn.Linear(in_features=512, out_features=512, bias=True),
        nn.Tanh(),
        nn.Linear(in_features=512, out_features=num_classes, bias=False),
    )

    new_layer = new_layer.to(device)

    model.classifier = new_layer

    model.load_state_dict(torch.load(weight_path, weights_only=True, map_location=torch.device(device)))

    model.eval()

    print("MaxViT model loaded")

    return model

def grad_cam_convnext(model: nn.Module):
    # for param in model.features.parameters():
    #   param.requires_grad = True

    target_layers = model.features[-1]

    return GradCAM(model=model, target_layers=target_layers)

def grad_cam_maxvit(model: nn.Module):
    # for params in model.parameters():
    #   params.requires_grad = True

    target_layers = model.blocks[-1].layers

    return GradCAM(model=model, target_layers=target_layers)

def inference_grad_cam(grad_cam: GradCAM, image: Image.Image, device: str="cpu"):
    image_transform = None
    display_image = None

    if isinstance(grad_cam.model, torchvision.models.convnext.ConvNeXt):
        image_transform = convnext_transform(image).unsqueeze(0)
        display_image = convnext_display_transform(image).permute(1, 2, 0).numpy()
    elif isinstance(grad_cam.model, torchvision.models.maxvit.MaxVit):
        image_transform = maxvit_transform(image).unsqueeze(0)
        display_image = maxvit_display_transform(image).permute(1, 2, 0).numpy()
    else:
        raise ValueError("Unsupported model type")

    # model.eval()
    # with torch.inference_mode():
    #   y_pred_logits = model(image_transform.to(device))

    # y_pred_softmax = torch.softmax(y_pred_logits, dim=1)
    # y_pred_class = y_pred_softmax.argmax(dim=1).cpu()

    grayscale_cam = grad_cam(image_transform.to(device))

    y_pred_logits = grad_cam.outputs

    y_pred_softmax = torch.softmax(y_pred_logits, dim=1)

    cam_image = show_cam_on_image(display_image, grayscale_cam[0, :], use_rgb=True)

    return InferenceResult(
        original_image=encode_img_base64(tensor_to_pil(display_image)),
        grad_cam_image=encode_img_base64(tensor_to_pil(cam_image)),
        scores=np.round(y_pred_softmax.squeeze().cpu().detach().numpy() * 100, 4).tolist(),
    )
