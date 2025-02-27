import sys
import torch
import cv2
import os
from PIL import Image

# Manually add Real-ESRGAN path
sys.path.append("D:/ImageEnhancer/backend/Real-ESRGAN")

from realesrgan.utils import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

# Load AI Model
model_path = "D:/ImageEnhancer/backend/Real-ESRGAN/weights/RealESRGAN_x4plus.pth"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Define model architecture
model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
upsampler = RealESRGANer(
    scale=4,
    model_path=model_path,
    model=model,
    tile=0,
    tile_pad=10,
    pre_pad=0,
    half=True if device == "cuda" else False
)

def enhance_image(image_path):
    """Enhance the image using Real-ESRGAN."""
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    enhanced_img, _ = upsampler.enhance(img, outscale=4)
    
    enhanced_path = image_path.replace(".", "_enhanced.")
    cv2.imwrite(enhanced_path, enhanced_img)
    
    return enhanced_path
