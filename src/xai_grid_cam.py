import os
import sys
import torch as t
import numpy as np
import matplotlib.pyplot as plt
from imageio.v2 import get_writer, imread
from typing import List, Tuple, Optional
from PIL import Image
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from utils import get_resnet_for_fine_tuning
from tqdm import tqdm
from natsort import natsorted

# Constants
MODEL_PATH = '../models/TFBS_resnet50.pth'
IMAGE_SIZE = 224
device = 'cuda' if t.cuda.is_available() else 'cpu'

# Image transforms
transform_resnet = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.to(device))
])

transform_heatmap = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.Lambda(lambda x: np.array(x) / 255)
])

def load_model(model_path: str, num_classes: int = 2):
    """Load and prepare the model for inference."""
    model = get_resnet_for_fine_tuning(num_classes=num_classes)
    model.load_state_dict(t.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def generate_gradcam_image(model, input_tensor: t.Tensor, targets: List, rgb_img_np: np.array, layer):
    """Generate a Grad-CAM heatmap for a given model layer."""
    with GradCAM(model=model, target_layers=[layer]) as cam:
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
        grayscale_cam = np.where(grayscale_cam > np.percentile(grayscale_cam, 50), grayscale_cam, 0)
        return rgb_img_np * np.expand_dims(grayscale_cam, axis=-1)

def save_image(image: np.array, output_path: str):
    """Save an image to disk."""
    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    plt.axis('off')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def parse_commandline(args: List[str]) -> Tuple[str, str, bool, Optional[str], Optional[int]]:
    """Parse command-line arguments."""
    if len(args) not in [3, 4, 5]:
        raise ValueError("Usage: xai_grid_cam.py <input_dir> <output_dir> <gif_flag> [<max_sequences>] [<gif_output_dir>]")
    
    indir, outdir, gif_flag = args[:3]
    gif_flag = bool(int(gif_flag))
    max_sequences = int(args[3]) if len(args) == 4 or len(args) == 5 else None
    outdir_gif = args[4] if gif_flag and len(args) == 5 else None
    
    if gif_flag and outdir_gif is None:
        raise ValueError("GIF output directory must be specified if gif_flag is set to 1.")
    
    return indir, outdir, gif_flag, max_sequences, outdir_gif

def create_gif(image_paths: List[str], output_path: str):
    """Create a GIF from a sequence of images."""
    with get_writer(output_path, mode='I', duration=0.5) as writer:
        for img_path in image_paths:
            writer.append_data(imread(img_path))

def main():
    indir, outdir, gif_flag, max_sequences, outdir_gif = parse_commandline(sys.argv[1:])

    imagefiles = natsorted(os.listdir(indir))

    if max_sequences is not None and (max_sequences < len(imagefiles)):
        imagefiles = imagefiles[:max_sequences]
    
    model = load_model(MODEL_PATH)
    target_layers = [model.layer4[-3], model.layer4[-2], model.layer4[-1]]
    targets = [ClassifierOutputTarget(0)]  # Assuming class index 0 for positive samples

    for imagefile in tqdm(imagefiles, total=len(imagefiles), colour='GREEN'):
        image_path = os.path.join(indir, imagefile)
        rgb_img = Image.open(image_path).convert('RGB')
        input_tensor = transform_resnet(rgb_img).unsqueeze(0)
        rgb_img_np = transform_heatmap(rgb_img)

        image_paths = []
        
        for idx, layer in enumerate(target_layers):
            output_image = generate_gradcam_image(model, input_tensor, targets, rgb_img_np, layer)
            layer_dir = os.path.join(outdir, f"Layer_{idx}")
            os.makedirs(layer_dir, exist_ok=True)
            
            output_file = os.path.join(layer_dir, f"{os.path.splitext(imagefile)[0]}_{idx}.png")
            save_image(output_image, output_file)
            image_paths.append(output_file)
        
        if gif_flag and outdir_gif:
            os.makedirs(outdir_gif, exist_ok=True)
            gif_path = os.path.join(outdir_gif, f"{os.path.splitext(imagefile)[0]}.gif")
            create_gif(image_paths, gif_path)

if __name__ == '__main__':
    main()
