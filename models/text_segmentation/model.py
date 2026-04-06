import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os


class MangaTextSegmenter:
    def __init__(self, model_path, device=None):
        """
        Initialize the model.
        :param model_path: Path to the model.pth file
        :param device: 'cuda' or 'cpu'. If None, it will be automatically selected.
        """
        self.device = device if device else (
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.encoder = "tu-efficientnetv2_rw_m"

        print(f"Device being used: {self.device}")
        self.model = self._load_model()

    def _convert_batchnorm_to_groupnorm(self, module):
        """Helper function to convert layers according to the author's architecture"""
        for name, child in module.named_children():
            if isinstance(child, nn.BatchNorm2d):
                num_channels = child.num_features
                num_groups = 8
                if num_channels < num_groups or num_channels % num_groups != 0:
                    for i in range(min(num_channels, 8), 1, -1):
                        if num_channels % i == 0:
                            num_groups = i
                            break
                    else:
                        num_groups = 1
                setattr(module, name, nn.GroupNorm(
                    num_groups=num_groups, num_channels=num_channels))
            else:
                self._convert_batchnorm_to_groupnorm(child)

    def _load_model(self):
        print(f"Loading model from {self.model_path}...")

        # Initialize the model architecture
        model = smp.UnetPlusPlus(
            encoder_name=self.encoder,
            encoder_weights=None,  # No need to download imagenet weights as we will load a .pth file
            in_channels=3,
            classes=1,
            activation=None,
            decoder_attention_type='scse'
        )

        # Convert layer structure (This step is mandatory to load the weights)
        self._convert_batchnorm_to_groupnorm(model.decoder)

        # Load weights
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model file not found at {self.model_path}")

        state_dict = torch.load(self.model_path, map_location=self.device)
        model.load_state_dict(state_dict)

        model.to(self.device)
        model.eval()
        print("Model loaded successfully!")
        return model

    def segment(self, pil_image: Image.Image, threshold=0.5) -> Image.Image:
        """
        Perform text segmentation.
        :param pil_image: Input image as PIL.Image
        :param threshold: Pixel classification threshold (0.0 - 1.0)
        :return: Black and white mask as PIL.Image (Mode 'L')
        """
        # 1. Preprocess: Convert PIL to Numpy & Resize padding
        image_np = np.array(pil_image.convert("RGB"))
        h, w = image_np.shape[:2]

        # Padding to make dimensions divisible by 32 (UNet requirement)
        pad_h = (32 - h % 32) % 32
        pad_w = (32 - w % 32) % 32

        transform = A.Compose([
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

        augmented = transform(image=image_np)
        tensor = augmented['image'].unsqueeze(0).to(self.device)

        if pad_h > 0 or pad_w > 0:
            tensor = F.pad(tensor, (0, pad_w, 0, pad_h),
                           mode='constant', value=0)

        # 2. Inference
        with torch.no_grad():
            if self.device == 'cuda':
                # Use mixed precision if cuda is available for faster inference
                try:
                    with torch.amp.autocast('cuda'):
                        logits = self.model(tensor)
                        probs = logits.sigmoid()
                except AttributeError:  # Fallback for older torch versions
                    logits = self.model(tensor)
                    probs = logits.sigmoid()
            else:
                logits = self.model(tensor)
                probs = logits.sigmoid()

        # 3. Post-process
        # Remove the excess padding added earlier
        prob_map = probs[0, 0, :h, :w].cpu().numpy()

        # Thresholding -> Binary Mask
        binary_mask = (prob_map > threshold).astype(np.uint8) * 255

        # Return as PIL Image
        return Image.fromarray(binary_mask, mode='L')
