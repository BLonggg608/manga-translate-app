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
        Khởi tạo model.
        :param model_path: Đường dẫn đến file model.pth
        :param device: 'cuda' hoặc 'cpu'. Nếu None sẽ tự động chọn.
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.encoder = "tu-efficientnetv2_rw_m"
        
        print(f"Device being used: {self.device}")
        self.model = self._load_model()

    def _convert_batchnorm_to_groupnorm(self, module):
        """Hàm phụ trợ để convert layer theo đúng kiến trúc của tác giả"""
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
                setattr(module, name, nn.GroupNorm(num_groups=num_groups, num_channels=num_channels))
            else:
                self._convert_batchnorm_to_groupnorm(child)

    def _load_model(self):
        print(f"Loading model from {self.model_path}...")
        
        # Khởi tạo kiến trúc model
        model = smp.UnetPlusPlus(
            encoder_name=self.encoder,
            encoder_weights=None, # Không cần download weights imagenet vì sẽ load file .pth
            in_channels=3,
            classes=1,
            activation=None,
            decoder_attention_type='scse'
        )

        # Convert cấu trúc layer (Bắt buộc phải có bước này mới load được weight)
        self._convert_batchnorm_to_groupnorm(model.decoder)

        # Load weights
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at {self.model_path}")

        state_dict = torch.load(self.model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        
        model.to(self.device)
        model.eval()
        print("Model loaded successfully!")
        return model

    def segment(self, pil_image: Image.Image, threshold=0.5) -> Image.Image:
        """
        Thực hiện segment text.
        :param pil_image: Ảnh đầu vào dạng PIL.Image
        :param threshold: Ngưỡng để phân loại pixel (0.0 - 1.0)
        :return: Ảnh mask trắng đen dạng PIL.Image (Mode 'L')
        """
        # 1. Preprocess: Convert PIL to Numpy & Resize padding
        image_np = np.array(pil_image.convert("RGB"))
        h, w = image_np.shape[:2]
        
        # Padding cho chia hết cho 32 (Yêu cầu của UNet)
        pad_h = (32 - h % 32) % 32
        pad_w = (32 - w % 32) % 32

        transform = A.Compose([
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
        
        augmented = transform(image=image_np)
        tensor = augmented['image'].unsqueeze(0).to(self.device)

        if pad_h > 0 or pad_w > 0:
            tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode='constant', value=0)

        # 2. Inference
        with torch.no_grad():
            if self.device == 'cuda':
                # Dùng mixed precision nếu có cuda để nhanh hơn
                try:
                    with torch.amp.autocast('cuda'):
                        logits = self.model(tensor)
                        probs = logits.sigmoid()
                except AttributeError: # Fallback cho torch cũ
                     logits = self.model(tensor)
                     probs = logits.sigmoid()
            else:
                logits = self.model(tensor)
                probs = logits.sigmoid()

        # 3. Post-process
        # Cắt bỏ phần padding thừa lúc nãy
        prob_map = probs[0, 0, :h, :w].cpu().numpy()
        
        # Thresholding -> Binary Mask
        binary_mask = (prob_map > threshold).astype(np.uint8) * 255
        
        # Trả về PIL Image
        return Image.fromarray(binary_mask, mode='L')