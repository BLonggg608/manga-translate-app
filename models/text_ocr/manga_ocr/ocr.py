import re
import os
import sys
from pathlib import Path

import jaconv
import torch
from PIL import Image
from loguru import logger
from transformers import ViTImageProcessor, AutoTokenizer, VisionEncoderDecoderModel, GenerationMixin

class MangaOcrModel(VisionEncoderDecoderModel, GenerationMixin):
    pass

class MangaOcr:
    def __init__(self, pretrained_model_name_or_path="kha-white/manga-ocr-base", force_cpu=False):
        logger.info(f"Loading OCR model from {pretrained_model_name_or_path}")
        
        # Load processor explicitly from local files
        self.processor = ViTImageProcessor.from_pretrained(
            pretrained_model_name_or_path, 
            local_files_only=True
        )
        
        # CRITICAL FIX: Set use_fast=False to bypass the Rust tokenizer backend 
        # which causes the 'Unable to load vocabulary' error in PyInstaller environments
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path, 
            local_files_only=True,
            use_fast=False
        )
        
        # Load model explicitly from local files
        self.model = MangaOcrModel.from_pretrained(
            pretrained_model_name_or_path, 
            local_files_only=True
        )

        if not force_cpu and torch.cuda.is_available():
            logger.info("Using CUDA")
            self.model.cuda()
        elif not force_cpu and torch.backends.mps.is_available():
            logger.info("Using MPS")
            self.model.to("mps")
        else:
            logger.info("Using CPU")

        # # Make the warmup phase optional to prevent crashes if example.jpg is stripped by PyInstaller
        # try:
        #     example_path = Path(__file__).parent / "assets" / "example.jpg"
        #     if example_path.is_file():
        #         self(example_path)
        #         logger.info("Model warmup successful")
        #     else:
        #         logger.warning("Warmup skipped: assets/example.jpg not found. (Safe to ignore)")
        # except Exception as e:
        #     logger.warning(f"Warmup skipped due to path resolution in compiled mode: {e}")

        logger.info("OCR ready")

    def __call__(self, img_or_path):
        if isinstance(img_or_path, str) or isinstance(img_or_path, Path):
            img = Image.open(img_or_path)
        elif isinstance(img_or_path, Image.Image):
            img = img_or_path
        else:
            raise ValueError(f"Invalid input. Expected path or PIL.Image, got: {img_or_path}")

        img = img.convert("L").convert("RGB")

        x = self._preprocess(img)
        x = self.model.generate(x[None].to(self.model.device), max_length=300)[0].cpu()
        x = self.tokenizer.decode(x, skip_special_tokens=True)
        x = post_process(x)
        return x

    def _preprocess(self, img):
        pixel_values = self.processor(img, return_tensors="pt").pixel_values
        return pixel_values.squeeze()


def post_process(text):
    text = "".join(text.split())
    text = text.replace("…", "...")
    text = re.sub("[・.]{2,}", lambda x: (x.end() - x.start()) * ".", text)
    text = jaconv.h2z(text, ascii=True, digit=True)

    return text