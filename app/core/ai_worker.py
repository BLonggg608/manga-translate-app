import os
import sys
import re
import gc
from PySide6.QtCore import QThread, Signal

HOME = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(HOME) 

from helper_func import filter_mask_by_boxes, manga_sort_boxes, get_system_prompt, create_user_payload, parse_json_output, render_text_on_manga

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

class ModelLoaderWorker(QThread):
    progress_updated = Signal(int, str)
    finished = Signal(dict)
    error_occurred = Signal(str)

    def __init__(self, gpu_id):
        super().__init__()
        self.gpu_id = str(gpu_id)

    def run(self):
        try:
            # CRITICAL FIX: Isolate the GPU BEFORE importing PyTorch.
            # This makes the selected physical GPU become the logical 'cuda:0' for this process.
            if self.gpu_id != "-1":
                os.environ['CUDA_VISIBLE_DEVICES'] = self.gpu_id
            else:
                os.environ['CUDA_VISIBLE_DEVICES'] = "-1"
                
            os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
            os.environ['LAMA_MODEL'] = f"{HOME}/models/simple-lama-inpainting/anime-manga-big-lama.pt"

            self.progress_updated.emit(5, "Initializing PyTorch Environment...")
            
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            from models.text_segmentation.model import MangaTextSegmenter
            from simple_lama_inpainting import SimpleLama
            from manga_ocr import MangaOcr
            from ultralytics import YOLO

            yolov8s_model_path = f'{HOME}/models/text-detector/comic-text-segmenter.pt'
            ocr_model_path = f"{HOME}/models/manga-ocr/models/manga-ocr-base"
            translate_model_path = f"{HOME}/models/translate-model/models/tiger-gemma-9b-v3"
            segment_model_path = f'{HOME}/models/text_segmentation/model.pth'

            self.progress_updated.emit(15, "Loading Text Detector (YOLO)...")
            detector = YOLO(yolov8s_model_path)

            self.progress_updated.emit(30, "Loading Text Segmenter...")
            segmenter = MangaTextSegmenter(model_path=segment_model_path)

            self.progress_updated.emit(45, "Loading Manga OCR Model...")
            mocr = MangaOcr(pretrained_model_name_or_path=ocr_model_path)

            self.progress_updated.emit(60, "Configuring AI Translation parameters...")
            compute_dtype = torch.bfloat16 if 'gemma-3-4b' in translate_model_path.lower() else torch.float16
            translator_tokenizer = AutoTokenizer.from_pretrained(translate_model_path)

            self.progress_updated.emit(75, "Loading LLM to GPU with 4-bit Quantization...")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True, 
                bnb_4bit_quant_type="nf4", 
                bnb_4bit_compute_dtype=compute_dtype, 
                bnb_4bit_use_double_quant=True
            )
            
            # Use 'auto' because the isolated GPU is now the only one available
            translator_model = AutoModelForCausalLM.from_pretrained(
                translate_model_path, 
                quantization_config=bnb_config, 
                device_map="auto", 
                trust_remote_code=False, 
                torch_dtype=compute_dtype
            )

            self.progress_updated.emit(90, "Loading Inpainting Model (Lama)...")
            simple_lama = SimpleLama()

            models = {
                'detector': detector,
                'segmenter': segmenter,
                'mocr': mocr,
                'translator_tokenizer': translator_tokenizer,
                'translator_model': translator_model,
                'simple_lama': simple_lama,
                'font_path': f'{HOME}/assets/fonts/animeace2_viethoa_reg.ttf'
            }

            self.progress_updated.emit(100, "All models loaded successfully.")
            self.finished.emit(models)

        except Exception as e:
            self.error_occurred.emit(f"MODEL LOAD ERROR: {str(e)}")


class AITranslatorWorker(QThread):
    progress_updated = Signal(int, str)
    finished = Signal(str)
    error_occurred = Signal(str)
    cancelled = Signal(str) 

    def __init__(self, config, models):
        super().__init__()
        self.config = config
        self.models = models 
        self._is_cancelled = False 

    def stop_gracefully(self):
        self._is_cancelled = True

    def check_cancel_and_cleanup(self, message="Process cancelled by user."):
        if self._is_cancelled:
            import torch
            torch.cuda.empty_cache()
            gc.collect()
            self.cancelled.emit(message)
            return True
        return False

    def run(self):
        try:
            import torch
            import cv2
            import numpy as np
            from PIL import Image

            detector = self.models['detector']
            segmenter = self.models['segmenter']
            mocr = self.models['mocr']
            translator_tokenizer = self.models['translator_tokenizer']
            translator_model = self.models['translator_model']
            simple_lama = self.models['simple_lama']
            font_path = self.models['font_path']

            input_dir = self.config['input_dir']
            output_dir = self.config['output_dir']
            valid_ext = ('.jpg', '.jpeg', '.png', '.webp')
            images = [f for f in os.listdir(input_dir) if f.lower().endswith(valid_ext)]
            images.sort(key=natural_sort_key)

            os.makedirs(output_dir, exist_ok=True)
            run_dir = os.path.join(output_dir, f'run_{len(os.listdir(output_dir)) + 1}')
            os.makedirs(run_dir, exist_ok=True)

            total_imgs = len(images)
            previous_translation_text = None

            for idx, img_name in enumerate(images):
                if self.check_cancel_and_cleanup("Translation process stopped at image start."): return

                base_percent = int((idx / total_imgs) * 100)
                img_path = os.path.join(input_dir, img_name)
                
                self.progress_updated.emit(base_percent + 2, f"[{idx+1}/{total_imgs}] Detecting text bubbles: {img_name}")
                
                # Default logic now routes safely to logical cuda:0
                detector.to('cuda')
                segmenter.model.to('cuda')
                mocr.model.to('cuda')

                image = Image.open(img_path).convert("RGB")
                text_mask_image = segmenter.segment(image)
                boxes = detector.predict(image, conf=0.1, imgsz=1024)[0].boxes.xyxy.cpu().numpy().tolist()
                text_mask_image = filter_mask_by_boxes(text_mask_image, boxes)

                self.progress_updated.emit(base_percent + 10, f"[{idx+1}/{total_imgs}] Reading Japanese text (OCR)...")
                boxes = manga_sort_boxes(boxes)
                jp_texts = []
                
                for box_idx, box in enumerate(boxes):
                    if self.check_cancel_and_cleanup(f"Cancelled during OCR phase (box {box_idx})."): return
                    cropped_img = image.crop(tuple(box))
                    jp_texts.append(mocr(cropped_img))
                    del cropped_img 

                detector.to('cpu')
                segmenter.model.to('cpu')
                mocr.model.to('cpu')
                torch.cuda.empty_cache()
                gc.collect()

                if self.check_cancel_and_cleanup("Stopped before LLM translation."): return

                self.progress_updated.emit(base_percent + 40, f"[{idx+1}/{total_imgs}] Generating translation using LLM...")
                system_prompt = get_system_prompt(genre=self.config['genre'], target_language=self.config['target_lang'])
                current_user_payload = create_user_payload(jp_texts)

                if previous_translation_text:
                    combined_prompt = f"{system_prompt}\n\n---\nCONTEXT FROM PREVIOUS PAGE:\n{previous_translation_text}\n\n---\nTRANSLATE NEW PAGE:\n{current_user_payload}"
                else:
                    combined_prompt = f"{system_prompt}\n\n---\n\n{current_user_payload}"

                messages = [{"role": "user", "content": combined_prompt}]
                
                input_ids = translator_tokenizer.apply_chat_template(
                    messages, 
                    add_generation_prompt=True, 
                    tokenize=True, 
                    return_dict=True, 
                    return_tensors="pt"
                ).to(translator_model.device)

                with torch.no_grad():
                    output_ids = translator_model.generate(**input_ids, max_new_tokens=4048, temperature=0.5, do_sample=True, top_p=0.95)

                generated_ids = output_ids[0][input_ids["input_ids"].shape[-1]:].cpu()
                response_text = translator_tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
                translated_texts = parse_json_output(response_text, len(jp_texts))
                
                previous_translation_text = response_text

                del input_ids, output_ids, generated_ids
                torch.cuda.empty_cache()
                gc.collect()

                if self.check_cancel_and_cleanup("Stopped after translation, before rendering."): return

                self.progress_updated.emit(base_percent + 75, f"[{idx+1}/{total_imgs}] Removing original text and rendering translation...")
                
                text_mask_image = np.array(text_mask_image)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
                text_mask_image = cv2.dilate(text_mask_image, kernel, iterations=1)
                text_mask_image = Image.fromarray(text_mask_image)

                cleared_image = simple_lama(image, text_mask_image)
                
                del image, text_mask_image
                torch.cuda.empty_cache()
                gc.collect()

                if self.check_cancel_and_cleanup("Stopped right before rendering output image."): return

                try:
                    result_img = render_text_on_manga(cleared_image, boxes, translated_texts, font_path)
                except Exception as e:
                    result_img = cleared_image

                save_path = os.path.join(run_dir, img_name)
                result_img.save(save_path)
                
                del cleared_image, result_img
                gc.collect()

            if not self._is_cancelled:
                self.progress_updated.emit(100, "100% Completed!")
                self.finished.emit(f"Translation successful! Results saved at:\n{run_dir}")

        except Exception as e:
            self.error_occurred.emit(f"SYSTEM ERROR: {str(e)}")