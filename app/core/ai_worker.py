import os
import sys
import re
import gc
import shutil
from PySide6.QtCore import QThread, Signal

def get_app_root():
    if getattr(sys, 'frozen', False):
        # When run as a .exe
        return os.path.dirname(sys.executable)
    else:
        # When run as a .py
        return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

HOME = get_app_root()
sys.path.append(HOME) 

# Ensure the manga-ocr model directory is in the path for imports
local_manga_ocr_path = os.path.normpath(os.path.join(HOME, 'models', 'text_ocr'))
if local_manga_ocr_path not in sys.path:
    sys.path.insert(0, local_manga_ocr_path)

from helper_func import natural_sort_key, filter_mask_by_boxes, manga_sort_boxes, get_system_prompt, create_user_payload, parse_json_output, render_text_on_manga

class ModelLoaderWorker(QThread):
    progress_updated = Signal(int, str)
    finished = Signal(dict)
    error_occurred = Signal(str)

    def __init__(self, gpu_id, engine):
        super().__init__()
        self.gpu_id = str(gpu_id)
        self.engine = engine # Store the selected engine

    def run(self):
        try:
            if self.gpu_id != "-1":
                os.environ['CUDA_VISIBLE_DEVICES'] = self.gpu_id
            else:
                os.environ['CUDA_VISIBLE_DEVICES'] = "-1"
                
            os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
            
            lama_path = os.path.join(HOME, "models", "simple-lama-inpainting", "anime-manga-big-lama.pt").replace('\\', '/')
            os.environ['LAMA_MODEL'] = lama_path

            self.progress_updated.emit(5, "Initializing PyTorch Environment...")
            
            import torch
            from models.text_segmentation.model import MangaTextSegmenter
            from simple_lama_inpainting import SimpleLama
            from manga_ocr import MangaOcr
            from ultralytics import YOLO

            yolov8_model_path = os.path.join(HOME, 'models', 'text-detector', 'comic-text-segmenter.pt').replace('\\', '/')
            ocr_model_path = os.path.join(HOME, 'models', 'text_ocr', 'models', 'manga-ocr-base').replace('\\', '/')
            segment_model_path = os.path.join(HOME, 'models', 'text_segmentation', 'model.pth').replace('\\', '/')
            font_path = os.path.join(HOME, 'assets', 'fonts', 'animeace2_viethoa_reg.ttf').replace('\\', '/')

            check_paths = [
                (yolov8_model_path, "YOLOv8 Text Detector"),
                (ocr_model_path, "Manga OCR (Base Directory)"),
                (segment_model_path, "Text Segmenter"),
                (font_path, "Font File")
            ]

            # Only verify and load Local LLM path if it is selected
            if self.engine == "Local LLM":
                translate_model_path = os.path.join(HOME, 'models', 'translate-model', 'models', 'tiger-gemma-9b-v3').replace('\\', '/')
                check_paths.append((translate_model_path, "Gemma LLM (Tiger Directory)"))

            for p, name in check_paths:
                if not os.path.exists(p):
                    raise FileNotFoundError(f"PATH ERROR: Could not find {name}!\nAttempted path: \n{p}\n\nPlease verify your directory structure in the build folder.")

            self.progress_updated.emit(15, "Loading Text Detector (YOLO)...")
            detector = YOLO(yolov8_model_path)

            self.progress_updated.emit(30, "Loading Text Segmenter...")
            segmenter = MangaTextSegmenter(model_path=segment_model_path)

            self.progress_updated.emit(45, "Loading Manga OCR Model...")
            mocr = MangaOcr(pretrained_model_name_or_path=ocr_model_path)

            translator_tokenizer = None
            translator_model = None

            # --- ENGINE CONDITIONAL LOADING ---
            if self.engine == "Local LLM":
                from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
                self.progress_updated.emit(60, "Configuring Local LLM Translation parameters...")
                compute_dtype = torch.bfloat16 if 'gemma-3-4b' in translate_model_path.lower() else torch.float16
                translator_tokenizer = AutoTokenizer.from_pretrained(translate_model_path)

                self.progress_updated.emit(75, "Loading LLM to GPU with 4-bit Quantization...")
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True, 
                    bnb_4bit_quant_type="nf4", 
                    bnb_4bit_compute_dtype=compute_dtype, 
                    bnb_4bit_use_double_quant=True
                )
                
                translator_model = AutoModelForCausalLM.from_pretrained(
                    translate_model_path, 
                    quantization_config=bnb_config, 
                    device_map="auto", 
                    trust_remote_code=False, 
                    dtype=compute_dtype
                )
            else:
                self.progress_updated.emit(75, "Bypassing Local LLM (Using Gemini API)...")

            self.progress_updated.emit(90, "Loading Inpainting Model (Lama)...")
            simple_lama = SimpleLama()

            models = {
                'detector': detector,
                'segmenter': segmenter,
                'mocr': mocr,
                'translator_tokenizer': translator_tokenizer, # Will be None if Gemini
                'translator_model': translator_model,         # Will be None if Gemini
                'simple_lama': simple_lama,
                'font_path': font_path
            }

            self.progress_updated.emit(100, "All models loaded successfully.")
            self.finished.emit(models)

        except Exception as e:
            self.error_occurred.emit(f"MODEL LOAD ERROR: {str(e)}")


class AITranslatorWorker(QThread):
    progress_updated = Signal(int, str)
    image_translated = Signal(str) # NEW SIGNAL: Emits the path of the newly translated image
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

            if self.config['engine'] == "Gemini API":
                from google.genai import types
                from google import genai
                gemini_client = genai.Client(api_key=self.config['api_key'])

            image_paths = self.config['image_paths']
            # sort image paths based on filename to ensure correct page order
            image_paths.sort(key=lambda x: natural_sort_key(os.path.basename(x)))
            
            # Prepare Temporary Cache Directory
            temp_dir = os.path.join(HOME, "temp_translation_cache")
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            os.makedirs(temp_dir, exist_ok=True)

            total_imgs = len(image_paths)
            previous_translation_text = None

            for idx, img_path in enumerate(image_paths):
                if self.check_cancel_and_cleanup("Translation process stopped at image start."): return

                img_name = os.path.basename(img_path)
                base_percent = int((idx / total_imgs) * 100)
                
                self.progress_updated.emit(base_percent + 2, f"[{idx+1}/{total_imgs}] Detecting text bubbles: {img_name}")
                
                detector.to('cuda')
                segmenter.model.to('cuda')
                mocr.model.to('cuda')

                image = Image.open(img_path).convert("RGB")
                text_mask_image = segmenter.segment(image)
                boxes = detector.predict(image, imgsz=1024)[0].boxes.xyxy.cpu().numpy().tolist()
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

                self.progress_updated.emit(base_percent + 40, f"[{idx+1}/{total_imgs}] Generating translation using {self.config['engine']}...")
                system_prompt = get_system_prompt(genre=self.config['genre'], target_language=self.config['target_lang'])
                current_user_payload = create_user_payload(jp_texts)

                if previous_translation_text:
                    combined_prompt = f"{system_prompt}\n\n---\nCONTEXT FROM PREVIOUS PAGE:\n{previous_translation_text}\n\n---\nTRANSLATE NEW PAGE:\n{current_user_payload}"
                else:
                    combined_prompt = f"{system_prompt}\n\n---\n\n{current_user_payload}"

                if self.config['engine'] == "Gemini API":
                    import google.generativeai as genai
                    genai.configure(api_key=self.config['api_key'])
                    
                    response = gemini_client.models.generate_content(
                        model="gemini-3-flash-preview",
                        contents=[combined_prompt],
                        config=types.GenerateContentConfig(
                            temperature=0.5,     # increase temperature for more diverse output
                            topP=0.95,
                        )
                    )
                    response_text = response.text.strip()
                    
                else:
                    # ORIGINAL LOCAL LLM LOGIC
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
                    
                    del input_ids, output_ids, generated_ids
                    
                translated_texts = parse_json_output(response_text, len(jp_texts))
                
                previous_translation_text = response_text

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

                # Save to Temporary Cache Directory
                save_path = os.path.join(temp_dir, img_name)
                result_img.save(save_path)
                
                # Emit the path of the saved image back to the UI
                self.image_translated.emit(save_path)
                
                del cleared_image, result_img
                gc.collect()

            if not self._is_cancelled:
                self.progress_updated.emit(100, "100% Completed!")
                self.finished.emit("Translation completed. Please review the previews and click Save.")

        except Exception as e:
            self.error_occurred.emit(f"SYSTEM ERROR: {str(e)}")