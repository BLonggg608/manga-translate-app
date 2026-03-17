import os
from helper_func import get_best_gpu_id, filter_mask_by_boxes, manga_sort_boxes, get_system_prompt, create_user_payload, parse_json_output, render_text_on_manga

HOME = os.getcwd()
# set env variable
# if you having multiple gpus and want to set specific gpu
os.environ['CUDA_VISIBLE_DEVICES'] = get_best_gpu_id()
# use this if you get CUDA Out of Memory errors
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
# set checkpoint file for simple lama
os.environ['LAMA_MODEL'] = f'{HOME}/models/simple-lama-inpainting/anime-manga-big-lama.pt'

from helper_func import filter_mask_by_boxes, manga_sort_boxes, get_system_prompt, create_user_payload, parse_json_output, render_text_on_manga
from models.text_segmentation.model import MangaTextSegmenter
from simple_lama_inpainting import SimpleLama
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer, BitsAndBytesConfig
from manga_ocr import MangaOcr
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import torch
import re
import gc


def natural_sort_key(s):
    # sort files properly (1, 2, 3... 10) instead of (1, 10, 2)
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]


def load_models(detector_model_path, ocr_model_path, segmenter_model_path, translate_model_path):
    # load models

    # yolov8s
    yolov8s = YOLO(detector_model_path)

    # segmentation model
    segmenter = MangaTextSegmenter(model_path=segmenter_model_path)

    # manga ocr
    mocr = MangaOcr(pretrained_model_name_or_path=ocr_model_path)

    # translate model
    # load quantization config
    compute_dtype = torch.bfloat16 if 'gemma-3-4b' in translate_model_path.lower() else torch.float16
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )

    if 'gemma-3-4b' in translate_model_path.lower():
        translate_model_tokenizer = AutoTokenizer.from_pretrained(
            translate_model_path)
        translate_model = AutoModelForImageTextToText.from_pretrained(
            translate_model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=False,
            dtype=compute_dtype
        )
    else:
        translate_model_tokenizer = AutoTokenizer.from_pretrained(
            translate_model_path)
        translate_model = AutoModelForCausalLM.from_pretrained(
            translate_model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=False,
            dtype=compute_dtype
        )

    # simple lama
    simple_lama = SimpleLama()

    return yolov8s, segmenter, mocr, translate_model_tokenizer, translate_model, simple_lama


def translate_manga_pipeline(image_path, detector, segmenter, mocr, translator_model_path,
                             translator_tokenizer, translator_model, simple_lama, previous_translation_text=None,
                             genre='General Manga', target_language='English',
                             font_path=f'{HOME}/assets/fonts/anime_ace_bb/animeace2bb_tt/animeace2_reg.ttf'):

    # move models to gpu
    detector.to('cuda')
    segmenter.model.to('cuda')
    mocr.model.to('cuda')

    # 1. text detection
    # load image
    image = Image.open(image_path).convert("RGB")

    # get text masks
    text_mask_image = segmenter.segment(image)

    # get text boxes
    boxes = detector.predict(image, conf=0.1, imgsz=1024)[
        0].boxes.xyxy.cpu().numpy().tolist()

    # filter text masks by boxes to remove non-text areas in masks
    text_mask_image = filter_mask_by_boxes(text_mask_image, boxes)

    # 2. ocr
    # sort boxes
    boxes = manga_sort_boxes(boxes)

    # crop and ocr
    jp_texts = []
    for box in boxes:
        cropped_img = image.crop(tuple(box))
        text = mocr(cropped_img)
        jp_texts.append(text)
        del cropped_img

    # move models to cpu to save memory
    detector.to('cpu')
    segmenter.model.to('cpu')
    mocr.model.to('cpu')

    torch.cuda.empty_cache()
    gc.collect()

    # 3. translation
    system_prompt = get_system_prompt(genre=genre, target_language=target_language)
    current_user_payload = create_user_payload(jp_texts) # Lấy payload của trang hiện tại

    if previous_translation_text:
        combined_prompt = f"""
{system_prompt}

---
CONTEXT FROM PREVIOUS PAGE (Use this for consistency in names/tone):
{previous_translation_text}

---
TRANSLATE THE FOLLOWING NEW PAGE:
{current_user_payload}
"""
    else:
        # first page won't have previous context: use system prompt + current payload
        combined_prompt = f"{system_prompt}\n\n---\n\n{current_user_payload}"

    messages = []
    
    if 'gemma' in translator_model_path.lower():
        messages = [{"role": "user", "content": combined_prompt}]
    # elif 'gemma-3-4b' in translator_model_path.lower():
    #     messages = [{"role": "user", "content": [{"type": "text", "text": combined_prompt}]}]
    else:
        messages = [{"role": "user", "content": combined_prompt}]

    # tokenizer
    input_ids = translator_tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt"
    ).to(translator_model.device)

    # generate translation
    with torch.no_grad():
        output_ids = translator_model.generate(
            **input_ids,
            max_new_tokens=4048,
            temperature=0.5,
            do_sample=True,
            top_p=0.95,
        )

    # decode output
    generated_ids = output_ids[0][input_ids["input_ids"].shape[-1]:].cpu()
    # skip special tokens and trim whitespace
    response_text = translator_tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    # parse output
    translated_texts = parse_json_output(response_text, len(jp_texts))

    # prepare context for next page
    next_page_context = response_text

    del input_ids, output_ids, generated_ids
    torch.cuda.empty_cache()
    gc.collect()

    # 4. remove origin text
    # dilate mask
    text_mask_image = np.array(text_mask_image)
    dilation_size = 21
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (dilation_size, dilation_size))
    text_mask_image = cv2.dilate(text_mask_image, kernel, iterations=1)
    text_mask_image = Image.fromarray(text_mask_image)

    # inpaint
    cleared_image = simple_lama(image, text_mask_image)

    # clean memory
    del image, text_mask_image
    torch.cuda.empty_cache()
    gc.collect()

    # 5. render text
    try:
        result_img = render_text_on_manga(
            cleared_image, boxes, translated_texts, font_path)
    except Exception as e:
        print(f"Error: {e}")
        result_img = cleared_image

    del cleared_image
    gc.collect()

    return result_img, next_page_context


def process_batch(input_dir, output_dir, detector, segmenter, mocr, translator_model_path,
                  translator_tokenizer, translator_model, simple_lama, 
                  genre='General Manga', target_language='English', font_path=f'{HOME}/assets/fonts/anime_ace_bb/animeace2bb_tt/animeace2_reg.ttf'):

    # get valid images and sort
    valid_ext = ('.jpg', '.jpeg', '.png', '.webp')
    images = [f for f in os.listdir(
        input_dir) if f.lower().endswith(valid_ext)]
    images.sort(key=natural_sort_key)

    # create run folder
    os.makedirs(output_dir, exist_ok=True)
    run_dir = os.path.join(
        output_dir, f'run_{len(os.listdir(output_dir)) + 1}')
    os.makedirs(run_dir, exist_ok=True)

    print(f"processing {len(images)} images from {input_dir}...")

    # initialize previous messages for context
    previous_translation_text = None

    for idx, img_name in enumerate(images):
        print(f"\n[{idx + 1}/{len(images)}] processing {img_name}")
        img_path = os.path.join(input_dir, img_name)

        # run pipeline
        result_img, previous_translation_text = translate_manga_pipeline(
            image_path=img_path,
            detector=detector,
            segmenter=segmenter,
            mocr=mocr,
            translator_model_path=translator_model_path,
            translator_tokenizer=translator_tokenizer,
            translator_model=translator_model,
            simple_lama=simple_lama,
            previous_translation_text=previous_translation_text,
            genre=genre,
            target_language=target_language,
            font_path=font_path
        )

        # save result
        save_path = os.path.join(run_dir, img_name)
        result_img.save(save_path)
        print(f"saved to {save_path}")


def main():
    # model paths
    yolov8s_model_path = f'{HOME}/models/text-detector/comic-text-segmenter.pt'
    ocr_model_path = f"{HOME}/models/manga-ocr/models/manga-ocr-base"

    # translation model path
    # translate_model_path = f'{HOME}/models/translate-model/models/gemma-2-9b-it'
    # translate_model_path = f"{HOME}/models/translate-model/models/gemma-2-9b-it-abliterated"
    translate_model_path = f"{HOME}/models/translate-model/models/tiger-gemma-9b-v3"

    segment_model_path = f'{HOME}/models/text_segmentation/model.pth'

    # directories
    # input_dir = '/home/tranhabaolong608/Downloads/frieren_chap_137'
    # input_dir = '/mnt/wwn-0x50c82d5000000206-part2/DiepClassC/Testing/bin/CrackedGames/Folder/Manga/New/test/origin'
    input_dir = "C:/Users/ADMIN/Downloads/frieren_chap_147"
    output_dir = f'{HOME}/output'

    # load models
    detector, segmenter, mocr, translate_model_tokenizer, translate_model, simple_lama = load_models(
        detector_model_path=yolov8s_model_path,
        ocr_model_path=ocr_model_path,
        segmenter_model_path=segment_model_path,
        translate_model_path=translate_model_path
    )

    # process entire folder
    process_batch(
        input_dir=input_dir,
        output_dir=output_dir,
        detector=detector,
        segmenter=segmenter,
        mocr=mocr,
        translator_model_path=translate_model_path,
        translator_tokenizer=translate_model_tokenizer,
        translator_model=translate_model,
        simple_lama=simple_lama,
        # genre='Fantasy, Adventure',
        # target_language='Vietnamese',
        font_path=f'{HOME}/assets/fonts/animeace2_viethoa_reg.ttf'
    )


if __name__ == "__main__":
    main()
