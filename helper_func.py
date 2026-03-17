import json
import re
import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import subprocess

def get_best_gpu_id():
    """
    This function scans all available Nvidia GPUs using nvidia-smi and selects the one with the most free VRAM.
    """
    try:
        # run nvidia-smi command to get GPU information
        result = subprocess.check_output(
            [
                'nvidia-smi', 
                '--query-gpu=index,memory.total,memory.used',
                '--format=csv,nounits,noheader'
            ], 
            encoding='utf-8'
        )
        
        best_gpu_id = "0"
        max_free_vram = -1
        
        # parse the output line by line
        for line in result.strip().split('\n'):
            # split data: "0, 24576, 1024" -> gpu_id=0, total=24576, used=1024
            gpu_id, total_vram, used_vram = map(int, line.split(','))
            
            # calculate free VRAM
            free_vram = total_vram - used_vram
            
            if free_vram > max_free_vram:
                max_free_vram = free_vram
                best_gpu_id = str(gpu_id)

        return best_gpu_id
        
    except FileNotFoundError:
        print("[ERROR] nvidia-smi not found. Make sure you have Nvidia drivers installed. Defaulting to GPU 0.")
        return "0"
    except Exception as e:
        print(f"[ERROR] Failed to get GPU information: {e}. Defaulting to GPU 0.")
        return "0"


def get_center(box):
    """calculate the center point of a box"""
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def manga_sort_boxes(boxes, threshold=15):
    # Step 1: Calculate centroids and average height
    if not boxes:
        return []

    # Preliminary sort from Top to Bottom (by Y center)
    # box[1] is y1, box[3] is y2. Take (y1+y2)/2
    boxes_with_center = []
    for box in boxes:
        cx, cy = get_center(box[:4])
        boxes_with_center.append(
            {'box': box, 'cx': cx, 'cy': cy, 'h': box[3] - box[1]})

    # Sort by Y first to iterate from top to bottom
    boxes_with_center.sort(key=lambda k: k['cy'])

    # Step 2: Group into "Virtual Rows"
    rows = []
    current_row = []

    if boxes_with_center:
        current_row.append(boxes_with_center[0])

        for i in range(1, len(boxes_with_center)):
            prev = current_row[-1]
            curr = boxes_with_center[i]

            # Important logic: Does the current box belong to the same row as the previous box?
            # If its Y center lies within the height range of the previous box (or vice versa)
            # We use a threshold to be flexible.

            y_diff = abs(curr['cy'] - prev['cy'])
            avg_height = (curr['h'] + prev['h']) / 2

            # If the Y difference is less than 50% of the average height -> Considered the same row
            if y_diff < (avg_height * 0.5):
                current_row.append(curr)
            else:
                # End the old row, push it into the rows list
                rows.append(current_row)
                # Create a new row
                current_row = [curr]

        # Push the last row in
        if current_row:
            rows.append(current_row)

    # Step 3: Within each row, sort from Right to Left
    sorted_boxes = []
    for row in rows:
        # Sort by CX descending (largest X = rightmost -> first)
        row.sort(key=lambda k: k['cx'], reverse=True)

        for item in row:
            sorted_boxes.append(item['box'])

    return sorted_boxes


def get_system_prompt(genre="General Manga", target_language="English"):
    if target_language.lower() in ["english", "en"]:
        few_shot_examples = """INPUT: [ {{"id": 0, "original": "まさか..."}}, {{"id": 1, "original": "お前が犯人なのか？"}} ]
OUTPUT: [ {{"id": 0, "translation": "No way..."}}, {{"id": 1, "translation": "Are you the culprit?"}} ]

INPUT: [ {{"id": 0, "original": "逃げるな！"}} ]
OUTPUT: [ {{"id": 0, "translation": "Don't you run!"}} ]

INPUT: [ {{"id": 0, "original": "あいつはまだ来てない"}} ]
OUTPUT: [ {{"id": 0, "translation": "He hasn't come yet."}} ] (Note: Used "He" instead of guessing a name like "Stark")

INPUT: [ {{"id": 0, "original": "雪、危ない！"}} ] 
OUTPUT: [ {{"id": 0, "translation": "Yuki, look out!"}} ] (Context: Yuki is a name, not "Snow")"""

    elif target_language.lower() in ["vietnamese", "vi"]:
        few_shot_examples = """INPUT: [ {{"id": 0, "original": "まさか..."}}, {{"id": 1, "original": "お前が犯人なのか？"}} ]
OUTPUT: [ {{"id": 0, "translation": "Không thể nào..."}}, {{"id": 1, "translation": "Cậu là thủ phạm à?"}} ]

INPUT: [ {{"id": 0, "original": "逃げるな！"}} ]
OUTPUT: [ {{"id": 0, "translation": "Đừng có chạy!"}} ]

INPUT: [ {{"id": 0, "original": "あいつはまだ来てない"}} ]
OUTPUT: [ {{"id": 0, "translation": "Hắn vẫn chưa đến."}} ] (Note: Used "Hắn" which is a generic pronoun instead of guessing a name like "Stark")

INPUT: [ {{"id": 0, "original": "雪、危ない！"}} ] 
OUTPUT: [ {{"id": 0, "translation": "Yuki, cẩn thận!"}} ] (Context: Yuki is a name, not "Snow")"""

    prompt = f"""Role: You are a professional Manga Localizer. 
Your goal is to adapt Japanese, Chinese and Korean dialogue into natural, colloquial {target_language} suitable for Western comic readers.
Target Genre: {genre}

### GUIDELINES:
1. **Localization over Translation:** Do not translate literally. Capture the *intent* and *emotion*.
   - Lit: "It cannot be helped." -> Loc: "Guess I have no choice."
2. **Contextual Grammar:**
   - Japanese ignores subjects -> You MUST add them (I, You, He, She) based on flow.
   - Read all text first to determine context.

3. **Proper Nouns & Subject Handling (CRITICAL):**
   - **NO NAME HALLUCINATION:** You are STRICTLY FORBIDDEN from inserting character names (e.g., "Frieren", "Fern", "Naruto") unless that name explicitly appears in the Japanese source text (as Kanji/Katakana/Hiragana).
   - **Implicit Subjects:** If the Japanese sentence omits the subject (Zero Pronoun), use pronouns (**I, You, He, She, They**) or generic nouns. DO NOT fill the gap with a specific character name based on your guess.
     - JP: "なぜ泣いているの？" (Why are [you] crying?)
     - WRONG: "Why are you crying, Fern?" (If "Fern" is not in the text).
     - RIGHT: "Why are you crying?"
   - **Transliteration:** When a name DOES appear, transliterate it phonetically (Romaji). Do not translate meanings (e.g., "Sakura" remains "Sakura").
   - **Honorifics:** Keep honorifics (e.g., "-san", "-kun", "-chan") as is.

4. **Tone:**
   - Use contractions (I'm, It's, Don't).
   - Use slang if the genre fits.

### FEW-SHOT EXAMPLES:
{few_shot_examples}

### OUTPUT FORMAT:
You must speak STRICTLY in JSON. Do not output any markdown text, explanations, or code blocks. Just the raw JSON list."""
    
    return prompt


def create_user_payload(text_list):
    input_data = [{"id": i, "original": txt}
                  for i, txt in enumerate(text_list)]
    json_str = json.dumps(input_data, ensure_ascii=False, indent=2)

    return f"""TRANSLATE THIS DATA:
{json_str}"""


def parse_json_output(raw_output, expected_length):
    try:
        # Extract JSON part using regex
        match = re.search(r"(\[.*\])", raw_output, re.DOTALL)
        clean_json = match.group(1) if match else raw_output

        data = json.loads(clean_json)

        # Create a map from id to translation
        result_map = {item.get('id'): item.get(
            'translation', '') for item in data}

        # Reconstruct the final list in order
        final_list = [result_map.get(i, "") for i in range(expected_length)]
        return final_list

    except Exception as e:
        print(f"⚠️ Parser Error: {e}")
        # Debug: print raw output
        print(f"DEBUG Output: {raw_output[:100]}...")
        return ["[Translation Error]"] * expected_length


def create_masked_image(image, boxes):
    # Convert PIL Image to numpy array
    img_np = np.array(image)

    # Create a mask of the same height and width, initialized to 0 (black)
    mask = np.zeros(img_np.shape[:2], dtype=np.uint8)

    # For each box, fill the corresponding area in the mask with 255 (white)
    for box in boxes:
        x1, y1, x2, y2 = map(int, box[:4])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_np.shape[1], x2), min(img_np.shape[0], y2)

        cv2.rectangle(mask, (x1, y1), (x2, y2), color=255,
                      thickness=-1)  # filled rectangle

    # dilate the mask slightly to cover edges
    dilation_size = 21
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (dilation_size, dilation_size))

    mask = cv2.dilate(mask, kernel, iterations=1)

    return mask  # Return as 2D numpy array for proper channel handling


def filter_mask_by_boxes(mask_image, boxes):
    is_pil = isinstance(mask_image, Image.Image)
    if is_pil:
        mask_np = np.array(mask_image)
    else:
        mask_np = mask_image.copy()

    filtered_mask_np = np.zeros_like(mask_np)

    # check if boxes is empty
    if not boxes:
        return Image.fromarray(filtered_mask_np)

    # for each box, copy the corresponding area from the original mask to the filtered mask
    for box in boxes:
        x1, y1, x2, y2 = map(int, box[:4])

        # ensure the box coordinates are within the image boundaries
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(mask_np.shape[1], x2)
        y2 = min(mask_np.shape[0], y2)

        # copy the area from the original mask to the filtered mask
        filtered_mask_np[y1:y2, x1:x2] = mask_np[y1:y2, x1:x2]

    # convert back to PIL Image if the input was PIL
    if is_pil:
        return Image.fromarray(filtered_mask_np)

    return filtered_mask_np


def text_wrap(text, font, max_width):
    lines = []
    # If the text has explicit newlines, split them first
    paragraphs = text.split('\n')

    for paragraph in paragraphs:
        words = paragraph.split()
        if not words:
            continue

        current_line = words[0]

        for word in words[1:]:
            # Check width if we add the next word
            test_line = current_line + " " + word
            # usage of getlength is preferred over getbbox for width in newer Pillow
            if font.getlength(test_line) <= max_width:
                current_line = test_line
            else:
                lines.append(current_line)
                current_line = word

        lines.append(current_line)

    return lines


def render_text_on_manga(cleared_image, bboxes, translated_texts, font_path=None):
    # Create a copy to avoid modifying the original
    image = cleared_image.copy()
    draw = ImageDraw.Draw(image)

    # Standard Manga Style: Black text, White outline
    TEXT_COLOR = (0, 0, 0)      # Black
    OUTLINE_COLOR = (255, 255, 255)  # White

    # Constraints
    MIN_FONT_SIZE = 12
    MAX_FONT_SIZE = 90
    PADDING_RATIO = 0.05  # 5% padding inside the box

    # If no font is provided, try to load a default, though it won't look "manga-like"
    default_font_ref = "arial.ttf"

    for bbox, text in zip(bboxes, translated_texts):
        if not text:
            continue

        # 1. Box Geometry
        x1, y1, x2, y2 = map(int, bbox[:4])
        box_w = x2 - x1
        box_h = y2 - y1

        # Apply padding to ensure text doesn't touch the box edges
        pad_x = int(box_w * PADDING_RATIO)
        pad_y = int(box_h * PADDING_RATIO)
        safe_w = max(10, box_w - (pad_x * 2))  # Ensure safe_w is not negative
        safe_h = max(10, box_h - (pad_y * 2))

        # 2. Iterative Font Sizing (Find the largest size that fits)
        final_font = None
        final_lines = []
        final_total_h = 0

        # Iterate from largest font size down to smallest
        for size in range(MAX_FONT_SIZE, MIN_FONT_SIZE - 1, -2):
            try:
                if font_path:
                    font = ImageFont.truetype(font_path, size)
                else:
                    # Fallback to system font if path not provided
                    font = ImageFont.truetype(default_font_ref, size)
            except OSError:
                # If loading fails, use default bitmap font (size cannot be changed)
                font = ImageFont.load_default()
                # Default font has fixed size, so we break the loop immediately
                lines = text_wrap(text, font, safe_w)
                final_font = font
                final_lines = lines
                break

            # Check Wrap
            lines = text_wrap(text, font, safe_w)

            # Check Height
            # getmetrics returns (ascent, descent). Line height = ascent + descent
            ascent, descent = font.getmetrics()
            line_height = ascent + descent
            total_h = line_height * len(lines)

            # check width
            max_line_w = 0
            if lines:
                max_line_w = max(font.getlength(line) for line in lines)

            # If it fits vertically and horizontally, we found our font!
            if total_h <= safe_h and max_line_w <= safe_w:
                final_font = font
                final_lines = lines
                final_total_h = total_h
                break

        # Fallback: If text is too long even for MIN_FONT_SIZE,
        # use the last configured font (MIN_FONT_SIZE) and let it overflow.
        if final_font is None and font_path:
            final_font = ImageFont.truetype(font_path, MIN_FONT_SIZE)
            final_lines = text_wrap(text, final_font, safe_w)
            ascent, descent = final_font.getmetrics()
            final_total_h = (ascent + descent) * len(final_lines)

        # 3. Calculate Rendering Position (Center Vertically)
        center_y = y1 + (box_h / 2)
        current_y = center_y - (final_total_h / 2)

        # Determine stroke width relative to font size (Standard Manga Look)
        # Usually 12.5% of font size is good. Minimum 4px.
        try:
            font_size = final_font.size
            stroke_width = max(4, int(font_size / 8))
        except:
            stroke_width = 4  # fallback for default font

        # 4. Draw Text Line by Line
        ascent, descent = final_font.getmetrics()
        line_height = ascent + descent

        for line in final_lines:
            # Calculate Horizontal Center
            line_w = final_font.getlength(line)
            center_x = x1 + (box_w / 2) - (line_w / 2)

            # Draw Outline (Stroke) + Text
            draw.text(
                (center_x, current_y),
                line,
                font=final_font,
                fill=TEXT_COLOR,
                stroke_width=stroke_width,
                stroke_fill=OUTLINE_COLOR
            )
            current_y += line_height

    return image

def natural_sort_key(s):
    # sort files properly (1, 2, 3... 10) instead of (1, 10, 2)
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]
