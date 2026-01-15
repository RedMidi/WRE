import os
import base64
import json
import re
from pathlib import Path
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()
API_KEY = os.getenv("DASHSCOPE_API_KEY")

client = OpenAI(
    api_key=API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

BASE_DIR = Path(__file__).parent.parent
IMAGE_DIR = BASE_DIR / "data" / "images"
OUTPUT_DIR = BASE_DIR / "data" / "output"
OUTPUT_FILE = OUTPUT_DIR / "data.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

def clean_money_text(text):
    match = re.search(r"(\d+(\.\d+)?)", str(text))
    if match:
        return match.group(1)
    return text

def analyze_image(image_path):
    base64_img = encode_image(image_path)
    
    # up-to-bottom make better results for Qwen-vl-max
    system_prompt = """
    Please extract the list records from the WeChat red packet screenshot.
    Rules:
    1. Ignore irrelevant labels such as “Best Luck”.
    2. Extract strictly in the visual order from top to bottom as shown in the image. Do not reorder.
    
    Please return a JSON array directly (no Markdown formatting):
    [{“sender”: “Nickname”, “datetime”: “Time”, ‘amount’: “Amount”}, ...]
    """

    response = client.chat.completions.create(
        model="qwen-vl-max",
        messages=[
            {"role": "user", "content": [
                {"type": "text", "text": system_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}}
            ]}
        ],
        temperature=0.01,
    )
    
    content = response.choices[0].message.content.strip()
    
    if content.startswith("```"):
        content = content.replace("```json", "").replace("```", "")
    
    return json.loads(content)

def main():
    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    images = sorted([p for p in IMAGE_DIR.iterdir() if p.suffix.lower() in valid_exts])

    all_rows = []

    for img_path in tqdm(images, desc="Processing images"):
        records = analyze_image(img_path)
        
        records.reverse() # Reverse to maintain bottom-to-top order
        
        for index, item in enumerate(records, start=1):
            row = {
                'image_id': img_path.name,
                'user': item.get('user', 'Unknown'),
                'money': clean_money_text(item.get('money', '0')),
                'order_id': index
            }
            all_rows.append(row)

    df = pd.DataFrame(all_rows)
    
    columns_order = ['image_id', 'user', 'money', 'order_id']
    df = df[columns_order]
    
    df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')

if __name__ == "__main__":
    main()