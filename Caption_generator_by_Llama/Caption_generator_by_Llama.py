import pandas as pd
import requests
import os
import time
import base64
from dotenv import load_dotenv
from openai import OpenAI

# --- Configuration ---
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

INPUT_FILE = r"C:\Users\olegl\Documents\6000_exprtiment\EXPERIMENTS_MERGED\DATA\6000merged_data_FINAL.csv"
OUTPUT_FILE = "llama_4_scout_results.csv"
PROMPT_FILE = "PROMPT.yaml"
MODEL_NAME = "meta-llama/llama-4-scout" 

# הגדרות הרצה
BATCH_LIMIT = 5000  # כמות שורות מקסימלית לעיבוד בכל הפעלה של הסקריפט

ID_COLUMN = "IMAGE_ID"
TEXT_COLUMN = "TEXT_POST" 
IMAGE_URL_COLUMN = "IMAGE_URL"
RESULT_COLUMN = "LLAMA4_SCOUT_CAPTION"

def safe_save(df, filename):
    """שומר את הקובץ בצורה אטומיסטית למניעת שחיתות נתונים"""
    temp_file = filename + ".tmp"
    df.to_csv(temp_file, index=False, encoding='utf-8-sig')
    if os.path.exists(filename):
        os.remove(filename)
    os.rename(temp_file, filename)

def clean_llm_response(text):
    if not text: return ""
    lines = text.split('\n')
    filtered_lines = [l for l in lines if not any(x in l.upper() for x in ["STEP ", "ANALYSIS:", "OUTPUT:", "FORMATTING"])]
    clean_text = " ".join(filtered_lines)
    clean_text = clean_text.replace('"', "'").replace('\r', '').replace('\n', ' ')
    return clean_text.strip()[:250]

def get_image_as_base64(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        return base64.b64encode(response.content).decode('utf-8')
    except Exception as e:
        print(f"   [ERROR] Download failed: {str(e)}")
        return None

def main():
    print("\n" + "="*60)
    print(f"      LLAMA 4 SCOUT - BATCH PROCESSING ({BATCH_LIMIT} rows)")
    print("="*60)

    if not os.path.exists(PROMPT_FILE):
        print(f"[FATAL] Prompt file {PROMPT_FILE} missing.")
        return
    
    # טעינת התבנית (Template) מה-YAML פעם אחת בתחילת הריצה
    with open(PROMPT_FILE, "r", encoding="utf-8") as f:
        instructions_template = f.read()
    
    # 1. טעינה או יצירה של קובץ הפלט
    if not os.path.exists(OUTPUT_FILE):
        print(f"[*] Creating initial output file...")
        df_working = pd.read_csv(INPUT_FILE)
        df_working.columns = df_working.columns.str.strip()
        df_working[RESULT_COLUMN] = ""
        safe_save(df_working, OUTPUT_FILE)
    else:
        print(f"[*] Loading existing output file: {OUTPUT_FILE}")
        df_working = pd.read_csv(OUTPUT_FILE)
        df_working.columns = df_working.columns.str.strip()

    # 2. זיהוי שורות שטרם עובדו
    df_working[RESULT_COLUMN] = df_working[RESULT_COLUMN].fillna("").astype(str)
    pending_mask = (df_working[RESULT_COLUMN] == "") | (df_working[RESULT_COLUMN].str.contains("Error:", na=False))
    rows_to_process = df_working[pending_mask]
    
    total_pending = len(rows_to_process)
    print(f"[*] Found {total_pending} rows left to process.")
    
    if total_pending == 0:
        print("[!] All tasks completed! No rows left.")
        return

    # הגבלת כמות השורות להרצה הנוכחית
    current_batch = rows_to_process.head(BATCH_LIMIT)
    print(f"[*] Starting this run: Processing next {len(current_batch)} rows.")
    print("="*60)

    processed_in_this_run = 0

    # 3. לולאת העיבוד
    try:
        for idx in current_batch.index:
            row = df_working.loc[idx]
            img_id = str(row[ID_COLUMN])
            
            processed_in_this_run += 1
            print(f"\n--> [{processed_in_this_run}/{len(current_batch)}] ID: {img_id}")
            
            img_url = row[IMAGE_URL_COLUMN]
            
            # חילוץ הטקסט מהפוסט וטיפול במקרה של שדה ריק
            post_text = str(row[TEXT_COLUMN]) if not pd.isna(row[TEXT_COLUMN]) else "No text provided"
            
            # --- הזרקת הטקסט ישירות לתוך הוראות המערכת ---
            current_instructions = instructions_template.replace("[INSERT_TEXT_POST]", post_text)
            
            b64_img = get_image_as_base64(img_url)
            if not b64_img:
                result = "Error: Download Failed"
            else:
                try:
                    print(f"   [...] API Call...", end="", flush=True)
                    response = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=[
                            # שליחת ההוראות המעודכנות הכוללות את תוכן הפוסט
                            {"role": "system", "content": current_instructions},
                            {"role": "user", "content": [
                                # שליחת התמונה בלבד בהודעת המשתמש
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}}
                            ]}
                        ],
                        temperature=0.1
                    )
                    result = clean_llm_response(response.choices[0].message.content)
                    print(f" DONE -> {result}")
                except Exception as e:
                    print(f" FAILED: {str(e)[:50]}")
                    result = f"Error: {str(e)[:50]}"

            # עדכון ושמירה (בכל שורה, בצורה בטוחה)
            df_working.at[idx, RESULT_COLUMN] = result
            safe_save(df_working, OUTPUT_FILE)
            
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n\n[!] User stopped the process. Data is saved.")

    print("\n" + "="*60)
    print(f"RUN FINISHED. Processed {processed_in_this_run} rows.")
    if 'idx' in locals():
        print(f"Next time you run this, it will pick up from row {idx + 2}.")
    print("="*60)

if __name__ == "__main__":
    main()