import pandas as pd
import requests
import json
import os
import time
from dotenv import load_dotenv

# טעינת הגדרות
load_dotenv()
API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL = "meta-llama/llama-4-scout" 
INPUT_FILE = "MERGED_ALL.csv"
OUTPUT_FILE = "MERGED_WITH_JUDGMENTS.csv"

def get_llm_judgment(image_url, text_post, topics):
    """
    שולח קריאה ל-OpenRouter. 
    הוסר ה-while True כדי לאפשר ל-MAIN לשלוט בניסיונות החוזרים.
    """
    prompt = f"""The Final Multimodal Judge Prompt (Optimized for Accuracy)

System Role:
You are a Senior BERTopic Auditor specializing in multimodal Twitter analysis. Your task is to evaluate the accuracy of topic predictions for posts that consist of two synchronized components:
The Post Text: The literal content and sentiment of the tweet.
The Post Image: The visual context and embedded information (OCR) from the attached file.
For each post, you will receive multiple predicted topics. Your objective is to judge their validity by determining how well they capture the unified context of both the text and the visual data.

Evaluation Categories (Definitions):
You must use these EXACT strings for the labels:
- "[Irrelevant] – No logical connection to the image/text, or completely misleading."
- "[Poor] – Tangentially related, too broad, or misses the main point/visual context."
- "[Good] – Accurate and identifies main entities from both image and text, but might miss a small nuance."
- "[Perfect] – Spot-on. Precisely captures the core intent and specific context of both the visual and textual data."

Your Process:
1. Vision & OCR: Analyze the attached image. Extract text via OCR and identify key visual elements/context.
2. Context Fusion: Combine the image data with the TEXT_POST to establish the "Ground Truth".
3. Reasoning (Chain of Thought): For each model, first write a 10-15 word reasoning explaining the alignment (or lack thereof) between the topic and the Ground Truth.
4. Final Labeling: Only after the reasoning, select and write the FULL category line (Label + Definition) exactly as quoted in the list above.

Output Format (Strict):
MODEL_NAME:
Reasoning: [15 words max]
Label: [Full Label - Full Definition]
Special Case: If any input topic is "-1", strictly output Reasoning: "-1" and Label: "-1" for that model.

(Note: A topic of "-1" represents unclassified noise and should not be analyzed).

Input Data:
TEXT_POST: {text_post}
Llama Topic: {topics['llama']}
Gemini Topic: {topics['gemini']}
OCR Topic: {topics['ocr']}
Baseline Topic: {topics['baseline']}
"""

    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}}
                ]
            }
        ]
    }

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
            data=json.dumps(payload),
            timeout=60
        )
        
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        else:
            print(f"   !! API Error {response.status_code}. Content: {response.text[:100]}")
            return None
    except Exception as e:
        print(f"   !! Connection Error: {e}")
        return None

def parse_model_output(response_text, model_key):
    """מחלץ נתונים לפי הפורמט הקשיח של הפרומפט"""
    if not response_text: return "Error", "Error"
    try:
        parts = response_text.split(f"{model_key}:")
        if len(parts) < 2: return "Not Found", "Not Found"
        
        block = parts[1].split("\n\n")[0].strip()
        reasoning, label = "N/A", "N/A"
        
        for line in block.split('\n'):
            line = line.strip()
            if line.startswith("Reasoning:"):
                reasoning = line.replace("Reasoning:", "").strip()
            if line.startswith("Label:"):
                label = line.replace("Label:", "").strip()
        return reasoning, label
    except:
        return "Parse Error", "Parse Error"

def main():
    # --- הגדרות התחלה ---
    START_IMAGE_ID = "HCenIAkWEAAZvmM" 
    MAX_RETRIES = 3 # מקסימום ניסיונות לשורה לפני כניעה
    WAIT_BETWEEN_RETRIES = 5 # המתנה מינימלית בשניות
    # ---------------------

    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found!")
        return

    df = pd.read_csv(INPUT_FILE)
    
    if os.path.exists(OUTPUT_FILE):
        existing_df = pd.read_csv(OUTPUT_FILE)
        processed_ids = set(existing_df['IMAGE_ID'].astype(str))
        print(f"Resuming... {len(processed_ids)} records already processed.")
    else:
        processed_ids = set()
        new_cols = ['JUDGE_LLAMA_REASON', 'JUDGE_LLAMA_LABEL', 'JUDGE_GEMINI_REASON', 'JUDGE_GEMINI_LABEL',
                    'JUDGE_OCR_REASON', 'JUDGE_OCR_LABEL', 'JUDGE_BASELINE_REASON', 'JUDGE_BASELINE_LABEL']
        header_df = pd.DataFrame(columns=list(df.columns) + new_cols)
        header_df.to_csv(OUTPUT_FILE, index=False)

    reached_start_id = True if START_IMAGE_ID is None else False

    for index, row in df.iterrows():
        img_id = str(row['IMAGE_ID'])

        if not reached_start_id:
            if img_id == START_IMAGE_ID:
                reached_start_id = True
                print(f"Found START_IMAGE_ID: {img_id}. Starting process...")
            else:
                continue 

        if img_id in processed_ids: 
            continue

        print(f"[{index+1}/{len(df)}] Processing ID: {img_id}...")
        
        topics = {
            'llama': str(row['TOPIC_BY_LLAMA']),
            'gemini': str(row['TOPIC_BY_GEMINI']),
            'ocr': str(row['TOPIC_BY_OCR']),
            'baseline': str(row['TOPIC_BY_BASELINE'])
        }

        results = {}
        needs_api_call = False
        
        for key in ['llama', 'gemini', 'ocr', 'baseline']:
            if topics[key] == "-1":
                results[key] = ("-1", "-1")
            else:
                needs_api_call = True 

        if needs_api_call:
            response = None
            # --- מנגנון ניסיונות חוזרים וכניעה ---
            for attempt in range(MAX_RETRIES):
                response = get_llm_judgment(row['IMAGE_URL'], row['TEXT_POST'], topics)
                if response:
                    break # הצלחנו! יוצאים מהלופ של הניסיונות
                
                if attempt < MAX_RETRIES - 1:
                    print(f"   -> Attempt {attempt+1} failed. Retrying in {WAIT_BETWEEN_RETRIES}s...")
                    time.sleep(WAIT_BETWEEN_RETRIES)
                else:
                    print(f"   !! Surrendering on ID {img_id} after {MAX_RETRIES} attempts.")

            # חילוץ תוצאות (אם response הוא None, הפונקציה parse_model_output תחזיר "Error")
            for key_name, topic_key in [("Llama Topic", "llama"), ("Gemini Topic", "gemini"), 
                                        ("OCR Topic", "ocr"), ("Baseline Topic", "baseline")]:
                if topics[topic_key] != "-1":
                    if response:
                        results[topic_key] = parse_model_output(response, key_name)
                    else:
                        results[topic_key] = ("API_SKIP_ERROR", "API_SKIP_ERROR")
            
            time.sleep(1.2) # מרווח בטיחות קטן בין שורות
        else:
            print(f"   >> All topics are noise (-1) for ID {img_id}. Skipping API.")

        # פירוק והצבה
        l_reas, l_lab = results.get('llama', ("-1", "-1"))
        g_reas, g_lab = results.get('gemini', ("-1", "-1"))
        o_reas, o_lab = results.get('ocr', ("-1", "-1"))
        b_reas, b_lab = results.get('baseline', ("-1", "-1"))

        # שמירה לקובץ
        new_row = list(row.values) + [l_reas, l_lab, g_reas, g_lab, o_reas, o_lab, b_reas, b_lab]
        pd.DataFrame([new_row]).to_csv(OUTPUT_FILE, mode='a', header=False, index=False)

    print(f"--- Task finished! Results in: {OUTPUT_FILE} ---")
    
if __name__ == "__main__":
    main()
