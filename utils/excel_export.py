# utils/excel_export.py
import pandas as pd
import json
import os

def save_to_excel(rows, filename="Judgments_Clean.xlsx"):
    df = pd.DataFrame(rows)
    # reorder columns if present
    cols = [
        'doc_id', 'title', 'date_iso', 'court_std', 'judges', 'acts', 'sections_structured',
        'decision_label', 'outcome_summary', 'plain_summary', 'key_reasons', 'conditions',
        'conclusion', 'citations'
    ]
    cols = [c for c in cols if c in df.columns]
    df = df[cols]
    df.to_excel(filename, index=False)
    print(f"[EXCEL] Saved {len(df)} rows to {filename}")

def save_final_json(case_json, out_dir="final_results"):
    os.makedirs(out_dir, exist_ok=True)
    doc_id = case_json.get("doc_id", "unknown")
    path = os.path.join(out_dir, f"{doc_id}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(case_json, f, ensure_ascii=False, indent=2)
    print(f"[JSON] Saved final JSON -> {path}")
