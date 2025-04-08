#!/usr/bin/env python
# src/advanced_feedback.py
import os
import pandas as pd

FEEDBACK_FILE = "feedback.log"

def load_weighted_feedback(threshold=2):
    """
    Loads weighted feedback from feedback.log.
    
    Expected format (tab-delimited):
      news_text <TAB> prediction <TAB> feedback [<TAB> weight]
      
    - "feedback" should be either "yes" or "no".  
    - If an optional weight is provided, it is used; otherwise, defaults to 1.
    
    Aggregates feedback by summing weights for "yes" and "no". If the total
    feedback (sum of weights) for a news item is at least the threshold, the final
    label is decided by majority (weighted vote). If "no" votes outweigh "yes" votes,
    the label is flipped compared to the original prediction.
    
    Returns a DataFrame with columns: 'combined_text', 'label', 'total_weight',
    'yes_weight', and 'no_weight'.
    """
    if not os.path.exists(FEEDBACK_FILE):
        return pd.DataFrame()

    feedback_dict = {}
    with open(FEEDBACK_FILE, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            news_text = parts[0].strip()
            prediction = parts[1].strip()
            feedback_val = parts[2].strip().lower()
            # Optional weight (if not present, default to 1.0)
            weight = 1.0
            if len(parts) >= 4:
                try:
                    weight = float(parts[3].strip())
                except:
                    weight = 1.0

            if news_text not in feedback_dict:
                feedback_dict[news_text] = {"yes": 0.0, "no": 0.0, "prediction": prediction}
            if feedback_val == "yes":
                feedback_dict[news_text]["yes"] += weight
            elif feedback_val == "no":
                feedback_dict[news_text]["no"] += weight

    aggregated_data = []
    for news_text, counts in feedback_dict.items():
        total = counts["yes"] + counts["no"]
        if total < threshold:
            continue
        original_pred = counts["prediction"].lower()
        if counts["yes"] >= counts["no"]:
            final_label = 1 if original_pred == "real" else 0
        else:
            final_label = 0 if original_pred == "real" else 1
        aggregated_data.append({
            "combined_text": news_text,
            "label": final_label,
            "total_weight": total,
            "yes_weight": counts["yes"],
            "no_weight": counts["no"]
        })
    return pd.DataFrame(aggregated_data)

if __name__ == "__main__":
    df = load_weighted_feedback(threshold=2)
    print("Aggregated weighted feedback:")
    print(df.head())