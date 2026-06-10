#%%
import pandas as pd
from transformers import pipeline
from langdetect import detect
from tqdm import tqdm
import re
import os

emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None)
sentiment_classifier = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-sentiment")

def is_english(text):
    try: return detect(str(text)) == 'en'
    except: return False

def get_rich_analysis(text):
    text = str(text)
    results = {}
    
    try:
        emotions = emotion_classifier(text, truncation=True, max_length=512)[0]
        top_emotion = max(emotions, key=lambda x: x['score'])
        results['top_emotion'] = top_emotion['label']
        results['emotion_score'] = top_emotion['score']
        
        sent = sentiment_classifier(text, truncation=True, max_length=512)[0]
        results['sentiment_label'] = sent['label']
        
    except Exception as e:
        results['top_emotion'] = 'neutral'
        results['emotion_score'] = 0.0
        results['sentiment_label'] = 'LABEL_1'
    
    return results

def tag_issues(text):
    text = str(text).lower()
    rules = {
        'Monetization': r'(fee|$|billing|order|upgrade|scammers|trial|charge|pay|paid|cancelllation|buy|expensive|purchase|refund|lock|buying|spam)',
        'Technical': r'(update|troubleshoot|fix|hardware|blocked|battery|connect)',
        'Design': r'(features|widgets|function|UI|design|complicated|button)',
        'Localization': r'(language|chinese|translate)',
        'Account/ Membership': r'(username|membership|subscription)',
        'Customer Service': r'(customer service|response)',
        'Accuracy': r'(Unreliable|track|data|comprehensive|inconsistent|insight|unprofessional|professional|algorithm|accurate|AI|inaccurate|distance|incorrect)'
    }

    found_tags = [cat for cat, pat in rules.items() if re.search(pat, text)]
    primary_issue = found_tags[0] if found_tags else "General"
    all_tags = ", ".join(found_tags) if found_tags else "General"
    
    return primary_issue, all_tags

def run_full_analysis():
    df = pd.read_csv('bio_analysis/data/raw_reviews.csv')
    
    df = df[df['review'].apply(is_english)].copy()
    
    print("Analyzing!")
    tqdm.pandas()
    
    analysis_results = df['review'].progress_apply(get_rich_analysis)
    df_results = pd.DataFrame(analysis_results.tolist())
    df = pd.concat([df.reset_index(drop=True), df_results], axis=1)
    
    df['topic_tags'] = df['review'].apply(tag_issues)
    
    df.to_csv('data/processed_sentiment.csv', index=False)
    print("Completed!")

if __name__ == "__main__":
    run_full_analysis()

# %%
