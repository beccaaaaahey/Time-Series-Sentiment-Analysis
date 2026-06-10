#%%
import pandas as pd
from transformers import pipeline
from langdetect import detect
from tqdm import tqdm
import re
import os
import spacy

nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])

emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None)
sentiment_classifier = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-sentiment")

def is_english(text):
    try: return detect(str(text)) == 'en'
    except: return False

def get_lemmatized_text(text):
    doc = nlp(str(text).lower())
    return " ".join([token.lemma_ for token in doc])

def get_rich_analysis(text):
    text = str(text)
    results = {}
    
    emotion_map = {
        'joy': 'Positive',
        'surprise': 'Positive',
        'neutral': 'Neutral',
        'anger': 'Negative',
        'disgust': 'Negative',
        'fear': 'Negative',
        'sadness': 'Negative'
    }
    try:
        emotions = emotion_classifier(text, truncation=True, max_length=512)[0]
        top_emo_obj = max(emotions, key=lambda x: x['score'])
        
        raw_label = top_emo_obj['label'] 
        results['top_emotion'] = emotion_map.get(raw_label, 'Neutral')
        results['emotion_score'] = top_emo_obj['score']
        
        sent = sentiment_classifier(text, truncation=True, max_length=512)[0]
        results['sentiment_label'] = sent['label']
        
    except Exception:
        results['top_emotion'] = 'Neutral'
        results['emotion_score'] = 0.0
        results['sentiment_label'] = 'LABEL_1'
    return results

def tag_hierarchy(lemmatized_text):
    #Product Category
    product_rules = {
    'Podcast': r'podcast|show|episode',
    'Audiobook': r'audio\s?book|narrat',
    'Music': r'music|song|artist|album|genre|track',
    'Playlist/Shuffle': r'playlist|shuffle|random'
    }
    
    # Issue Category
    issue_rules = {
        'Monetization': r'ad|price|cost|subscription|premium|pay|free|billing|cancel',
        'Technical': r'crash|freeze|bug|lag|load|offline|download|skip|fix|slow|feature|update',
        'Design/UI': r'interface|button|layout|ui|design|navigation',
        'Recommendation': r'recommend|suggest|mix|algorithm|accurate|discovery',
        'Customer Service': r'support|customer service|help'
    }

    found_products = [prod for prod, pat in product_rules.items() if re.search(pat, lemmatized_text)]
    found_issues = [issue for issue, pat in issue_rules.items() if re.search(pat, lemmatized_text)]
    
    product_final = ", ".join(found_products) if found_products else "General"
    issue_final = ", ".join(found_issues) if found_issues else "General"
    
    return product_final, issue_final

def run_full_analysis():
    data = pd.read_csv('data/raw_reviews.csv')
    data2 = pd.read_csv('data/raw_reviews2.csv')
    data3 = pd.read_csv('data/raw_reviews3.csv')
    df = pd.concat([data, data2, data3], ignore_index=True)
    df = df[df['review'].apply(is_english)].copy()
    
    tqdm.pandas()
    df['lemmatized_review'] = df['review'].progress_apply(get_lemmatized_text)
    
    print("Analyzing Emotions & Sentiments...")
    analysis_results = df['review'].progress_apply(get_rich_analysis)
    df_results = pd.DataFrame(analysis_results.tolist())
    df = pd.concat([df.reset_index(drop=True), df_results], axis=1)
    
    print("Tagging Product Hierarchy...")
    hierarchy = df['lemmatized_review'].apply(tag_hierarchy)
    df[['product_type', 'issue_type']] = pd.DataFrame(hierarchy.tolist(), index=df.index)
    
    if not os.path.exists('data'): os.makedirs('data')
    df.to_csv('data/processed_sentiment.csv', index=False)
    print("Completed!")

if __name__ == "__main__":
    run_full_analysis()
# %%
