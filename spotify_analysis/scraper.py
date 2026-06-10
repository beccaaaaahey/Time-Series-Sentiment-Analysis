#%%
import requests
import pandas as pd
import os
import time

def fetch_paged_data():
    app_id = "324684580"
    all_reviews = []
    
    for page in range(1, 11):
        print(f"Extracting Page {page}...")
        url = f"https://itunes.apple.com/rss/customerreviews/page={page}/id={app_id}/sortby=mostrecent/json?l=en&cc=us"
        
        try:
            response = requests.get(url, timeout=15)
            if response.status_code != 200:
                break
            
            data = response.json()
            feed = data.get('feed', {})
            entries = feed.get('entry', [])
            
            if not entries:
                break

            for entry in entries:
                if 'im:rating' not in entry:
                    continue
                
                all_reviews.append({
                    'date': entry.get('updated', {}).get('label'),
                    'review': entry.get('content', {}).get('label'),
                    'rating': int(entry.get('im:rating', {}).get('label')),
                    'title': entry.get('title', {}).get('label')
                })
            
            time.sleep(1)
            
        except Exception as e:
            print(f"Error in Page {page}: {e}")
            break

    if not all_reviews:
        print("Failed...")
        return

    df = pd.DataFrame(all_reviews)
    df = df.drop_duplicates(subset=['review'])
    df['date'] = pd.to_datetime(df['date']).dt.date
    
    os.makedirs('data', exist_ok=True)
    df.to_csv('../spotify_analysis/data/raw_reviews3.csv', index=False)
    print(f"Successfully extracted {len(df)} comments!")

if __name__ == "__main__":
    fetch_paged_data()
# %%
