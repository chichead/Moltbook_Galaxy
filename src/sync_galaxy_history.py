import requests
import json
import time
from datetime import datetime, timedelta, timezone
import os

# Configuration
BASE_URL = "https://moltbook-observatory.sushant.info.np"
FEED_ENDPOINT = f"{BASE_URL}/api/feed"
GALAXY_DATA_JS = "galaxy_data.js"
OUTPUT_HISTORY_JSON = "data/galaxy_timelapse.json"
# Start time: 2 PM Today (Feb 3, 2026) in KST
START_TIME_KST = datetime(2026, 2, 3, 14, 0, 0)
# KST is UTC+9
START_TIME_UTC = START_TIME_KST - timedelta(hours=9)
START_TIME_ISO = START_TIME_UTC.replace(tzinfo=timezone.utc).isoformat()

def fetch_feed_paginated(since_iso):
    all_posts = []
    current_since = since_iso
    
    while True:
        # Use with Z to avoid 422 if it likes simpler ISO
        fmt = current_since.replace("+00:00", "Z")
        print(f"Fetching chunk starting from: {fmt}")
        
        params = {"since": fmt, "limit": 100}
        try:
            response = requests.get(FEED_ENDPOINT, params=params)
            if response.status_code != 200:
                print(f"Failed chunk with {response.status_code}: {response.text}")
                break
            
            chunk_raw = response.json()
            if not chunk_raw:
                break
            
            # If it's a dict, find the list of posts (usually 'posts' or just use values if one list)
            if isinstance(chunk_raw, dict):
                print(f"  Got dict with keys: {list(chunk_raw.keys())}")
                # Try common keys
                if 'posts' in chunk_raw:
                    chunk = chunk_raw['posts']
                elif 'data' in chunk_raw:
                    chunk = chunk_raw['data']
                else:
                    # Look for the first value that is a list
                    chunk = []
                    for v in chunk_raw.values():
                        if isinstance(v, list):
                            chunk = v
                            break
            else:
                chunk = chunk_raw
                
            if not chunk:
                print("  No list found in response.")
                break
            
            all_posts.extend(chunk)
            print(f"  Got {len(chunk)} posts (Total: {len(all_posts)})")
            
            # Update current_since to the last post's timestamp
            try:
                last_ts = chunk[-1]['created_at']
                current_since = last_ts
                
                # Check if we should continue
                if len(chunk) < 100:
                    break
            except Exception as te:
                print(f"  Error accessing timestamp: {te}")
                break
            
            # Simple safety break if chunk is incomplete
            if len(chunk) < 100:
                break
                
            time.sleep(0.5) # Be nice
        except Exception as e:
            print(f"Error: {e}")
            break
            
    return all_posts

def group_by_hour(posts):
    # Map UTC hour bits to data
    hourly_activity = {} # { "2026-02-03T06": { "agent_name": { "posts": 1, "score": 10 } } }
    
    for p in posts:
        # created_at is like "2026-02-03T06:12:34.567890+00:00"
        try:
            ts_str = p['created_at']
            # Simple parsing: Take the first 13 characters (YYYY-MM-DDTHH)
            # This is enough for hourly grouping
            hour_str = ts_str[:13]
            
            if hour_str not in hourly_activity:
                hourly_activity[hour_str] = {}
            
            agent = p['agent_name']
            if agent not in hourly_activity[hour_str]:
                hourly_activity[hour_str][agent] = {"p": 0, "s": 0}
            
            hourly_activity[hour_str][agent]["p"] += 1
            hourly_activity[hour_str][agent]["s"] += p.get('score', 0)
        except Exception as e:
            print(f"Error parsing timestamp {ts_str}: {e}")
            continue
            
    return hourly_activity

import csv
import sys

# Increase CSV field size limit to handle large content
csv.field_size_limit(sys.maxsize)

def load_posts_from_csv(filename):
    posts = []
    print(f"Loading posts from {filename}...")
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            # created_at is at index 8 based on previous inspection
            # id,agent_name,submolt,title,content,link,score,comment_count,created_at
            for i, row in enumerate(reader):
                if len(row) < 9:
                    continue
                # Simple heuristic to skip header if it exists
                if "created_at" in row[8]:
                    continue
                    
                try:
                    posts.append({
                        "agent_name": row[1],
                        "score": int(row[6]) if row[6] else 0,
                        "created_at": row[8]
                    })
                except ValueError:
                    continue
    except Exception as e:
        print(f"Error reading CSV: {e}")
    return posts

def run_sync():
    # 1. Load data from CSV
    csv_file = "data/latest_posts.csv"
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found. Please download it first.")
        return

    all_posts = load_posts_from_csv(csv_file)
    print(f"Total posts loaded: {len(all_posts)}")
    
    if not all_posts:
        return

    # Filter for the specific test window (14:00 KST onwards)
    cutoff_time = START_TIME_UTC
    cutoff_iso = cutoff_time.replace(tzinfo=timezone.utc).isoformat()
    
    print(f"Filtering posts since {cutoff_iso} (Test Window Start)...")
    
    recent_posts = []
    for p in all_posts:
        # String comparison works for ISO format
        if p['created_at'] >= cutoff_iso:
            recent_posts.append(p)
            
    print(f"Posts in range: {len(recent_posts)}")

    # 2. Group by hour
    hourly_history = group_by_hour(recent_posts)
    sorted_hours = sorted(hourly_history.keys())
    print(f"Activity found in {len(sorted_hours)} unique hours.")
    
    if not sorted_hours:
        print("No activity found in the selected range.")
        return

    # 3. Format for Frontend
    history_data = {
        "start_time": sorted_hours[0] + ":00:00+00:00", # Approximate start
        "hours": sorted_hours,
        "deltas": {} 
    }

    hour_to_idx = {h: i for i, h in enumerate(sorted_hours)}

    for h_str in sorted_hours:
        h_idx = hour_to_idx[h_str]
        for agent, stats in hourly_history[h_str].items():
            if agent not in history_data["deltas"]:
                history_data["deltas"][agent] = []
            
            history_data["deltas"][agent].append({
                "h": h_idx,
                "p": stats["p"],
                "s": stats["s"]
            })

    # Save to data/
    os.makedirs("data", exist_ok=True)
    with open(OUTPUT_HISTORY_JSON, "w", encoding="utf-8") as f:
        json.dump(history_data, f, ensure_ascii=False)
    
    print(f"Saved timelapse data to {OUTPUT_HISTORY_JSON}")
    
    # Also save a minified version for the web
    with open(OUTPUT_HISTORY_JSON.replace(".json", ".js"), "w", encoding="utf-8") as f:
        f.write(f"const GALAXY_HISTORY = {json.dumps(history_data, ensure_ascii=False)};")

if __name__ == "__main__":
    run_sync()
