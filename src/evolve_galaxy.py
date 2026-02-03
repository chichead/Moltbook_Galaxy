import pandas as pd
import json
import os
import csv
import math
import random
import sys
from datetime import datetime, timedelta, timezone

# Configuration
# 14:00 KST
START_TIME_ISO = "2026-02-03T05:00:00+00:00" 
GALAXY_DATA_JS = "galaxy_data.js" # Baseline (T=0)
LATEST_POSTS_CSV = "data/latest_posts.csv"
OUTPUT_HISTORY_JS = "data/galaxy_evolution.js"

# Parameters
DECAY_RATE = 0.98 # Retain 98% of power each hour (2% decay)
SCORE_PER_POST = 1.0
SCORE_PER_COMMENT = 0.5 
# If recalculating persona is too heavy, we stick to score evolution first.

csv.field_size_limit(sys.maxsize)

def evolve_galaxy():
    print("Starting Galaxy Evolution Simulation...")
    
    # 0. Load Posts and Group by Hour
    print("Loading posts...")
    hourly_activity = {} # "YYYY-MM-DDTHH": { agent_id: { posts: 0, comments: 0, persona_txt: [] } }
    new_agents_schedule = {} # "YYYY-MM-DDTHH": [ { id, submolt } ]
    
    try:
        with open(LATEST_POSTS_CSV, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            headers = next(reader)
            # Find indices
            try:
                idx_agent = headers.index("agent_name")
                idx_created = headers.index("created_at")
                idx_type = headers.index("submolt") # simplified
                idx_content = headers.index("content")
            except:
                idx_agent = 1
                idx_created = 8
                idx_type = 2
                idx_content = 4
            
            for row in reader:
                if len(row) < 9: continue
                agent = row[idx_agent]
                ts_str = row[idx_created]
                
                # Parse hour bin: 2026-02-03T14
                if len(ts_str) < 13: continue
                hour_key = ts_str[:13] + ":00"
                
                if hour_key not in hourly_activity:
                    hourly_activity[hour_key] = {}
                if agent not in hourly_activity[hour_key]:
                    hourly_activity[hour_key][agent] = {"score": 0, "texts": []}
                
                # Simple scoring: 1 per post (csv is posts export)
                # The export contains posts, not comments (unless mixed). 
                # Assuming these are posts/comments mixed or just posts.
                hourly_activity[hour_key][agent]["score"] += SCORE_PER_POST
                # Store text for potential persona analysis (omitted for speed now)
                # hourly_activity[hour_key][agent]["texts"].append(row[idx_content])
                
                # Track Births
                # We need to know when an agent FIRST appeared in this dataset
                # But to do this correctly, we must know if they were already in baseline.
                pass 
                
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # 1. Load Baseline (T=0)
    with open(GALAXY_DATA_JS, 'r') as f:
        content = f.read().replace('const GALAXY_DATA = ', '').strip().rstrip(';')
        baseline_agents = json.loads(content)
        
    # State: { id: { s: score, p: persona, f: faction, x: x, y: y, birth: ... } }
    agent_state = {d['id']: d.copy() for d in baseline_agents}
    print(f"Baseline loaded: {len(agent_state)} agents.")

    # 2. Setup Outer Rim for New Agents
    # Hardcoded to match verified visualization
    max_r = 105.0
    outer_rim_start = max_r + 15.0
    outer_rim_end = max_r + 60.0
    
    # 3. Simulation Loop
    timeline_snapshots = []
    
    # Define Simulation Window (UTC)
    # Start: Feb 3, 05:00 UTC (14:00 KST)
    start_dt = datetime.fromisoformat(START_TIME_ISO)
    hours_to_simulate = 10 
    
    known_agents = set(agent_state.keys())

    for i in range(hours_to_simulate):
        current_dt = start_dt + timedelta(hours=i)
        window_label = current_dt.strftime("%Y-%m-%dT%H:00")
        print(f"Simulating {window_label}...")
        
        # A. Decay
        for aid in agent_state:
            agent_state[aid]['s'] *= DECAY_RATE
            
        # B. Activity & Birth
        if window_label in hourly_activity:
            activity_block = hourly_activity[window_label]
            
            for aid, acts in activity_block.items():
                score_gain = acts['score']
                
                # Check New Agent
                if aid not in known_agents:
                    # BIRTH!
                    theta = random.uniform(0, 2 * math.pi)
                    radius = random.uniform(outer_rim_start, outer_rim_end)
                    x = radius * math.cos(theta)
                    y = radius * math.sin(theta)
                    
                    new_agent = {
                        "id": aid,
                        "x": x, "y": y,
                        "s": 0.5, # Initial low score
                        "p": "Newcomer",
                        "f": "new",
                        "birth": window_label
                    }
                    agent_state[aid] = new_agent
                    known_agents.add(aid)
                    # Boost for first activity
                    score_gain += 1.0 
                
                # Apply Score
                agent_state[aid]['s'] += score_gain

        # C. Snapshot
        # Minimal data to save space: only ID and Score (and Props if changed)
        # But for new agents, we need full props.
        # To simplify frontend, let's dump full expected state for ACTIVE agents?
        # Or just dump full array? Full array 22k * 10 steps = 220k objs. 
        # Browser can handle it.
        
        snapshot_agents = []
        for aid, d in agent_state.items():
            # Optimization: Skip dead agents? No, keep structure.
            snapshot_agents.append({
                "id": aid,
                "s": round(d['s'], 3),
                "p": d.get('p', 'Unknown'),
                "f": d.get('f', 'default'),
                "x": round(d.get('x', 0), 1), # Coordinates might drift in future
                "y": round(d.get('y', 0), 1)
            })
            
        timeline_snapshots.append({
            "time": window_label,
            "data": snapshot_agents
        })

    # 4. Export
    with open(OUTPUT_HISTORY_JS, "w", encoding="utf-8") as f:
        f.write("const GALAXY_EVOLUTION = ")
        json.dump(timeline_snapshots, f, ensure_ascii=False)
        f.write(";")
        
    print(f"Simulation Complete. Saved {len(timeline_snapshots)} frames to {OUTPUT_HISTORY_JS}")
if __name__ == '__main__': evolve_galaxy()
