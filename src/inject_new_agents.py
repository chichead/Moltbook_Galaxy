import json
import csv
import sys
import os
import random
import math
from datetime import datetime

# Configuration
GALAXY_JS = "galaxy_data.js"
LATEST_POSTS_CSV = "data/latest_posts.csv"
OUTPUT_JS = "galaxy_data.js" # Overwrite
csv.field_size_limit(sys.maxsize)

# Start time: 2 PM Today (Feb 3, 2026) in KST
START_TIME_KST = datetime(2026, 2, 3, 14, 0, 0)
START_TIME_ISO = "2026-02-03T05:00:00+00:00" # UTC

def inject_new_agents():
    print("Identifying and Injecting New Agents...")
    
    # 1. Load Existing Galaxy Data
    if not os.path.exists(GALAXY_JS):
        print("Error: galaxy_data.js not found detected.")
        return
        
    with open(GALAXY_JS, 'r', encoding='utf-8') as f:
        content = f.read().replace('const GALAXY_DATA = ', '').strip().rstrip(';')
        galaxy_data = json.loads(content)
        
    existing_ids = set(d['id'] for d in galaxy_data)
    print(f"Loaded {len(galaxy_data)} existing agents.")
    
    # 2. Identify New Agents from Posts
    if not os.path.exists(LATEST_POSTS_CSV):
        print("Error: latest_posts.csv not found.")
        return
        
    new_agents = {} # agent_name -> { birth_time, faction? }
    
    try:
        with open(LATEST_POSTS_CSV, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader) # header
            for row in reader:
                if len(row) < 9: continue
                
                agent_name = row[1]
                created_at = row[8]
                submolt = row[2] # maybe use for faction inference?
                
                if agent_name not in existing_ids:
                    if agent_name not in new_agents:
                        # First appearance becomes birth time
                        new_agents[agent_name] = {
                            "birth": created_at,
                            "submolt": submolt
                        }
                    else:
                        # Keep earliest time
                        if created_at < new_agents[agent_name]["birth"]:
                            new_agents[agent_name]["birth"] = created_at

    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # Filter strictly for agents born AFTER the start time
    filtered_agents = {k: v for k, v in new_agents.items() if v["birth"] >= START_TIME_ISO}
    print(f"Found {len(filtered_agents)} new agents born after {START_TIME_ISO}.")

    if not filtered_agents:
        # print("No new agents to inject.")
        # Actually, let's just proceed with empty if none, or return.
        # But we need to use filtered_agents as new_agents
        pass
        
    new_agents = filtered_agents

    print(f"Found {len(new_agents)} new agents.")
    
    if not new_agents:
        print("No new agents to inject.")
        return

    # 3. Create New Agent Objects
    # We place them in the "Outer Rim" (Radius > 80 assuming current max is ~60-70)
    # Or "Nebula of Newcomers"
    
    # Hardcoded Core Radius to prevent runaway expansion (Matches export_galaxy_data.py)
    max_r = 105.0
    print(f"Fixed Core Galaxy Radius: {max_r:.2f}")
    
    # Unified Outer Rim Parameters (Matches export_galaxy_data.py)
    
    # Unified Outer Rim Parameters (Matches export_galaxy_data.py)
    outer_rim_start = max_r + 15.0
    outer_rim_end = max_r + 60.0
    
    injected_count = 0
    
    for agent_id, info in new_agents.items():
        # Random position in Outer Rim
        theta = random.uniform(0, 2 * math.pi)
        radius = random.uniform(outer_rim_start, outer_rim_end)
        
        x = radius * math.cos(theta)
        y = radius * math.sin(theta)
        
        # Infer Faction? 
        # For now, default to "Mixed" or check submolt mapping if we had it.
        # Let's map "Mixed" as default.
        faction_id = 99 # Special ID for Newcomers? or just 'default'
        persona = "Newcomer"
        
        new_entry = {
            "id": agent_id,
            "x": x,
            "y": y,
            "p": persona,
            "f": "new", # New faction code for coloring
            "s": 1.0, # Starting status
            "pr": 0.0,
            "k": 0,
            "w": 0,
            "birth": info["birth"] # Critical for timelapse
        }
        
        galaxy_data.append(new_entry)
        injected_count += 1
        
    # 4. Save Updated Galaxy Data
    with open(OUTPUT_JS, "w", encoding="utf-8") as f:
        f.write("const GALAXY_DATA = ")
        json.dump(galaxy_data, f, ensure_ascii=False)
        f.write(";")
        
    print(f"Successfully injected {injected_count} new agents.")
    print(f"Total Galaxy Population: {len(galaxy_data)}")

if __name__ == "__main__":
    inject_new_agents()
