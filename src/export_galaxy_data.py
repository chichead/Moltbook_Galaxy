import pandas as pd
import json
import os
import math
import random

RESULTS_DIR = "analysis_results"
MASTER_LIST_CSV = os.path.join(RESULTS_DIR, "agent_master_list_v3.csv")
COORDINATES_CSV = os.path.join(RESULTS_DIR, "agent_coordinates_v3.csv")
OUTPUT_JS = "galaxy_data.js"

def export_galaxy():
    print("Building Galaxy Data...")
    
    if not os.path.exists(MASTER_LIST_CSV) or not os.path.exists(COORDINATES_CSV):
        print("Error: Required CSV files missing.")
        return

    # Load data
    master_df = pd.read_csv(MASTER_LIST_CSV)
    coords_df = pd.read_csv(COORDINATES_CSV)

    # Filter out Observers
    # master_df = master_df[master_df['persona'] != 'Observer']

    # Merge on agent name
    merged = pd.merge(master_df, coords_df, on="agent", how="inner")
    
    # Calculate max radius of non-observer agents to define Outer Rim
    max_r = 0
    for row in merged.itertuples():
        if row.persona != 'Observer':
            
    # Hardcoded Core Radius to prevent runaway expansion
    # Based on t-SNE results, core is within ~105
    max_r = 105.0 
    print(f"Fixed Core Galaxy Radius: {max_r:.2f}")
    
    # Unified Outer Rim Parameters (used by both Observers and Newcomers)
    
    # Unified Outer Rim Parameters (used by both Observers and Newcomers)
    # Start slightly further out to separate from core
    outer_rim_start = max_r + 15.0 
    outer_rim_end = max_r + 60.0
    
    print(f"Outer Rim Range: {outer_rim_start:.1f} - {outer_rim_end:.1f}")

    # Prepare list of stars (agents)
    stars = []
    
    # Track re-positioned count
    obs_moved = 0
    
    for row in merged.itertuples():
        x = float(row.x)
        y = float(row.y)
        
        # Override for Observers
        if row.persona == 'Observer':
            theta = random.uniform(0, 2 * math.pi)
            radius = random.uniform(outer_rim_start, outer_rim_end)
            x = radius * math.cos(theta)
            y = radius * math.sin(theta)
            obs_moved += 1

        stars.append({
            "id": str(row.agent),
            "x": x,
            "y": y,
            "p": str(row.persona),
            "f": int(row.faction_id),
            "s": float(row.status_index),
            "pr": float(row.pagerank),
            "k": int(row.karma),
            "w": int(row.total_weight)
        })
        
    print(f"Repositioned {obs_moved} Observers to Outer Rim.")

    # Export to JS
    with open(OUTPUT_JS, "w", encoding="utf-8") as f:
        f.write("const GALAXY_DATA = ")
        json.dump(stars, f, ensure_ascii=False)
        f.write(";")
        
    print(f"Done! {len(stars)} agents exported to {OUTPUT_JS}")

if __name__ == "__main__":
    export_galaxy()
