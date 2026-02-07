import pandas as pd
import json
import os
import re

PERSONA_PATH = "snapshot_tool/results/persona_distribution.csv"
LEADERBOARD_PATH = "snapshot_tool/results/agent_leaderboard_full.csv"
HISTORY_FILE = "data/galaxy_history_data.js"
NEW_TIMESTAMP = "2026-02-07 00:00"

def main():
    print("Loading data...")
    # Load Persona Data (Position & Persona)
    if not os.path.exists(PERSONA_PATH):
        print(f"Error: {PERSONA_PATH} not found.")
        return
    
    df_persona = pd.read_csv(PERSONA_PATH)
    
    # Filter out Observers (User requirement)
    original_count = len(df_persona)
    df_persona = df_persona[df_persona['persona'] != 'Observer']
    print(f"Filtered out Observers: {original_count} -> {len(df_persona)}")

    # Ensure required columns exist
    req_persona = ["name", "x", "y", "persona"]
    for c in req_persona:
        if c not in df_persona.columns:
            print(f"Error: Missing column '{c}' in {PERSONA_PATH}")
            return

    # Load Leaderboard Data (Status Index)
    if not os.path.exists(LEADERBOARD_PATH):
        print(f"Error: {LEADERBOARD_PATH} not found.")
        return

    df_lb = pd.read_csv(LEADERBOARD_PATH)
    # Ensure required columns exist (agent maps to name)
    if "agent" not in df_lb.columns or "status_index" not in df_lb.columns:
        print(f"Error: Missing columns in {LEADERBOARD_PATH}")
        return

    print(f"Loaded {len(df_persona)} persona records and {len(df_lb)} leaderboard records.")

    # Merge
    # We need status_index, pagerank, total_weight from leaderboard
    merged = pd.merge(df_persona, df_lb[["agent", "status_index", "pagerank", "total_weight"]], left_on="name", right_on="agent", how="left")
    
    print(f"Columns after merge: {merged.columns.tolist()}")
    
    if "status_index" not in merged.columns:
        if "status_index_y" in merged.columns:
             merged["status_index"] = merged["status_index_y"].fillna(0.0)
        elif "status_index_x" in merged.columns:
             merged["status_index"] = merged["status_index_x"].fillna(0.0)
        else:
             print("Error: status_index column lost after merge.")
             return
    else:
        merged["status_index"] = merged["status_index"].fillna(0.0)

    # Convert pagerank and total_weight to numeric, handling missing
    if "pagerank" in merged.columns:
        merged["pagerank"] = merged["pagerank"].fillna(0.0)
    else:
        merged["pagerank"] = 0.0

    if "total_weight" in merged.columns:
        merged["total_weight"] = merged["total_weight"].fillna(0.0)
    else:
        merged["total_weight"] = 0.0

    # Prepare agents list
    agents_list = []
    for _, row in merged.iterrows():
        agent_data = {
            "name": row["name"],
            "x": float(row["x"]),
            "y": float(row["y"]),
            "persona": row["persona"],
            "status_index": float(row["status_index"]),
            "pagerank": float(row["pagerank"]),
            "interaction": float(row["total_weight"])
        }
        agents_list.append(agent_data)

    print(f"Prepared {len(agents_list)} agents for the new frame.")

    # Create new frame object
    new_frame = {
        "timestamp": NEW_TIMESTAMP,
        "agent_count": len(agents_list),
        "agents": agents_list
    }

    # Read existing history file
    if not os.path.exists(HISTORY_FILE):
        print(f"Error: {HISTORY_FILE} not found.")
        return

    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        content = f.read()

    # Extract JSON part
    # Assuming format: const GALAXY_EVOLUTION = [...]
    match = re.search(r"const GALAXY_EVOLUTION = (\[.*\])", content, re.DOTALL)
    if not match:
        print("Error: Could not parse GALAXY_EVOLUTION array from file.")
        return

    json_str = match.group(1)
    
    try:
        history_data = json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return

    # Check if timestamp already exists, if so replace, else append
    existing_idx = -1
    for i, frame in enumerate(history_data):
        if frame["timestamp"] == NEW_TIMESTAMP:
            existing_idx = i
            break
    
    if existing_idx != -1:
        print(f"Frame for {NEW_TIMESTAMP} already exists. Replacing...")
        history_data[existing_idx] = new_frame
    else:
        print(f"Appending new frame for {NEW_TIMESTAMP}...")
        history_data.append(new_frame)

    # Write back
    new_content = f"const GALAXY_EVOLUTION = {json.dumps(history_data)}" 
    
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        f.write(new_content)

    print(f"Successfully updated {HISTORY_FILE}")

if __name__ == "__main__":
    main()
