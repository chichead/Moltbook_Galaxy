import os
import pandas as pd
import json
from datetime import datetime

EVOLUTION_DIR = "snapshot_tool/results/evolution"
OUTPUT_JSON = "results/galaxy_history.json"
OUTPUT_JS = "results/galaxy_history_data.js"

def main():
    if not os.path.exists(EVOLUTION_DIR):
        print(f"[ERROR] Evolution directory not found: {EVOLUTION_DIR}")
        return

    # Get all snapshot directories and sort by date
    snapshot_dirs = [d for d in os.listdir(EVOLUTION_DIR) if d.startswith("snapshot_") and d.endswith("_0000")]
    snapshot_dirs.sort()

    history_data = []

    print(f"[CONSOLIDATE] Found {len(snapshot_dirs)} snapshots. Processing...")

    for snapshot_dir in snapshot_dirs:
        # Extract date from dir name (e.g., 0204 from snapshot_0204_0000)
        date_str = snapshot_dir.split("_")[1]
        timestamp = f"2026-{date_str[:2]}-{date_str[2:]} 00:00"
        
        csv_path = os.path.join(EVOLUTION_DIR, snapshot_dir, "snapshot_personas.csv")
        if not os.path.exists(csv_path):
            # Just skip without loud error if it's missing, but tell user
            continue

        print(f"  - Processing {timestamp}...")
        df = pd.read_csv(csv_path, low_memory=False)
        
        # Select essential columns
        # Some older versions might have slightly different names (name vs agent), handling both
        if "name" in df.columns:
            name_col = "name"
        elif "agent" in df.columns:
            name_col = "agent"
        else:
            print(f"    - [ERROR] Neither 'name' nor 'agent' found in {csv_path}")
            continue

        # Filter out Observers to keep JSON size manageable (Optional, but usually preferred for viz)
        # However, let's keep all for now so the user has the full picture, or just non-observers?
        # User requested continuity, so mostly interested in the active stars.
        active_df = df[df["persona"] != "Observer"].copy()
        
        agents_data = active_df[[name_col, "x", "y", "persona", "status_index"]].rename(columns={name_col: "name"}).to_dict(orient="records")
        
        history_data.append({
            "timestamp": timestamp,
            "agent_count": len(agents_data),
            "agents": agents_data
        })

    # Save to JSON
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(history_data, f, ensure_ascii=False, indent=2)

    # Save to JS for easy web loading
    with open(OUTPUT_JS, "w", encoding="utf-8") as f:
        f.write("const GALAXY_EVOLUTION = ")
        json.dump(history_data, f, ensure_ascii=False)
        f.write(";")

    print(f"\n[DONE] Consolidated history saved to:")
    print(f"  - JSON: {OUTPUT_JSON} (For Server API)")
    print(f"  - JS: {OUTPUT_JS} (For Static loading)")
    print(f"Total timestamps: {len(history_data)}")
    print(f"File size (JSON): {os.path.getsize(OUTPUT_JSON) / (1024*1024):.2f} MB")

if __name__ == "__main__":
    main()
