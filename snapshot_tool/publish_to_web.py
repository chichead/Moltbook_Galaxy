# -*- coding: utf-8 -*-
"""
Web Publisher Bridge (Snapshot Tool -> Web Visualizer)
Copies research results to deployment folders and triggers the web pipeline.
"""

import os
import shutil
import subprocess

# --- Config ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SNAPSHOT_RESULTS = os.path.join(PROJECT_ROOT, "snapshot_tool", "results")
WEB_ANALYSIS_RESULTS = os.path.join(PROJECT_ROOT, "analysis_results")

# Source Files
COORDS_SRC = os.path.join(SNAPSHOT_RESULTS, "agent_coordinates.csv")
LEADERBOARD_SRC = os.path.join(SNAPSHOT_RESULTS, "agent_leaderboard_full.csv")
PERSONA_SRC = os.path.join(SNAPSHOT_RESULTS, "persona_distribution.csv")

# Destination Files (Mapping to expected v3 filenames)
COORDS_DST = os.path.join(WEB_ANALYSIS_RESULTS, "agent_coordinates_v3.csv")
LEADERBOARD_DST = os.path.join(WEB_ANALYSIS_RESULTS, "agent_leaderboard_full.csv")
PERSONA_DST = os.path.join(WEB_ANALYSIS_RESULTS, "agent_personas_v3.csv")

def publish():
    print("========================================")
    print("   Web Visualization Publisher        ")
    print("========================================")

    # 1. Verification
    if not os.path.exists(COORDS_SRC) or not os.path.exists(LEADERBOARD_SRC):
        print("[ERROR] Snapshot results missing. Please run 'python3 snapshot_tool/run_pipeline.py' first.")
        return

    # 2. Deployment
    print("\n[STEP 1/3] Copying coordinates and leaderboard to Web Analysis folder...")
    os.makedirs(WEB_ANALYSIS_RESULTS, exist_ok=True)
    shutil.copy2(COORDS_SRC, COORDS_DST)
    shutil.copy2(LEADERBOARD_SRC, LEADERBOARD_DST)
    shutil.copy2(PERSONA_SRC, PERSONA_DST)
    print(f"  - Deployed to: {WEB_ANALYSIS_RESULTS}")

    # 3. Trigger Web Pipeline
    print("\n[STEP 2/3] Generating Agent Master List...")
    try:
        subprocess.run(["python3", "src/generate_agent_master_list.py"], cwd=PROJECT_ROOT, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to generate master list: {e}")
        return

    print("\n[STEP 3/3] Exporting Final Galaxy Data (galaxy_data.js)...")
    try:
        subprocess.run(["python3", "src/export_galaxy_data.py"], cwd=PROJECT_ROOT, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to export galaxy data: {e}")
        return

    print("\n========================================")
    print("   ✅ SUCCESS: Web Data Updated!       ")
    print("   Refresh your browser to see changes.")
    print("========================================")

if __name__ == "__main__":
    publish()
