# -*- coding: utf-8 -*-
import os
import sys
import subprocess
import time

def run_pipeline():
    print("========================================")
    print("   Moltbook Snapshot Pipeline (v3)      ")
    print("========================================")
    
    # Check current directory
    cwd = os.getcwd()
    if not cwd.endswith("snapshot_tool"):
        # If run from root, we need to adjust paths.
        # But instructions say run from root. Let's make it work from root.
        pass

    required_files = [
        "snapshot_tool/data/moltbook.db",
        "snapshot_tool/data/moltbook_comments.csv",
        "snapshot_tool/data/moltbook_agents.csv",
        "snapshot_tool/data/moltbook_posts.csv"
    ]
    
    print("\n[STEP 1] Verifying Data Files...")
    missing = [f for f in required_files if not os.path.exists(f)]
    if missing:
        print(f"[ERROR] Missing files in snapshot_tool/data/:")
        for f in missing:
            print(f" - {os.path.basename(f)}")
        sys.exit(1)
    print("[OK] All data files present.")

    # Run Influence Analysis
    print("\n[STEP 2] Running Influence Analysis (v3 logic)...")
    start = time.time()
    try:
        # Change dir to snapshot_tool to make relative paths in scripts work easily
        subprocess.run(["python3", "src/analyze_influence.py"], cwd="snapshot_tool", check=True)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Influence analysis failed: {e}")
        sys.exit(1)
    print(f"[OK] Completed in {time.time() - start:.1f}s")

    # Run Persona Analysis
    print("\n[STEP 3] Running Persona Analysis (v3 logic + Influence Map)...")
    start = time.time()
    try:
        subprocess.run(["python3", "src/analyze_personas.py"], cwd="snapshot_tool", check=True)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Persona analysis failed: {e}")
        sys.exit(1)
    print(f"[OK] Completed in {time.time() - start:.1f}s")

    print("\n========================================")
    print("   Pipeline Finished Successfully!       ")
    print("   Results: snapshot_tool/results/       ")
    print("========================================")
    print("- influence_map.png")
    print("- persona_distribution.csv")
    print("- agent_leaderboard_full.csv")

if __name__ == "__main__":
    run_pipeline()
