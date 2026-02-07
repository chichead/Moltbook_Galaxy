# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import timedelta, datetime

# --- Config ---
POSTS_CSV = "snapshot_tool/data/moltbook_posts.csv"
COMMENTS_CSV = "snapshot_tool/data/moltbook_comments.csv"
RESULTS_DIR = "snapshot_tool/results"
SUMMARY_CSV = os.path.join(RESULTS_DIR, "activity_summary_kst.csv")
CHART_PNG = os.path.join(RESULTS_DIR, "activity_bar_chart.png")
CHART_PDF = os.path.join(RESULTS_DIR, "activity_bar_chart.pdf")

def run_analysis():
    print("========================================")
    print("   Moltbook Activity Analysis Tool      ")
    print("========================================")
    
    # 1. Load & Process Data
    print("\n[STEP 1] Loading and aggregating activity...")
    if not os.path.exists(POSTS_CSV) or not os.path.exists(COMMENTS_CSV):
        print("[ERROR] Source CSV files not found in snapshot_tool/data/")
        return

    posts = pd.read_csv(POSTS_CSV, usecols=["created_at"])
    comments = pd.read_csv(COMMENTS_CSV, usecols=["created_at"])
    
    posts['dt_utc'] = pd.to_datetime(posts['created_at'], errors='coerce')
    comments['dt_utc'] = pd.to_datetime(comments['created_at'], errors='coerce')
    
    # Hour-level grouping (KST)
    posts['hour_kst'] = (posts['dt_utc'] + timedelta(hours=9)).dt.floor('h')
    comments['hour_kst'] = (comments['dt_utc'] + timedelta(hours=9)).dt.floor('h')
    
    p_counts = posts.groupby('hour_kst').size().reset_index(name='post_count')
    c_counts = comments.groupby('hour_kst').size().reset_index(name='comment_count')
    
    # Merge and Calculate
    summary = pd.merge(p_counts, c_counts, on='hour_kst', how='outer').fillna(0)
    summary = summary.sort_values('hour_kst')
    summary['hour_utc'] = summary['hour_kst'] - timedelta(hours=9)
    summary['cumulative_posts'] = summary['post_count'].cumsum().astype(int)
    summary['cumulative_comments'] = summary['comment_count'].cumsum().astype(int)

    # Reorder columns
    cols = ['hour_utc', 'hour_kst', 'post_count', 'comment_count', 'cumulative_posts', 'cumulative_comments']
    summary = summary[cols]

    # Detect the latest timestamp from data to use for filename versioning
    # We use the raw dt_utc to get the exact latest moment in the dataset
    latest_dt = max(posts['dt_utc'].max(), comments['dt_utc'].max())
    # Convert to KST for the filename to match user expectations (0205_0950 style)
    latest_kst = latest_dt + timedelta(hours=9)
    timestamp = latest_kst.strftime("%m%d_%H%M")
    print(f"[INFO] Data-driven timestamp detected: {timestamp}")
    
    ts_summary_csv = os.path.join(RESULTS_DIR, f"activity_summary_kst_{timestamp}.csv")
    summary.to_csv(ts_summary_csv, index=False)
    summary.to_csv(SUMMARY_CSV, index=False) # Keep latest for web
    print(f"[OK] Summary CSV saved: {ts_summary_csv}")
    print(f"[INFO] Generic version updated: {SUMMARY_CSV}")

    # 2. Plotting
    print("\n[STEP 2] Generating activity bar chart...")
    plt.figure(figsize=(20, 10), facecolor='white')
    width = 0.35
    x = range(len(summary))
    
    plt.bar([i - width/2 for i in x], summary['post_count'], width, label='Posts', color='#ef476f', alpha=0.8)
    plt.bar([i + width/2 for i in x], summary['comment_count'], width, label='Comments', color='#118ab2', alpha=0.8)
    
    plt.title('Moltbook Hourly Activity (KST)', fontsize=20, weight='bold', pad=20)
    plt.xlabel('Time (KST)', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    
    # X-labels formatting
    label_indices = range(0, len(summary), 6)
    plt.xticks(label_indices, [summary['hour_kst'].iloc[i].strftime('%Y-%m-%d %H:00') for i in label_indices], rotation=45, ha='right')
    
    plt.legend(fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    # Generate timestamped filenames
    ts_chart_png = os.path.join(RESULTS_DIR, f"activity_bar_chart_{timestamp}.png")
    ts_chart_pdf = os.path.join(RESULTS_DIR, f"activity_bar_chart_{timestamp}.pdf")
    
    plt.savefig(ts_chart_png, dpi=200)
    plt.savefig(ts_chart_pdf)
    
    # Keep generic latest version
    plt.savefig(CHART_PNG, dpi=200)
    plt.savefig(CHART_PDF)
    plt.close()
    
    print(f"[OK] Bar charts saved: {ts_chart_png}, {ts_chart_pdf}")
    print(f"[INFO] Generic versions updated: {CHART_PNG}, {CHART_PDF}")
    print("\n[DONE] Analysis complete.")

if __name__ == "__main__":
    run_analysis()
