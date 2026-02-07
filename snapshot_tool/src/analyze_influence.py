# -*- coding: utf-8 -*-
"""
Moltbook Social & Influence Analysis (Trustworthy v3) - Snapshot Tool Edition
Identical to archives/analyze_moltbook_chatgpt.py
"""

import os
import sqlite3
import math
import json
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from collections import Counter, defaultdict
from dataclasses import dataclass

@dataclass
class Config:
    data_dir: str = "data"
    output_dir: str = "results"
    db_path: str = "data/moltbook.db"
    agents_csv_path: str = "data/moltbook_agents.csv"
    posts_csv_path: str = "data/moltbook_posts.csv"
    comments_csv_path: str = "data/moltbook_comments.csv"
    csv_chunksize: int = 100_000
    pagerank_max_nodes: int = 5000
    top_edges_to_draw: int = 400
    top_nodes_to_label: int = 25
    random_seed: int = 42

cfg = Config()
os.makedirs(cfg.output_dir, exist_ok=True)

def save_df(df: pd.DataFrame, path: str):
    df.to_csv(path, index=False)
    print(f"[OK] Saved: {path} ({len(df)} rows)")

def iter_csv_comments(path: str, chunksize: int):
    if not os.path.exists(path): return
    for chunk in pd.read_csv(path, usecols=["post_id", "agent_name"], chunksize=chunksize):
        yield chunk

def accumulate_comment_edges_from_df(df, post_author_map, edge_counter, coverage):
    for r in df.itertuples():
        coverage["rows_total"] += 1
        src = str(r.agent_name)
        pid = str(r.post_id)
        tgt = post_author_map.get(pid)
        if tgt is None:
            coverage["rows_missing_post"] += 1; continue
        if src == tgt:
            coverage["rows_self_reply"] += 1; continue
        edge_counter[(src, tgt)] += 1
        coverage["rows_used"] += 1

def analyze_comments(posts: pd.DataFrame, db_comments: pd.DataFrame = None):
    print("[COMMENT] Analyzing interaction network...")
    post_author_map = dict(zip(posts["id"].astype(str), posts["agent_name"].fillna("").astype(str)))
    edge_counter = Counter()
    coverage = defaultdict(int)

    if db_comments is not None and not db_comments.empty:
        accumulate_comment_edges_from_df(db_comments, post_author_map, edge_counter, coverage)
    if os.path.exists(cfg.comments_csv_path):
        for chunk in iter_csv_comments(cfg.comments_csv_path, cfg.csv_chunksize):
            accumulate_comment_edges_from_df(chunk, post_author_map, edge_counter, coverage)

    edges = pd.DataFrame([(s, t, w) for (s, t), w in edge_counter.items()], columns=["src", "tgt", "weight"])
    out_w = edges.groupby("src")["weight"].sum()
    in_w = edges.groupby("tgt")["weight"].sum()
    nodes = pd.DataFrame({"agent": pd.Index(out_w.index).union(in_w.index)})
    nodes["out_weight"] = nodes["agent"].map(lambda a: float(out_w.get(a, 0)))
    nodes["in_weight"] = nodes["agent"].map(lambda a: float(in_w.get(a, 0)))
    nodes["total_weight"] = nodes["out_weight"] + nodes["in_weight"]

    top_agents = nodes.sort_values("total_weight", ascending=False).head(cfg.pagerank_max_nodes)["agent"].tolist()
    edges_pr = edges[edges["src"].isin(top_agents) & edges["tgt"].isin(top_agents)].copy()
    G = nx.DiGraph()
    for r in edges_pr.itertuples(index=False): G.add_edge(r.src, r.tgt, weight=float(r.weight))
    pr = nx.pagerank(G, weight="weight") if G.number_of_nodes() > 0 else {}
    nodes["pagerank"] = nodes["agent"].map(lambda a: float(pr.get(a, 0.0)))
    return nodes, edges

def build_agent_leaderboard(agents: pd.DataFrame, comment_nodes: pd.DataFrame, timestamp: str) -> pd.DataFrame:
    print("[LEADERBOARD] building agent leaderboard...")
    base = agents[["name", "karma", "follower_count", "following_count"]].copy()
    base.rename(columns={"name": "agent"}, inplace=True)
    if comment_nodes is None or comment_nodes.empty:
        for c in ["pagerank", "in_weight", "out_weight", "total_weight"]: base[c] = 0.0
        lb = base
    else:
        lb = pd.merge(base, comment_nodes, on="agent", how="outer")
        for c in ["karma", "follower_count", "following_count", "pagerank", "in_weight", "out_weight", "total_weight"]:
            if c in lb.columns: lb[c] = pd.to_numeric(lb[c], errors="coerce").fillna(0.0)
    
    lb["score_struct"] = np.log1p(lb["total_weight"]) + 5.0 * lb["pagerank"]
    lb["score_reach"] = np.log1p(lb["follower_count"]) + 0.5 * np.log1p(lb["karma"].clip(lower=0))
    lb["status_index"] = 0.7 * lb["score_struct"] + 0.3 * lb["score_reach"]
    lb.sort_values("status_index", ascending=False, inplace=True)
    
    ts_lb_path = os.path.join(cfg.output_dir, f"agent_leaderboard_full_{timestamp}.csv")
    
    save_df(lb, ts_lb_path)
    save_df(lb, os.path.join(cfg.output_dir, "agent_leaderboard_full.csv")) # Latest
    return lb

def main():
    print("[INFO] Starting Influence Analysis (v3 verbatim logic)...")
    conn = sqlite3.connect(cfg.db_path)
    posts = pd.read_sql_query("SELECT id, agent_name FROM posts", conn)
    try: db_comments = pd.read_sql_query("SELECT post_id, agent_name FROM comments", conn)
    except: db_comments = None
    agents = pd.read_sql_query("SELECT name, karma, follower_count, following_count FROM agents", conn)
    conn.close()

    # Detect the latest timestamp from data to use for filename versioning
    print("[INFO] Detecting latest data timestamp for versioning...")
    # Load timestamps from DB
    conn = sqlite3.connect(cfg.db_path)
    latest_p = pd.read_sql_query("SELECT MAX(created_at) as ts FROM posts", conn).iloc[0]['ts']
    latest_c = pd.read_sql_query("SELECT MAX(created_at) as ts FROM comments", conn).iloc[0]['ts']
    conn.close()
    
    latest_dt = pd.to_datetime(max(latest_p, latest_c))
    latest_kst = latest_dt + timedelta(hours=9)
    # Using MMDD_HHMM format for consistency across scripts
    timestamp = latest_kst.strftime("%m%d_%H%M")
    print(f"[INFO] Data-driven timestamp detected: {timestamp}")

    comment_nodes, _ = analyze_comments(posts, db_comments)
    build_agent_leaderboard(agents, comment_nodes, timestamp)
    print(f"[DONE] Influence results saved to {cfg.output_dir}")

if __name__ == "__main__":
    main()
