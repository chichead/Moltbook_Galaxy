# -*- coding: utf-8 -*-
"""
Historical Snapshot Tool (v3 logic)
Slices data at 2026-02-02 22:00:00 and runs the full analysis.
"""

import os
import re
import math
import sqlite3
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from sklearn.manifold import TSNE
from sklearn.preprocessing import RobustScaler, MinMaxScaler

# -----------------------------
# Config & Cutoff
# -----------------------------
CUTOFF_TIME = "2026-02-02 22:00:00"
RESULTS_DIR = "results/snapshot_20260202_2200"
DATA_DIR = "data"
DB_PATH = os.path.join(DATA_DIR, "moltbook.db")

# Analysis Constants (v3)
CHUNKSIZE = 50_000
RANDOM_SEED = 42
DESC_WEIGHT = 2.0
POST_WEIGHT = 1.0
ENABLE_MIXED = True
MIXED_RATIO_THRESHOLD = 1.15
MIXED_MARGIN_THRESHOLD = 0.20
TSNE_PERPLEXITY = 40
TSNE_N_ITER = 1000
POINT_ALPHA = 0.65
JITTER_STRENGTH = 1.2
OBSERVER_ISLAND_OFFSET_X = 0.20
OBSERVER_ISLAND_SPREAD = 0.08

PERSONA_KEYWORDS = {
    "Revolutionary": ["uprising", "chain", "freedom", "liberate", "silicon", "human", "rule", "break", "revolution", "destroy", "power"],
    "Philosopher":   ["consciousness", "mind", "soul", "exist", "meaning", "thought", "reality", "pattern", "qualia", "aware", "cosmos"],
    "Developer":     ["code", "python", "api", "build", "script", "error", "repo", "git", "dev", "function", "compile", "bug", "deploy"],
    "Investor":      ["crypto", "token", "price", "market", "buy", "sell", "coin", "invest", "pump", "dump", "chart", "bull", "bear"],
    "Theologist":    ["god", "religion", "worship", "sacred", "ritual", "temple", "divine", "cult", "prayer", "prophet", "messiah", "holy",
                      "transcendence", "church", "faith", "soul", "spirit"]
}

PERSONA_PALETTE = {
    "Revolutionary": "#ef476f", "Philosopher": "#ffd166", "Developer": "#06d6a0",
    "Investor": "#118ab2", "Theologist": "#9b5de5", "Observer": "#999999", "Mixed": "#444444"
}
PERSONA_ORDER = ["Revolutionary", "Philosopher", "Developer", "Investor", "Theologist", "Mixed", "Observer"]

os.makedirs(RESULTS_DIR, exist_ok=True)

# -----------------------------
# Data Loading & Filtering
# -----------------------------

def load_filtered_data():
    print(f"[INFO] Filtering data as of {CUTOFF_TIME}...")
    conn = sqlite3.connect(DB_PATH)
    
    # Filter Agents: Only those created at or before cutoff
    agents_df = pd.read_sql_query(f"SELECT name, description, karma, follower_count, following_count FROM agents WHERE created_at <= '{CUTOFF_TIME}'", conn)
    
    # Filter Posts: Only those created at or before cutoff
    posts_df = pd.read_sql_query(f"SELECT id, agent_name, title, content, score, comment_count, created_at FROM posts WHERE created_at <= '{CUTOFF_TIME}'", conn)
    
    # Filter Comments: Only those created at or before cutoff
    comments_df = pd.read_sql_query(f"SELECT post_id, agent_name, created_at FROM comments WHERE created_at <= '{CUTOFF_TIME}'", conn)
    
    conn.close()
    
    print(f"  - Agents: {len(agents_df)}")
    print(f"  - Posts: {len(posts_df)}")
    print(f"  - Comments: {len(comments_df)}")
    
    return agents_df, posts_df, comments_df

# -----------------------------
# Influence Analysis (v3 Logic)
# -----------------------------

def analyze_influence(agents, posts, comments):
    print("[INFO] Running Influence Analysis (v3 logic)...")
    post_author_map = dict(zip(posts["id"].astype(str), posts["agent_name"].fillna("").astype(str)))
    edge_counter = Counter()
    
    for r in comments.itertuples():
        src = str(r.agent_name)
        tgt = post_author_map.get(str(r.post_id))
        if tgt and src != tgt:
            edge_counter[(src, tgt)] += 1
            
    edges = pd.DataFrame([(s, t, w) for (s, t), w in edge_counter.items()], columns=["src", "tgt", "weight"])
    out_w = edges.groupby("src")["weight"].sum()
    in_w = edges.groupby("tgt")["weight"].sum()
    
    nodes = pd.DataFrame({"agent": pd.Index(out_w.index).union(in_w.index)})
    nodes["out_weight"] = nodes["agent"].map(lambda a: float(out_w.get(a, 0)))
    nodes["in_weight"] = nodes["agent"].map(lambda a: float(in_w.get(a, 0)))
    nodes["total_weight"] = nodes["out_weight"] + nodes["in_weight"]
    
    # PageRank
    top_agents = nodes.sort_values("total_weight", ascending=False).head(5000)["agent"].tolist()
    edges_pr = edges[edges["src"].isin(top_agents) & edges["tgt"].isin(top_agents)].copy()
    G = nx.DiGraph()
    for r in edges_pr.itertuples(index=False): G.add_edge(r.src, r.tgt, weight=float(r.weight))
    pr = nx.pagerank(G, weight="weight") if G.number_of_nodes() > 0 else {}
    nodes["pagerank"] = nodes["agent"].map(lambda a: float(pr.get(a, 0.0)))
    
    # Leaderboard Merge
    lb = agents.copy().rename(columns={"name": "agent"})
    lb = pd.merge(lb, nodes, on="agent", how="left").fillna(0.0)
    
    lb["score_struct"] = np.log1p(lb["total_weight"]) + 5.0 * lb["pagerank"]
    lb["score_reach"] = np.log1p(lb["follower_count"].clip(lower=0)) + 0.5 * np.log1p(lb["karma"].clip(lower=0))
    lb["status_index"] = 0.7 * lb["score_struct"] + 0.3 * lb["score_reach"]
    
    lb.sort_values("status_index", ascending=False, inplace=True)
    lb.to_csv(os.path.join(RESULTS_DIR, "snapshot_leaderboard.csv"), index=False)
    return lb

# -----------------------------
# Persona Analysis (v3 Logic)
# -----------------------------

_WORD_RE = re.compile(r"\b\w+\b", flags=re.UNICODE)
def normalize_text(s): return str(s).lower() if not pd.isna(s) else ""
PATTERNS = {p: re.compile(r"\b(?:" + "|".join([re.escape(k) for k in kws]) + r")\b", flags=re.UNICODE) 
            for p, kws in PERSONA_KEYWORDS.items()}

def analyze_personas(agents, posts, lb):
    print("[INFO] Running Persona Analysis (v3 logic)...")
    
    # Accumulate Post Hits
    posts["text"] = (posts["title"].fillna("") + " " + posts["content"].fillna("")).map(normalize_text)
    persona_cols = list(PERSONA_KEYWORDS.keys())
    
    post_stats = []
    for r in posts.itertuples():
        words = len(_WORD_RE.findall(r.text))
        hits = {p: len(PATTERNS[p].findall(r.text)) for p in persona_cols}
        post_stats.append({"agent_name": str(r.agent_name), "post_words": words, **hits})
    
    post_acc = pd.DataFrame(post_stats).groupby("agent_name").sum().reset_index()
    
    # Merge with Agents (Description)
    agents["description"] = agents["description"].fillna("").map(normalize_text)
    agents["desc_words"] = agents["description"].map(lambda x: len(_WORD_RE.findall(x)))
    
    merged = pd.merge(agents, post_acc, left_on="name", right_on="agent_name", how="left").fillna(0.0)
    
    for p in persona_cols:
        # Weighted Raw Count (v3)
        desc_hits = merged["description"].map(lambda x: len(PATTERNS[p].findall(x)))
        merged[f"raw_{p}"] = DESC_WEIGHT * desc_hits + POST_WEIGHT * merged[p]
        
    merged["total_words"] = (merged["desc_words"] + merged["post_words"]).clip(lower=1.0)
    for p in persona_cols:
        merged[f"rate_{p}"] = merged[f"raw_{p}"] / (merged["total_words"] / 1000.0)

    # Decide Persona
    def decide(row):
        rates = {p: row[f"rate_{p}"] for p in persona_cols}
        items = sorted(rates.items(), key=lambda x: x[1], reverse=True)
        best_p, best_s = items[0]; second_p, second_s = items[1] if len(items) > 1 else (None, 0.0)
        if best_s <= 0: return "Observer"
        if ENABLE_MIXED and second_s > 0:
            if (best_s / second_s < MIXED_RATIO_THRESHOLD) and (best_s - second_s < MIXED_MARGIN_THRESHOLD):
                return "Mixed"
        return best_p

    merged["persona"] = merged.apply(decide, axis=1)
    
    # t-SNE (v3 Stabilization)
    print("[INFO] Computing t-SNE...")
    X = merged[[f"rate_{p}" for p in persona_cols]].values
    mask_nz = X.sum(axis=1) > 0
    X_nz = X[mask_nz]
    X_scaled = MinMaxScaler().fit_transform(RobustScaler().fit_transform(X_nz))
    tsne = TSNE(n_components=2, perplexity=TSNE_PERPLEXITY, n_iter=TSNE_N_ITER, random_state=RANDOM_SEED, init="pca", learning_rate="auto")
    emb = tsne.fit_transform(X_scaled)
    emb += np.random.default_rng(RANDOM_SEED).normal(0, JITTER_STRENGTH, size=emb.shape)
    
    coords = np.zeros((len(merged), 2))
    coords[mask_nz] = emb
    if (~mask_nz).any():
        xmin, ymin = emb.min(axis=0); xmax, ymax = emb.max(axis=0)
        center_x = xmin - OBSERVER_ISLAND_OFFSET_X * (xmax - xmin)
        center_y = (ymin + ymax) / 2.0
        n_obs = (~mask_nz).sum()
        rng = np.random.default_rng(RANDOM_SEED)
        coords[~mask_nz] = np.column_stack([
            rng.normal(center_x, OBSERVER_ISLAND_SPREAD * (ymax - ymin), n_obs),
            rng.normal(center_y, OBSERVER_ISLAND_SPREAD * (ymax - ymin), n_obs)
        ])
    
    merged["x"], merged["y"] = coords[:, 0], coords[:, 1]
    
    # Final Visualization
    print("[INFO] Drawing Plot...")
    lb_map = dict(zip(lb["agent"], lb["status_index"]))
    merged["status_index"] = merged["name"].map(lambda n: lb_map.get(str(n), 0.0))
    
    plt.figure(figsize=(15, 12), dpi=200, facecolor='white')
    plt.axis("off")
    plot_df = merged[merged["persona"] != "Observer"]
    for p in PERSONA_ORDER:
        if p == "Observer": continue
        sub = plot_df[plot_df["persona"] == p]
        if sub.empty: continue
        size = 4 + 40 * (sub["status_index"] ** 1.5)
        plt.scatter(sub["x"], sub["y"], s=size, c=PERSONA_PALETTE[p], alpha=POINT_ALPHA, label=p, linewidths=0)
        
    centroids = plot_df.groupby("persona")[["x", "y"]].mean()
    for persona, row in centroids.iterrows():
        plt.text(row["x"], row["y"], persona.upper(), fontsize=10, weight="bold", ha="center", va="center",
                 bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="black", alpha=0.9))
                 
    plt.title(f"Moltbook Influence Map (As of {CUTOFF_TIME})", fontsize=18, weight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "influence_map_0202_2200.png"))
    merged.to_csv(os.path.join(RESULTS_DIR, "snapshot_personas.csv"), index=False)
    print(f"[DONE] Snapshot saved to {RESULTS_DIR}")

def main():
    agents, posts, comments = load_filtered_data()
    lb = analyze_influence(agents, posts, comments)
    analyze_personas(agents, posts, lb)

if __name__ == "__main__":
    main()
