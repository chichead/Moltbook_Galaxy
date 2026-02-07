# -*- coding: utf-8 -*-
"""
Moltbook Galaxy Snapshot Generator (Feb 7th 00:00 KST)
Uses Web Coordinates (v3) as a master anchor for visual continuity.
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
from datetime import datetime
from sklearn.manifold import TSNE
from sklearn.preprocessing import RobustScaler, MinMaxScaler

# -----------------------------
# Config
# -----------------------------
CUTOFF_TIME = "2026-02-07 00:00:00"
DATA_DIR = "data" # Relative to snapshot_tool or absolute? Using relative assuming run from snapshot_tool dir or root.. let's be safe.
# Adjusting assuming script is in snapshot_tool/
# But wait, run_pipeline runs from root.
# Let's use absolute relative to project root or assume running from project root.

# If running from project root:
DB_PATH = "snapshot_tool/data/moltbook.db"
RESULTS_DIR = "snapshot_tool/results/snapshot_0207_0000"
WEB_COORD_PATH = "analysis_results/agent_coordinates_v3.csv" 

# Analysis Constants (v3)
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
    "Investor": "#118ab2", "Theologist": "#9b5de5", "Observer": "#999999", "Mixed": "#4b5563"
}
PERSONA_ORDER = ["Revolutionary", "Philosopher", "Developer", "Investor", "Theologist", "Mixed", "Observer"]

os.makedirs(RESULTS_DIR, exist_ok=True)

# -----------------------------
# Utils
# -----------------------------
_WORD_RE = re.compile(r"\b\w+\b", flags=re.UNICODE)
def normalize_text(s): return str(s).lower() if not pd.isna(s) else ""
PATTERNS = {p: re.compile(r"\b(?:" + "|".join([re.escape(k) for k in kws]) + r")\b", flags=re.UNICODE) 
            for p, kws in PERSONA_KEYWORDS.items()}

# -----------------------------
# Core Functions
# -----------------------------

def load_data_at(cutoff):
    conn = sqlite3.connect(DB_PATH)
    agents_query = f"""
        SELECT name, description, karma, follower_count, following_count 
        FROM agents 
        WHERE COALESCE(NULLIF(created_at, ''), first_seen_at) <= '{cutoff}'
    """
    agents = pd.read_sql_query(agents_query, conn)
    posts = pd.read_sql_query(f"SELECT id, agent_name, title, content, score, comment_count, created_at FROM posts WHERE created_at <= '{cutoff}'", conn)
    comments = pd.read_sql_query(f"SELECT post_id, agent_name, created_at FROM comments WHERE created_at <= '{cutoff}'", conn)
    conn.close()
    return agents, posts, comments

def analyze_influence(agents, posts, comments):
    post_author_map = dict(zip(posts["id"].astype(str), posts["agent_name"].fillna("").astype(str)))
    edge_counter = Counter()
    for r in comments.itertuples():
        src, tgt = str(r.agent_name), post_author_map.get(str(r.post_id))
        if tgt and src != tgt: edge_counter[(src, tgt)] += 1
            
    edges = pd.DataFrame([(s, t, w) for (s, t), w in edge_counter.items()], columns=["src", "tgt", "weight"])
    if edges.empty:
        # Fallback for empty edges
        agents["total_weight"] = 0
        agents["pagerank"] = 0.0
        agents["status_index"] = 0.0
        return agents.rename(columns={"name": "agent"})

    out_w, in_w = edges.groupby("src")["weight"].sum(), edges.groupby("tgt")["weight"].sum()
    nodes = pd.DataFrame({"agent": pd.Index(out_w.index).union(in_w.index)})
    nodes["total_weight"] = nodes["agent"].map(lambda a: float(out_w.get(a, 0) + in_w.get(a, 0)))
    
    top_agents = nodes.sort_values("total_weight", ascending=False).head(5000)["agent"].tolist()
    edges_pr = edges[edges["src"].isin(top_agents) & edges["tgt"].isin(top_agents)].copy()
    G = nx.DiGraph()
    for r in edges_pr.itertuples(index=False): G.add_edge(r.src, r.tgt, weight=float(r.weight))
    pr = nx.pagerank(G, weight="weight") if G.number_of_nodes() > 0 else {}
    
    lb = agents.copy().rename(columns={"name": "agent"})
    lb = pd.merge(lb, nodes, on="agent", how="left").fillna(0.0)
    lb["pagerank"] = lb["agent"].map(lambda a: float(pr.get(a, 0.0)))
    
    lb["score_struct"] = np.log1p(lb["total_weight"]) + 5.0 * lb["pagerank"]
    lb["score_reach"] = np.log1p(lb["follower_count"].clip(lower=0)) + 0.5 * np.log1p(lb["karma"].clip(lower=0))
    lb["status_index"] = 0.7 * lb["score_struct"] + 0.3 * lb["score_reach"]
    return lb.sort_values("status_index", ascending=False)

def analyze_personas(agents, posts, lb, prev_coords_map=None):
    posts["text"] = (posts["title"].fillna("") + " " + posts["content"].fillna("")).map(normalize_text)
    persona_cols = list(PERSONA_KEYWORDS.keys())
    
    post_stats = []
    for r in posts.itertuples():
        words = len(_WORD_RE.findall(r.text))
        hits = {p: len(PATTERNS[p].findall(r.text)) for p in persona_cols}
        post_stats.append({"agent_name": str(r.agent_name), "post_words": words, **hits})
    
    post_acc = pd.DataFrame(post_stats)
    if not post_acc.empty:
        post_acc = post_acc.groupby("agent_name").sum().reset_index()
    else:
        post_acc = pd.DataFrame(columns=["agent_name", "post_words"] + persona_cols)
    
    agents["description"] = agents["description"].fillna("").map(normalize_text)
    agents["desc_words"] = agents["description"].map(lambda x: len(_WORD_RE.findall(x)))
    merged = pd.merge(agents, post_acc, left_on="name", right_on="agent_name", how="left").fillna(0.0)
    
    for p in persona_cols:
        desc_hits = merged["description"].map(lambda x: len(PATTERNS[p].findall(x)))
        merged[f"raw_{p}"] = DESC_WEIGHT * desc_hits + POST_WEIGHT * merged[p]
        
    merged["total_words"] = (merged["desc_words"] + merged["post_words"]).clip(lower=1.0)
    for p in persona_cols: merged[f"rate_{p}"] = merged[f"raw_{p}"] / (merged["total_words"] / 1000.0)

    def decide(row):
        rates = {p: row[f"rate_{p}"] for p in persona_cols}
        items = sorted(rates.items(), key=lambda x: x[1], reverse=True)
        best_p, best_s = items[0]; second_p, second_s = items[1] if len(items) > 1 else (None, 0.0)
        
        conf = float((best_s - second_s) / (best_s + 1e-9))
        
        if best_s <= 0: return "Observer", 0.0, 0.0, 0.0
        
        persona = best_p
        if ENABLE_MIXED and second_s > 0:
            if (best_s / second_s < MIXED_RATIO_THRESHOLD) and (best_s - second_s < MIXED_MARGIN_THRESHOLD):
                persona = "Mixed"
        
        return persona, float(best_s), float(second_s), conf

    # Apply decide and expand result to columns
    decision_data = merged.apply(decide, axis=1, result_type='expand')
    merged["persona"] = decision_data[0]
    merged["best_score"] = decision_data[1]
    merged["second_score"] = decision_data[2]
    merged["confidence"] = decision_data[3]
    
    # t-SNE Warm Start
    coords = np.zeros((len(merged), 2))
    X = merged[[f"rate_{p}" for p in persona_cols]].values
    mask_has_features = X.sum(axis=1) > 0
    
    if mask_has_features.any():
        X_nz = X[mask_has_features]
        X_scaled = MinMaxScaler().fit_transform(RobustScaler().fit_transform(X_nz))
        
        init_mode = "pca"
        if prev_coords_map:
            agents_has_feat = merged[mask_has_features]["name"].tolist()
            init_coords = np.zeros((len(agents_has_feat), 2))
            
            all_coords = list(prev_coords_map.values())
            global_mean = np.mean(all_coords, axis=0) if all_coords else np.zeros(2)
            
            hit_count = 0
            for i, name in enumerate(agents_has_feat):
                if name in prev_coords_map:
                    init_coords[i] = prev_coords_map[name]
                    hit_count += 1
                else:
                    init_coords[i] = global_mean
            
            print(f"  - Anchored {hit_count}/{len(agents_has_feat)} agents using previous coordinates.")
            init_mode = init_coords

        n_samples = X_nz.shape[0]
        safe_perplexity = min(TSNE_PERPLEXITY, max(1, n_samples - 1))
        
        tsne = TSNE(n_components=2, perplexity=safe_perplexity, n_iter=TSNE_N_ITER, 
                    random_state=RANDOM_SEED, init=init_mode, learning_rate="auto")
        emb = tsne.fit_transform(X_scaled)
        
        # Apply Jitter
        rng = np.random.default_rng(RANDOM_SEED)
        emb += rng.normal(0, JITTER_STRENGTH, size=emb.shape)
        
        coords[mask_has_features] = emb
        
        # Observer Island
        if (~mask_has_features).any():
            xmin, ymin = emb.min(axis=0); xmax, ymax = emb.max(axis=0)
            center_x, center_y = xmin - OBSERVER_ISLAND_OFFSET_X * (xmax - xmin), (ymin + ymax) / 2.0
            n_obs = (~mask_has_features).sum()
            rng = np.random.default_rng(RANDOM_SEED)
            coords[~mask_has_features] = np.column_stack([
                rng.normal(center_x, OBSERVER_ISLAND_SPREAD * (ymax - ymin), n_obs),
                rng.normal(center_y, OBSERVER_ISLAND_SPREAD * (ymax - ymin), n_obs)
            ])
    else:
        rng = np.random.default_rng(RANDOM_SEED)
        coords = rng.normal(0, 0.1, size=coords.shape)
    
    merged["x"], merged["y"] = coords[:, 0], coords[:, 1]
    
    # Merge LB status index back to Merged DF for plotting
    lb_map = dict(zip(lb["agent"], lb["status_index"]))
    merged["status_index"] = merged["name"].map(lambda n: lb_map.get(str(n), 0.0))
    
    return merged

def plot_snapshot(df, timestamp, output_path):
    plt.figure(figsize=(15, 12), dpi=150, facecolor='#0b0e14')
    plt.gca().set_facecolor('#0b0e14')
    plt.axis("off")
    
    plot_df = df[df["persona"] != "Observer"]
    for p in PERSONA_ORDER:
        if p == "Observer": continue
        sub = plot_df[plot_df["persona"] == p]
        if sub.empty: continue
        size = 4 + 40 * (sub["status_index"] ** 1.5)
        plt.scatter(sub["x"], sub["y"], s=size, c=PERSONA_PALETTE[p], alpha=POINT_ALPHA, label=p, linewidths=0)
        
    plt.title(f"Moltbook Galaxy Snapshot - {timestamp}", fontsize=20, weight="bold", color="white", pad=20)
    plt.gca().invert_yaxis() 
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def main():
    print(f"[INFO] Generating snapshot for {CUTOFF_TIME}...")
    
    # Load Master Anchor (Web Coordinates)
    if os.path.exists(WEB_COORD_PATH):
        print(f"  - Using master anchor: {WEB_COORD_PATH}")
        anchor_df = pd.read_csv(WEB_COORD_PATH)
        prev_coords_map = dict(zip(anchor_df["agent"], zip(anchor_df["x"], anchor_df["y"])))
    else:
        print(f"  - [WARNING] Master anchor {WEB_COORD_PATH} not found. Initializing from scratch.")
        prev_coords_map = None

    agents, posts, comments = load_data_at(CUTOFF_TIME)
    print(f"  - Loaded {len(agents)} agents, {len(posts)} posts, {len(comments)} comments.")

    if len(agents) == 0:
        print("  - [ERROR] No agents found at this cutoff.")
        return

    lb = analyze_influence(agents, posts, comments)
    merged = analyze_personas(agents, posts, lb, prev_coords_map)
    
    # Save CSVs
    merged.to_csv(os.path.join(RESULTS_DIR, "snapshot_personas.csv"), index=False)
    lb.to_csv(os.path.join(RESULTS_DIR, "snapshot_leaderboard.csv"), index=False)
    
    # Also save separate coordinates file for publish_to_web compatibility
    coords_export = merged[["name", "x", "y"]].rename(columns={"name": "agent"})
    coords_path = os.path.join(RESULTS_DIR, "agent_coordinates.csv") # Used by publish_to_web
    coords_export.to_csv(coords_path, index=False)
    # Also save to snapshot_tool/results/ root so publish_to_web finds it directly
    root_results = "snapshot_tool/results"
    os.makedirs(root_results, exist_ok=True)
    coords_export.to_csv(os.path.join(root_results, "agent_coordinates.csv"), index=False)
    lb.to_csv(os.path.join(root_results, "agent_leaderboard_full.csv"), index=False)
    merged.to_csv(os.path.join(root_results, "persona_distribution.csv"), index=False) # publish_to_web expects this
    
    # Save Plot
    plot_path = os.path.join(RESULTS_DIR, "galaxy_0207_0000.png")
    plot_snapshot(merged, CUTOFF_TIME, plot_path)
    
    print(f"[DONE] Snapshot generated in {RESULTS_DIR}")
    print(f"  - Visualized stars: {len(merged[merged['persona']!='Observer'])}")
    print(f"  - Copied to snapshot_tool/results/ for publishing.")

if __name__ == "__main__":
    main()
