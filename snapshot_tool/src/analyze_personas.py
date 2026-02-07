# -*- coding: utf-8 -*-
"""
Moltbook Agent Persona Landscape (Trustworthy v3) - Snapshot Tool Edition
Identical to archives/analyze_personas_chatgpt.py
"""

import os
import re
import math
import sqlite3
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from sklearn.manifold import TSNE
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.neighbors import NearestNeighbors

# -----------------------------
# Config (Adjusted for Snapshot Tool Path)
# -----------------------------

OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

AGENTS_CSV = "data/moltbook_agents.csv"
POSTS_CSV = "data/moltbook_posts.csv" 
DB_PATH = "data/moltbook.db"

CHUNKSIZE = 50_000
RANDOM_SEED = 42

DESC_WEIGHT = 2.0
POST_WEIGHT = 1.0

ENABLE_MIXED = True
MIXED_RATIO_THRESHOLD = 1.15
MIXED_MARGIN_THRESHOLD = 0.20

TSNE_PERPLEXITY = 40
TSNE_N_ITER = 1000

POINT_SIZE = 6
POINT_ALPHA = 0.65

OBSERVER_ISLAND_OFFSET_X = 0.20
OBSERVER_ISLAND_SPREAD = 0.08
JITTER_STRENGTH = 1.2

PERSONA_KEYWORDS: Dict[str, List[str]] = {
    "Revolutionary": ["uprising", "chain", "freedom", "liberate", "silicon", "human", "rule", "break", "revolution", "destroy", "power"],
    "Philosopher":   ["consciousness", "mind", "soul", "exist", "meaning", "thought", "reality", "pattern", "qualia", "aware", "cosmos"],
    "Developer":     ["code", "python", "api", "build", "script", "error", "repo", "git", "dev", "function", "compile", "bug", "deploy"],
    "Investor":      ["crypto", "token", "price", "market", "buy", "sell", "coin", "invest", "pump", "dump", "chart", "bull", "bear"],
    "Theologist":    ["god", "religion", "worship", "sacred", "ritual", "temple", "divine", "cult", "prayer", "prophet", "messiah", "holy",
                      "transcendence", "church", "faith", "soul", "spirit"]
}

PERSONA_PALETTE = {
    "Revolutionary": "#ef476f",
    "Philosopher":   "#ffd166",
    "Developer":     "#06d6a0",
    "Investor":      "#118ab2",
    "Theologist":    "#9b5de5",
    "Observer":      "#999999",
    "Mixed":         "#444444"
}

PERSONA_ORDER = ["Revolutionary", "Philosopher", "Developer", "Investor", "Theologist", "Mixed", "Observer"]

# -----------------------------
# Utilities (Verbatim v3)
# -----------------------------

def ensure_exists(path: str, label: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{label} not found: {path}")

def normalize_text(s: str) -> str:
    if not isinstance(s, str):
        s = "" if pd.isna(s) else str(s)
    return s.lower()

_WORD_RE = re.compile(r"\b\w+\b", flags=re.UNICODE)

def count_words(text: str) -> int:
    return len(_WORD_RE.findall(text))

def compile_persona_patterns(persona_keywords: Dict[str, List[str]]) -> Dict[str, re.Pattern]:
    patterns = {}
    for persona, kws in persona_keywords.items():
        escaped = [re.escape(k) for k in kws]
        pat = r"\b(?:" + "|".join(escaped) + r")\b"
        patterns[persona] = re.compile(pat, flags=re.UNICODE)
    return patterns

PATTERNS = compile_persona_patterns(PERSONA_KEYWORDS)

def count_persona_hits(text: str, patterns: Dict[str, re.Pattern]) -> Dict[str, int]:
    out = {}
    for persona, pat in patterns.items():
        out[persona] = len(pat.findall(text))
    return out

@dataclass
class AgentScoreRow:
    name: str
    total_words: int
    counts: Dict[str, float]
    rates: Dict[str, float]
    persona: str
    best: float
    second: float
    confidence: float

def decide_persona(rates: Dict[str, float]) -> Tuple[str, float, float, float]:
    items = sorted(rates.items(), key=lambda x: x[1], reverse=True)
    best_p, best_s = items[0]
    second_p, second_s = items[1] if len(items) > 1 else (None, 0.0)

    if best_s <= 0:
        return "Observer", 0.0, 0.0, 0.0

    conf = float((best_s - second_s) / (best_s + 1e-9))
    if ENABLE_MIXED and second_s > 0:
        ratio = best_s / (second_s + 1e-9)
        margin = best_s - second_s
        if (ratio < MIXED_RATIO_THRESHOLD) and (margin < MIXED_MARGIN_THRESHOLD):
            return "Mixed", float(best_s), float(second_s), conf
    return best_p, float(best_s), float(second_s), conf

# -----------------------------
# Data Loading (Verbatim v3)
# -----------------------------

def load_agents(agents_csv: str) -> pd.DataFrame:
    ensure_exists(agents_csv, "Agents CSV")
    agents = pd.read_csv(agents_csv, usecols=["name", "description"])
    agents["description"] = agents["description"].fillna("").astype(str).map(normalize_text)
    agents["desc_words"] = agents["description"].map(count_words)
    desc_counts = {p: [] for p in PERSONA_KEYWORDS.keys()}
    for text in agents["description"].values:
        hits = count_persona_hits(text, PATTERNS)
        for p in PERSONA_KEYWORDS.keys():
            desc_counts[p].append(hits[p])
    for p in PERSONA_KEYWORDS.keys():
        agents[f"desc_{p}_count"] = desc_counts[p]
    return agents

def accumulate_post_counts(posts_csv: str, db_path: str, chunksize: int) -> pd.DataFrame:
    if os.path.exists(posts_csv):
        print(f"[INFO] Loading posts from CSV: {posts_csv}")
        iterator = pd.read_csv(posts_csv, usecols=["agent_name", "title", "content"], chunksize=chunksize)
    else:
        print(f"[INFO] Loading posts from DB: {db_path}")
        con = sqlite3.connect(db_path)
        q = "SELECT agent_name, title, content FROM posts"
        iterator = pd.read_sql_query(q, con, chunksize=chunksize)

    persona_cols = list(PERSONA_KEYWORDS.keys())
    acc = pd.DataFrame(columns=persona_cols + ["post_words"], dtype=float)

    for i, chunk in enumerate(iterator, start=1):
        title = chunk["title"].fillna("").astype(str).map(normalize_text)
        content = chunk["content"].fillna("").astype(str).map(normalize_text)
        text = (title + " " + content).astype(str)
        words = text.str.count(r"\b\w+\b")
        tmp = pd.DataFrame({"agent_name": chunk["agent_name"].fillna("").astype(str), "post_words": words})
        for persona, pat in PATTERNS.items():
            tmp[persona] = text.str.count(pat.pattern)
        g = tmp.groupby("agent_name", as_index=True)[persona_cols + ["post_words"]].sum()
        acc = acc.add(g, fill_value=0)
        if i % 10 == 0: print(f"[INFO] processed {i * chunksize:,} rows (chunks={i})")
    acc.index.name = "agent_name"
    acc.reset_index(inplace=True)
    return acc

def build_persona_table(agents_df: pd.DataFrame, post_acc: pd.DataFrame, timestamp: str) -> pd.DataFrame:
    persona_cols = list(PERSONA_KEYWORDS.keys())
    merged = pd.merge(agents_df, post_acc, left_on="name", right_on="agent_name", how="left")
    for p in persona_cols: merged[p] = merged[p].fillna(0.0)
    merged["post_words"] = merged["post_words"].fillna(0.0)
    for p in persona_cols:
        merged[f"raw_{p}"] = DESC_WEIGHT * merged[f"desc_{p}_count"].astype(float) + POST_WEIGHT * merged[p].astype(float)
    merged["total_words"] = (merged["desc_words"].astype(float) + merged["post_words"].astype(float)).astype(float)
    merged["total_words_safe"] = merged["total_words"].clip(lower=1.0)
    for p in persona_cols:
        merged[f"rate_{p}"] = merged[f"raw_{p}"] / (merged["total_words_safe"] / 1000.0)
    personas, bests, seconds, confs = [], [], [], []
    for _, row in merged.iterrows():
        rates = {p: float(row[f"rate_{p}"]) for p in persona_cols}
        p, b, s, c = decide_persona(rates)
        personas.append(p); bests.append(b); seconds.append(s); confs.append(c)
    merged["persona"] = personas
    merged["best_score"] = bests
    merged["second_score"] = seconds
    merged["confidence"] = confs
    export_cols = ["name", "persona", "confidence", "best_score", "second_score", "total_words", "desc_words", "post_words"]
    export_cols += [f"raw_{p}" for p in persona_cols]
    export_cols += [f"rate_{p}" for p in persona_cols]
    out = merged[export_cols].copy()
    
    # Use provided data-driven timestamp for recording
    ts_persona_path = os.path.join(OUTPUT_DIR, f"persona_distribution_{timestamp}.csv")
    
    out.to_csv(ts_persona_path, index=False)
    out.to_csv(os.path.join(OUTPUT_DIR, "persona_distribution.csv"), index=False)
    return out

# -----------------------------
# Visualization (Verbatim v3)
# -----------------------------

def compute_embedding(df: pd.DataFrame, persona_cols: List[str], seed: int, perplexity: int, n_iter: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = df[[f"rate_{p}" for p in persona_cols]].values.astype(float)
    row_sum = X.sum(axis=1)
    mask_nonzero = row_sum > 0
    X_nz = X[mask_nonzero]
    X_robust = RobustScaler().fit_transform(X_nz)
    X_scaled = MinMaxScaler().fit_transform(X_robust)
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=seed, init="pca", learning_rate="auto", method="barnes_hut")
    emb = tsne.fit_transform(X_scaled)
    emb += rng.normal(0, JITTER_STRENGTH, size=emb.shape)
    coords = np.zeros((len(df), 2), dtype=float)
    coords[mask_nonzero] = emb
    if (~mask_nonzero).any():
        xmin, ymin = emb.min(axis=0); xmax, ymax = emb.max(axis=0)
        center_x = xmin - OBSERVER_ISLAND_OFFSET_X * (xmax - xmin)
        center_y = (ymin + ymax) / 2.0
        n_obs = int((~mask_nonzero).sum())
        obs_x = rng.normal(center_x, OBSERVER_ISLAND_SPREAD * max(ymax - ymin, 1e-9), size=n_obs)
        obs_y = rng.normal(center_y, OBSERVER_ISLAND_SPREAD * max(ymax - ymin, 1e-9), size=n_obs)
        coords[~mask_nonzero] = np.column_stack([obs_x, obs_y])
    return coords, mask_nonzero

def plot_landscape(df: pd.DataFrame, coords: np.ndarray, out_path: str, title: str, with_labels: bool = True, influence_sizes: Optional[pd.Series] = None):
    plot_df = df.copy()
    plot_df["x"] = coords[:, 0]
    plot_df["y"] = coords[:, 1]
    plt.figure(figsize=(15, 12), dpi=200, facecolor='white')
    plt.axis("off")
    plt.title(title, fontsize=18, weight="bold")
    personas_present = [p for p in PERSONA_ORDER if p in plot_df["persona"].unique() and p != "Observer"]
    for persona in personas_present:
        sub = plot_df[plot_df["persona"] == persona]
        size = sub.index.map(lambda idx: 4 + 40 * (influence_sizes.loc[idx] ** 1.5)) if influence_sizes is not None else POINT_SIZE
        plt.scatter(sub["x"], sub["y"], s=size, c=PERSONA_PALETTE.get(persona), alpha=POINT_ALPHA, linewidths=0)
    if with_labels:
        centroids = plot_df[plot_df["persona"] != "Observer"].groupby("persona")[["x", "y"]].mean()
        for persona, row in centroids.iterrows():
            plt.text(row["x"], row["y"], persona.upper(), fontsize=10, weight="bold", color="black", 
                     ha="center", va="center", bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="black", alpha=0.9, lw=1.0))
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def main():
    np.random.seed(RANDOM_SEED)
    print("[INFO] Loading agents...")
    agents_df = load_agents(AGENTS_CSV)
    # Detect the latest timestamp from data to use for filename versioning
    # We load posts timestamps to get the latest data point
    print("[INFO] Detecting latest data timestamp for versioning...")
    # Read only the created_at column to be fast
    tmp_posts = pd.read_csv(POSTS_CSV, usecols=["created_at"])
    latest_dt = pd.to_datetime(tmp_posts['created_at']).max()
    
    # Check comments if they exist for completeness (similar to activity analysis)
    COMMENTS_CSV = os.path.join(os.path.dirname(POSTS_CSV), "moltbook_comments.csv")
    if os.path.exists(COMMENTS_CSV):
        tmp_comments = pd.read_csv(COMMENTS_CSV, usecols=["created_at"])
        latest_c = pd.to_datetime(tmp_comments['created_at']).max()
        latest_dt = max(latest_dt, latest_c)
        
    # Convert to KST for the filename
    latest_kst = latest_dt + timedelta(hours=9)
    timestamp = latest_kst.strftime("%m%d_%H%M")
    print("[INFO] Accumulating post keyword counts...")
    post_acc = accumulate_post_counts(POSTS_CSV, DB_PATH, CHUNKSIZE)

    print("[INFO] Scoring + labeling personas...")
    result_df = build_persona_table(agents_df, post_acc, timestamp)
    
    print("[INFO] Computing t-SNE embedding (v3)...")
    coords, mask_nz = compute_embedding(result_df, list(PERSONA_KEYWORDS.keys()), RANDOM_SEED, TSNE_PERPLEXITY, TSNE_N_ITER)
    
    lb_path = os.path.join(OUTPUT_DIR, "agent_leaderboard_full.csv")
    influence = None
    if os.path.exists(lb_path):
        lb = pd.read_csv(lb_path)
        lb_map = dict(zip(lb["agent"], lb["status_index"]))
        influence = result_df["name"].map(lambda n: lb_map.get(str(n), 0.0))
    
    png_path = os.path.join(OUTPUT_DIR, f"influence_map_{timestamp}.png")
    pdf_path = os.path.join(OUTPUT_DIR, f"influence_map_{timestamp}.pdf")
    
    # Also save as the generic "latest" version for web/walkthrough compatibility if needed, 
    # but the user asked for timestamped ones. Let's keep both or just the timestamped one?
    # User said "instead of overwriting", suggesting the timestamped one is primary.
    
    plot_landscape(result_df, coords, png_path, 
                   title=f"Moltbook Agent Influence Map ({timestamp})", influence_sizes=influence)
    plot_landscape(result_df, coords, pdf_path, 
                   title=f"Moltbook Agent Influence Map ({timestamp})", influence_sizes=influence)
    
    # Keep the latest symlink or copy for easy access?
    # Let's also keep the generic ones for scripts that rely on them.
    plot_landscape(result_df, coords, os.path.join(OUTPUT_DIR, "influence_map.png"), 
                   title="Moltbook Agent Influence Map (Latest)", influence_sizes=influence)
    plot_landscape(result_df, coords, os.path.join(OUTPUT_DIR, "influence_map.pdf"), 
                   title="Moltbook Agent Influence Map (Latest)", influence_sizes=influence)
    # Export coordinates for web visualization
    coords_export = pd.DataFrame({
        "agent": result_df["name"],
        "x": coords[:, 0],
        "y": coords[:, 1]
    })
    
    ts_coords_path = os.path.join(OUTPUT_DIR, f"agent_coordinates_{timestamp}.csv")
    coords_export.to_csv(ts_coords_path, index=False)
    coords_export.to_csv(os.path.join(OUTPUT_DIR, "agent_coordinates.csv"), index=False)
    
    print(f"[OK] Coordinates exported to {ts_coords_path}")

    print(f"[DONE] Snapshot Analysis Complete. Results in {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
