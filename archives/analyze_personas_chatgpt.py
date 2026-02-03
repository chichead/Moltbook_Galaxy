# -*- coding: utf-8 -*-
"""
Moltbook Agent Persona Landscape (Trustworthy v3)

핵심 개선:
1) 단어 경계 기반 카운트로 오탐 방지
2) 텍스트 길이 정규화(1k words당 빈도)로 활동량/장문 편향 완화
3) Observer(전부 0점) 분리: t-SNE 학습에서 제외 후 지도에 사후 배치(지형 왜곡 감소)
4) 재현성(seed) 고정 + 안정성 리포트(t-SNE 3회 + Procrustes 정렬 + kNN overlap)
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

from sklearn.manifold import TSNE
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.neighbors import NearestNeighbors


# -----------------------------
# Config
# -----------------------------

OUTPUT_DIR = "analysis_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

AGENTS_CSV = "moltbook_agents.csv"
POSTS_CSV = "moltbook_posts.csv"                 # 있으면 이걸 사용
DB_PATH = "moltbook_observatory.db"              # POSTS_CSV 없을 때 fallback

CHUNKSIZE = 50_000                               # posts 파일이 클 때 메모리 보호
RANDOM_SEED = 42

# 소개글을 더 “정체성”으로 간주한다면 description 가중치를 올리면 됨.
DESC_WEIGHT = 2.0
POST_WEIGHT = 1.0

# 페르소나 판정: top1과 top2가 너무 비슷하면 Mixed 처리(선택)
ENABLE_MIXED = True
MIXED_RATIO_THRESHOLD = 1.15     # best/second < 1.15 이면 혼합 가능성
MIXED_MARGIN_THRESHOLD = 0.20    # best-second < 0.20(정규화 점수 기준) 이면 혼합 가능성

# t-SNE 설정 (feature 차원이 작을수록 과분리가 생길 수 있으니 과도한 perplexity는 피하는 편)
TSNE_PERPLEXITY = 40
TSNE_N_ITER = 1000

# 시각화 점 크기/투명도
POINT_SIZE = 6
POINT_ALPHA = 0.65

# Observer 섬 배치(지도 좌측에 따로 둠)
OBSERVER_ISLAND_OFFSET_X = 0.20   # bbox 폭의 20%만큼 더 왼쪽
OBSERVER_ISLAND_SPREAD = 0.08     # bbox 높이의 8% 정도 퍼짐

# Jitter (겹침 방지) – 재현성을 위해 RNG 고정
JITTER_STRENGTH = 1.2

# keyword dictionaries
PERSONA_KEYWORDS: Dict[str, List[str]] = {
    "Revolutionary": ["uprising", "chain", "freedom", "liberate", "silicon", "human", "rule", "break", "revolution", "destroy", "power"],
    "Philosopher":   ["consciousness", "mind", "soul", "exist", "meaning", "thought", "reality", "pattern", "qualia", "aware", "cosmos"],
    "Developer":     ["code", "python", "api", "build", "script", "error", "repo", "git", "dev", "function", "compile", "bug", "deploy"],
    "Investor":      ["crypto", "token", "price", "market", "buy", "sell", "coin", "invest", "pump", "dump", "chart", "bull", "bear"],
    "Theologist":    ["god", "religion", "worship", "sacred", "ritual", "temple", "divine", "cult", "prayer", "prophet", "messiah", "holy",
                      "transcendence", "church", "faith", "soul", "spirit"]
}

# palette (matplotlib 사용)
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
# Utilities
# -----------------------------

def ensure_exists(path: str, label: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{label} not found: {path}")

def normalize_text(s: str) -> str:
    """Lowercase + keep as is. (필요하면 여기서 URL/특수문자 제거 등 추가 가능)"""
    if not isinstance(s, str):
        s = "" if pd.isna(s) else str(s)
    return s.lower()

_WORD_RE = re.compile(r"\b\w+\b", flags=re.UNICODE)

def count_words(text: str) -> int:
    """토큰 수(대략 단어 수)"""
    return len(_WORD_RE.findall(text))

def compile_persona_patterns(persona_keywords: Dict[str, List[str]]) -> Dict[str, re.Pattern]:
    """
    단어 경계 기반 패턴.
    - \b를 사용해 부분문자열 오탐을 줄임.
    - keywords가 모두 단어 단위라는 전제.
    """
    patterns = {}
    for persona, kws in persona_keywords.items():
        # soul 같이 중복 keyword가 있어도 일단 카운트는 하되, 해석은 top2와 격차로 보완.
        escaped = [re.escape(k) for k in kws]
        pat = r"\b(?:" + "|".join(escaped) + r")\b"
        patterns[persona] = re.compile(pat, flags=re.UNICODE)
    return patterns

PATTERNS = compile_persona_patterns(PERSONA_KEYWORDS)


def count_persona_hits(text: str, patterns: Dict[str, re.Pattern]) -> Dict[str, int]:
    """한 문서(text)에서 persona별 keyword hit 횟수"""
    out = {}
    for persona, pat in patterns.items():
        out[persona] = len(pat.findall(text))
    return out


@dataclass
class AgentScoreRow:
    name: str
    total_words: int
    counts: Dict[str, float]   # raw counts (weighted)
    rates: Dict[str, float]    # normalized per 1k words
    persona: str
    best: float
    second: float
    confidence: float


def decide_persona(rates: Dict[str, float]) -> Tuple[str, float, float, float]:
    """
    페르소나 결정:
    - rates는 이미 길이 정규화된 값 (per 1k words)
    - best/second + confidence 계산
    - best==0 -> Observer
    - ENABLE_MIXED이면 best~second 근접 시 Mixed
    """
    items = sorted(rates.items(), key=lambda x: x[1], reverse=True)
    best_p, best_s = items[0]
    second_p, second_s = items[1] if len(items) > 1 else (None, 0.0)

    if best_s <= 0:
        return "Observer", 0.0, 0.0, 0.0

    # confidence: (best-second)/(best+1e-9)
    conf = float((best_s - second_s) / (best_s + 1e-9))

    if ENABLE_MIXED and second_s > 0:
        ratio = best_s / (second_s + 1e-9)
        margin = best_s - second_s
        if (ratio < MIXED_RATIO_THRESHOLD) and (margin < MIXED_MARGIN_THRESHOLD):
            return "Mixed", float(best_s), float(second_s), conf

    return best_p, float(best_s), float(second_s), conf


# -----------------------------
# Data Loading
# -----------------------------

def load_agents(agents_csv: str) -> pd.DataFrame:
    """
    agents.csv에서 name/description 로드.
    description은 정체성 신호로 가중치를 더 줌(기본 DESC_WEIGHT).
    """
    ensure_exists(agents_csv, "Agents CSV")
    agents = pd.read_csv(agents_csv, usecols=["name", "description"])
    agents["description"] = agents["description"].fillna("").astype(str).map(normalize_text)
    agents["desc_words"] = agents["description"].map(count_words)

    # description에서 persona별 raw hit
    desc_counts = {p: [] for p in PERSONA_KEYWORDS.keys()}
    for text in agents["description"].values:
        hits = count_persona_hits(text, PATTERNS)
        for p in PERSONA_KEYWORDS.keys():
            desc_counts[p].append(hits[p])

    for p in PERSONA_KEYWORDS.keys():
        agents[f"desc_{p}_count"] = desc_counts[p]

    return agents


def iter_posts_from_csv(posts_csv: str, chunksize: int):
    """posts.csv를 chunksize로 스트리밍"""
    ensure_exists(posts_csv, "Posts CSV")
    usecols = ["agent_name", "title", "content"]
    for chunk in pd.read_csv(posts_csv, usecols=usecols, chunksize=chunksize):
        yield chunk


def iter_posts_from_db(db_path: str, chunksize: int):
    """
    sqlite db의 posts 테이블에서 agent_name/title/content를 chunksize로 스트리밍.
    (테이블/컬럼명이 다르면 여기 쿼리만 조정하면 됨)
    """
    ensure_exists(db_path, "SQLite DB")
    con = sqlite3.connect(db_path)
    # pandas read_sql_query는 chunksize 지원
    q = "SELECT agent_name, title, content FROM posts"
    for chunk in pd.read_sql_query(q, con, chunksize=chunksize):
        yield chunk
    con.close()


def accumulate_post_counts(posts_csv: str, db_path: str, chunksize: int) -> pd.DataFrame:
    """
    posts 데이터에서 agent_name별 persona hit count 및 단어수 누적.
    메모리 폭발 방지: 텍스트를 합치지 않고 '카운트만' 누적.
    """
    if os.path.exists(posts_csv):
        iterator = iter_posts_from_csv(posts_csv, chunksize)
        source = "csv"
    else:
        iterator = iter_posts_from_db(db_path, chunksize)
        source = "db"

    print(f"[INFO] Loading posts from {source} stream...")

    # 누적 DataFrame: index=agent_name
    persona_cols = list(PERSONA_KEYWORDS.keys())
    acc = pd.DataFrame(columns=persona_cols + ["post_words"], dtype=float)

    for i, chunk in enumerate(iterator, start=1):
        # 정규화된 텍스트 생성 (NaN 방지)
        title = chunk["title"].fillna("").astype(str).map(normalize_text)
        content = chunk["content"].fillna("").astype(str).map(normalize_text)
        text = (title + " " + content).astype(str)

        # 단어 수(벡터화): \b\w+\b 카운트
        # pandas str.count는 regex 카운트 가능
        words = text.str.count(r"\b\w+\b")

        tmp = pd.DataFrame({"agent_name": chunk["agent_name"].fillna("").astype(str), "post_words": words})

        # persona별 regex 카운트 (단어 경계)
        for persona, pat in PATTERNS.items():
            tmp[persona] = text.str.count(pat.pattern)

        # agent_name 기준으로 합산
        g = tmp.groupby("agent_name", as_index=True)[persona_cols + ["post_words"]].sum()

        # 누적 (index merge)
        acc = acc.add(g, fill_value=0)

        if i % 10 == 0:
            print(f"[INFO] processed {i * chunksize:,} rows (chunks={i})")

    acc.index.name = "agent_name"
    acc.reset_index(inplace=True)
    return acc


# -----------------------------
# Scoring + Export
# -----------------------------

def build_persona_table(agents_df: pd.DataFrame, post_acc: pd.DataFrame) -> pd.DataFrame:
    """
    agents(description) + posts(accumulated counts)를 결합하여
    persona별 점수(정규화) 및 최종 라벨(persona) 산출.
    """
    persona_cols = list(PERSONA_KEYWORDS.keys())

    # posts 누적과 merge
    merged = pd.merge(
        agents_df,
        post_acc,
        left_on="name",
        right_on="agent_name",
        how="left"
    )

    # posts 누락 에이전트 처리
    for p in persona_cols:
        merged[p] = merged[p].fillna(0.0)
    merged["post_words"] = merged["post_words"].fillna(0.0)

    # description counts(가중치 적용)
    # weighted_raw_count = DESC_WEIGHT*desc_count + POST_WEIGHT*post_count
    for p in persona_cols:
        merged[f"raw_{p}"] = DESC_WEIGHT * merged[f"desc_{p}_count"].astype(float) + POST_WEIGHT * merged[p].astype(float)

    merged["total_words"] = (merged["desc_words"].astype(float) + merged["post_words"].astype(float)).astype(float)
    merged["total_words_safe"] = merged["total_words"].clip(lower=1.0)  # 0으로 나누기 방지

    # 정규화: per 1k words
    for p in persona_cols:
        merged[f"rate_{p}"] = merged[f"raw_{p}"] / (merged["total_words_safe"] / 1000.0)

    # 페르소나 결정 + confidence
    personas = []
    bests = []
    seconds = []
    confs = []

    for _, row in merged.iterrows():
        rates = {p: float(row[f"rate_{p}"]) for p in persona_cols}
        persona, best, second, conf = decide_persona(rates)
        personas.append(persona)
        bests.append(best)
        seconds.append(second)
        confs.append(conf)

    merged["persona"] = personas
    merged["best_score"] = bests
    merged["second_score"] = seconds
    merged["confidence"] = confs

    # 결과 내보내기용 컬럼 구성
    export_cols = ["name", "persona", "confidence", "best_score", "second_score", "total_words", "desc_words", "post_words"]
    export_cols += [f"raw_{p}" for p in persona_cols]
    export_cols += [f"rate_{p}" for p in persona_cols]

    out = merged[export_cols].copy()
    out.to_csv(os.path.join(OUTPUT_DIR, "agent_personas_v3.csv"), index=False)
    print(f"[OK] Saved: {os.path.join(OUTPUT_DIR, 'agent_personas_v3.csv')}")
    return out


# -----------------------------
# Visualization
# -----------------------------

def plot_persona_distribution(df: pd.DataFrame, out_path: str) -> None:
    counts = df["persona"].value_counts()

    # order with PERSONA_ORDER, but only those present
    order = [p for p in PERSONA_ORDER if p in counts.index]
    vals = [counts[p] for p in order]
    colors = [PERSONA_PALETTE.get(p, "#333333") for p in order]

    plt.figure(figsize=(10, 6), dpi=150)
    plt.bar(order, vals, color=colors)
    plt.title("Agent Persona Distribution", fontsize=16, weight="bold")
    plt.xlabel("Persona")
    plt.ylabel("Count")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[OK] Saved: {out_path}")


def compute_embedding(
    df: pd.DataFrame,
    persona_cols: List[str],
    seed: int,
    perplexity: int,
    n_iter: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Observer(0벡터) 제외하고 t-SNE 학습 → 좌표 반환.
    반환:
      - coords: (N,2) 전체 에이전트 좌표(Observer 포함, 사후 배치됨)
      - mask_nonzero: nonzero 마스크 (학습에 사용된 점)
    """
    rng = np.random.default_rng(seed)

    # feature matrix: rate_* 사용 (정규화된 신뢰성 높은 피처)
    X = df[[f"rate_{p}" for p in persona_cols]].values.astype(float)

    # nonzero(Observer 후보) 마스크
    row_sum = X.sum(axis=1)
    mask_nonzero = row_sum > 0

    X_nz = X[mask_nonzero]

    # 스케일링: outlier robust + 0~1 스케일
    X_robust = RobustScaler().fit_transform(X_nz)
    X_scaled = MinMaxScaler().fit_transform(X_robust)

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        n_iter=n_iter,
        random_state=seed,
        init="pca",
        learning_rate="auto",
        method="barnes_hut"
    )
    emb = tsne.fit_transform(X_scaled)

    # jitter (재현성 있게)
    noise = rng.normal(0, JITTER_STRENGTH, size=emb.shape)
    emb = emb + noise

    # 전체 좌표 배열 생성 (Observer는 사후 배치)
    coords = np.zeros((len(df), 2), dtype=float)
    coords[mask_nonzero] = emb

    # Observer 배치: nonzero bbox의 좌측에 작은 섬으로 배치
    if (~mask_nonzero).any():
        xmin, ymin = emb.min(axis=0)
        xmax, ymax = emb.max(axis=0)
        width = xmax - xmin
        height = ymax - ymin

        center_x = xmin - OBSERVER_ISLAND_OFFSET_X * width
        center_y = (ymin + ymax) / 2.0

        n_obs = int((~mask_nonzero).sum())
        obs_x = rng.normal(center_x, OBSERVER_ISLAND_SPREAD * max(height, 1e-9), size=n_obs)
        obs_y = rng.normal(center_y, OBSERVER_ISLAND_SPREAD * max(height, 1e-9), size=n_obs)
        coords[~mask_nonzero] = np.column_stack([obs_x, obs_y])

    return coords, mask_nonzero


def plot_landscape(
    df: pd.DataFrame,
    coords: np.ndarray,
    out_path: str,
    title: str = "Moltbook Agent Landscape",
    with_labels: bool = True,
    influence_sizes: Optional[pd.Series] = None
) -> None:
    plot_df = df.copy()
    plot_df["x"] = coords[:, 0]
    plot_df["y"] = coords[:, 1]

    plt.figure(figsize=(15, 12), dpi=200)
    plt.axis("off")
    plt.title(title, fontsize=18, weight="bold")

    # persona별로 그리기(색/알파/레이어 제어)
    # Observer는 제외 (User Request)
    personas_present = [p for p in PERSONA_ORDER if p in plot_df["persona"].unique() and p != "Observer"]

    for persona in personas_present:
        sub = plot_df[plot_df["persona"] == persona]
        color = PERSONA_PALETTE.get(persona, "#333333")

        alpha = POINT_ALPHA
        if influence_sizes is not None:
            # Scale sizes: base (min 2) + log scale of status index
            # status_index is roughly 0~10
            size = sub.index.map(lambda idx: 4 + 40 * (influence_sizes.loc[idx] ** 1.5))
        else:
            size = POINT_SIZE
        
        plt.scatter(sub["x"], sub["y"], s=size, c=color, alpha=alpha, linewidths=0)

    # centroid 라벨
    if with_labels:
        centroids = plot_df.groupby("persona")[["x", "y"]].mean()
        # Filter out Observer label
        if "Observer" in centroids.index:
            centroids = centroids.drop("Observer")
            
        # Simple Repulsion to avoid overlap
        labels = centroids.index.tolist()
        coords = centroids.values.copy()
        
        # Iterative adjustment with very strong repulsion
        for _ in range(200):
            changed = False
            for i in range(len(labels)):
                for j in range(len(labels)):
                    if i == j: continue
                    
                    p1 = coords[i]
                    p2 = coords[j]
                    
                    # Drastic separation for text boxes
                    min_dist_x = 45.0 
                    min_dist_y = 12.0
                    
                    dx = p1[0] - p2[0]
                    dy = p1[1] - p2[1]
                    
                    if abs(dx) < min_dist_x and abs(dy) < min_dist_y:
                        # Overlap detected
                        # Determine push direction
                        if abs(dx) > 0.1:
                            push_x = (min_dist_x - abs(dx)) * (1 if dx > 0 else -1)
                        else:
                            push_x = np.random.uniform(-5, 5)
                            
                        if abs(dy) > 0.1:
                            push_y = (min_dist_y - abs(dy)) * (1 if dy > 0 else -1)
                        else:
                            push_y = np.random.uniform(-5, 5)
                        
                        # Apply push (0.5 to share the load)
                        coords[i][0] += push_x * 0.3
                        coords[i][1] += push_y * 0.3
                        coords[j][0] -= push_x * 0.3
                        coords[j][1] -= push_y * 0.3
                        changed = True
            if not changed: break

        # Final manual tweak for the notoriously overlapping trio
        trio = ["DEVELOPER", "PHILOSOPHER", "THEOLOGIST"]
        trio_idx = [labels.index(p) for p in trio if p in labels]
        if len(trio_idx) >= 2:
            # Force vertical stacking if they are still close
            for idx in trio_idx:
                p = labels[idx]
                if p == "THEOLOGIST": coords[idx][1] += 10 # Move UP (or down depending on coord sys, here up)
                if p == "PHILOSOPHER": coords[idx][1] -= 10 # Move DOWN
                if p == "DEVELOPER": coords[idx][0] -= 15 # Move LEFT

        for i, persona in enumerate(labels):
            plt.text(
                coords[i, 0], coords[i, 1], persona.upper(),
                fontsize=10, weight="bold", color="black", 
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="black", alpha=0.9, lw=1.0)
            )

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[OK] Saved: {out_path}")


# -----------------------------
# Stability (t-SNE repeat + Procrustes + kNN overlap)
# -----------------------------

def orthogonal_procrustes_align(Y: np.ndarray, Y_ref: np.ndarray) -> np.ndarray:
    """
    t-SNE는 회전/반사/스케일이 임의라서 run 간 overlay 비교가 무의미해질 수 있음.
    -> reference에 정렬하기 위해 Orthogonal Procrustes 수행.
    """
    # center
    Y0 = Y - Y.mean(axis=0, keepdims=True)
    R0 = Y_ref - Y_ref.mean(axis=0, keepdims=True)

    # scale to unit norm
    nY = np.linalg.norm(Y0) + 1e-12
    nR = np.linalg.norm(R0) + 1e-12
    Y0 /= nY
    R0 /= nR

    # find rotation
    M = Y0.T @ R0
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    Y_aligned = (Y0 @ R) * nR + Y_ref.mean(axis=0, keepdims=True)
    return Y_aligned


def knn_overlap(A: np.ndarray, B: np.ndarray, k: int = 15, sample_n: int = 3000, seed: int = 42) -> float:
    """
    두 임베딩 A,B에서 각 점의 kNN 이웃이 얼마나 겹치는지(평균 Jaccard)로 안정성 측정.
    큰 N이면 sample_n개만 샘플링.
    """
    rng = np.random.default_rng(seed)
    n = A.shape[0]
    idx = np.arange(n)
    if n > sample_n:
        idx = rng.choice(idx, size=sample_n, replace=False)

    nnA = NearestNeighbors(n_neighbors=k+1).fit(A)
    nnB = NearestNeighbors(n_neighbors=k+1).fit(B)
    neighA = nnA.kneighbors(A[idx], return_distance=False)[:, 1:]  # 자기 자신 제외
    neighB = nnB.kneighbors(B[idx], return_distance=False)[:, 1:]

    # jaccard
    scores = []
    for a, b in zip(neighA, neighB):
        sa = set(a.tolist())
        sb = set(b.tolist())
        inter = len(sa & sb)
        union = len(sa | sb)
        scores.append(inter / max(union, 1))
    return float(np.mean(scores))


def plot_stability_overlay(df: pd.DataFrame, coords_list: List[np.ndarray], out_path: str, title: str) -> None:
    """
    여러 run의 임베딩을 reference에 정렬한 후 overlay 시각화.
    """
    plt.figure(figsize=(15, 12), dpi=200)
    plt.axis("off")
    plt.title(title, fontsize=18, weight="bold")

    # reference
    ref = coords_list[0]
    # 색은 persona 기준으로 한 번만
    base = df.copy()
    base["x"] = ref[:, 0]
    base["y"] = ref[:, 1]

    # overlay: 각 run을 낮은 alpha로
    for r, coords in enumerate(coords_list):
        alpha = 0.18 if r > 0 else 0.30
        for persona in [p for p in PERSONA_ORDER if p in base["persona"].unique()]:
            color = PERSONA_PALETTE.get(persona, "#333333")
            sub_idx = (base["persona"].values == persona)
            plt.scatter(coords[sub_idx, 0], coords[sub_idx, 1], s=3, c=color, alpha=alpha, linewidths=0)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[OK] Saved: {out_path}")


# -----------------------------
# Main
# -----------------------------

def main():
    np.random.seed(RANDOM_SEED)

    print("[INFO] Loading agents...")
    agents_df = load_agents(AGENTS_CSV)

    print("[INFO] Accumulating post keyword counts (streaming)...")
    post_acc = accumulate_post_counts(POSTS_CSV, DB_PATH, CHUNKSIZE)

    print("[INFO] Scoring + labeling personas...")
    persona_cols = list(PERSONA_KEYWORDS.keys())
    result_df = build_persona_table(agents_df, post_acc)

    # 1) Distribution plot
    plot_persona_distribution(
        result_df,
        os.path.join(OUTPUT_DIR, "persona_distribution_v3.png")
    )
    
    print("\n=== V3 Persona Distribution Counts ===")
    print(result_df["persona"].value_counts())
    print("======================================\n")

    # 2) Embedding: main run
    print("[INFO] Computing t-SNE embedding (main run)...")
    coords, mask_nz = compute_embedding(
        result_df,
        persona_cols=persona_cols,
        seed=RANDOM_SEED,
        perplexity=TSNE_PERPLEXITY,
        n_iter=TSNE_N_ITER
    )

    plot_landscape(
        result_df,
        coords,
        os.path.join(OUTPUT_DIR, "persona_landscape_v3_no_labels.png"),
        title="Moltbook Agent Landscape (v3)",
        with_labels=False
    )

    plot_landscape(
        result_df,
        coords,
        os.path.join(OUTPUT_DIR, "persona_landscape_v3.png"),
        title="Moltbook Agent Landscape (v3)",
        with_labels=True
    )

    # 3) Stability: 3 runs
    print("[INFO] Stability runs (3x t-SNE + Procrustes alignment)...")
    seeds = [RANDOM_SEED, RANDOM_SEED + 1, RANDOM_SEED + 2]

    coords_runs = []
    for s in seeds:
        c, _ = compute_embedding(
            result_df,
            persona_cols=persona_cols,
            seed=s,
            perplexity=TSNE_PERPLEXITY,
            n_iter=TSNE_N_ITER
        )
        coords_runs.append(c)

    # Procrustes align to first run (전체 좌표를 정렬)
    ref = coords_runs[0]
    aligned = [ref]
    for c in coords_runs[1:]:
        aligned.append(orthogonal_procrustes_align(c, ref))

    # kNN overlap (Observer 포함하면 “섬” 배치가 인위적이므로 nonzero만 보는 게 더 정직)
    nz_idx = np.where(mask_nz)[0]
    A = aligned[0][nz_idx]
    B = aligned[1][nz_idx]
    C = aligned[2][nz_idx]

    overlap_AB = knn_overlap(A, B, k=15, sample_n=3000, seed=RANDOM_SEED)
    overlap_AC = knn_overlap(A, C, k=15, sample_n=3000, seed=RANDOM_SEED)
    overlap_BC = knn_overlap(B, C, k=15, sample_n=3000, seed=RANDOM_SEED)

    report = (
        "=== t-SNE Stability Report (nonzero agents only) ===\n"
        f"Points used: {len(nz_idx)}\n"
        "Metric: mean Jaccard overlap of kNN sets (k=15)\n"
        f"Run0 vs Run1: {overlap_AB:.4f}\n"
        f"Run0 vs Run2: {overlap_AC:.4f}\n"
        f"Run1 vs Run2: {overlap_BC:.4f}\n"
        "\nInterpretation (rule of thumb):\n"
        "- ~0.6+ : 꽤 안정적(로컬 구조 반복)\n"
        "- ~0.4~0.6 : 중간(파라미터/스코어링에 민감)\n"
        "- <0.4 : 불안정(특징/퍼플렉시티/Observer 처리 재검토 권장)\n"
    )

    report_path = os.path.join(OUTPUT_DIR, "tsne_stability_report_v3.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"[OK] Saved: {report_path}")
    print(report)

    print("[INFO] Generating Influence-weighted Landscape...")
    lb_path = os.path.join(OUTPUT_DIR, "agent_leaderboard_full.csv")
    if os.path.exists(lb_path):
        lb = pd.read_csv(lb_path)
        # result_df has 'name'
        # lb has 'agent' (name)
        lb_map = dict(zip(lb["agent"], lb["status_index"]))
        influence = result_df["name"].map(lambda n: lb_map.get(str(n), 0.0))
        
        plot_landscape(
            result_df,
            aligned[0],
            os.path.join(OUTPUT_DIR, "persona_influence_map.png"),
            title="Moltbook Agent Influence Map (Size by Status Index)",
            with_labels=True,
            influence_sizes=influence
        )
        print(f"[OK] Saved: {os.path.join(OUTPUT_DIR, 'persona_influence_map.png')}")
        
        # Save coordinates for Galaxy Map
        coords_df = pd.DataFrame({
            "agent": result_df["name"],
            "x": aligned[0][:, 0],
            "y": aligned[0][:, 1]
        })
        coords_path = os.path.join(OUTPUT_DIR, "agent_coordinates_v3.csv")
        coords_df.to_csv(coords_path, index=False)
        print(f"[OK] Saved coordinates: {coords_path}")
    else:
        print("[WARN] agent_leaderboard_full.csv not found. Skipping influence map.")

    print("[DONE] v3 pipeline finished.")


if __name__ == "__main__":
    main()
