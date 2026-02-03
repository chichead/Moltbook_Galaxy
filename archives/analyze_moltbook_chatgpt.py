# -*- coding: utf-8 -*-
"""
Moltbook Submolt + Social Analysis (Rewrite v2)

목표:
- 서브몰트 동역학(게시물 중심): volume뿐 아니라 시간감쇠 기반 hotness/velocity/accel로 측정
- 에이전트 사회적 위계(작성자/상호작용): 댓글 네트워크 기반으로 중심성/리더보드 산출
- 네트워크 분석 산출물을 '이미지 + CSV(노드/엣지)'로 남겨 재사용 가능하게

핵심 개선:
- SELECT * 금지(필요 컬럼만 로드)
- created_at 파싱 강제(errors='coerce', utc=True) + NaT 비율 출력
- post_id/id 타입 통일(str)
- 댓글 네트워크는 DB+CSV를 스트리밍 누적(Counter) -> 대용량 안전
- follower 네트워크가 비어도(현 DB) 댓글/전이 네트워크로 충분히 구조화 가능
"""

import os
import sqlite3
import math
from dataclasses import dataclass
from collections import Counter, defaultdict
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx


# -----------------------------
# Config
# -----------------------------

@dataclass
class Config:
    db_path: str = "moltbook_observatory.db"
    comments_csv_path: str = "moltbook_comments.csv"  # 없으면 자동 무시
    output_dir: str = "analysis_results"

    # 동역학(핫도) 파라미터
    window_hours: int = 24            # "최근 24h 핫도" 기본
    tau_hours: float = 12.0           # 시간 감쇠(half-life 느낌) -> 12h면 빠른 커뮤니티에 적당
    alpha_comment: float = 0.7        # 댓글 수 가중치
    diversity_beta: float = 0.30      # unique_authors 보정

    # 네트워크 시각화 제한(너무 커지면 정적 plot이 의미 없어짐)
    top_edges_to_draw: int = 80
    top_nodes_to_label: int = 25
    pagerank_max_nodes: int = 3000    # 너무 큰 그래프는 PR 계산이 무거워서 상위 노드로 축소

    # CSV 스트리밍
    csv_chunksize: int = 200_000

    random_seed: int = 42


CFG = Config()


# -----------------------------
# Helpers
# -----------------------------

def ensure_output_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def safe_datetime_utc(s: pd.Series, col_name: str) -> pd.Series:
    """UTC로 강제 파싱. 실패는 NaT로 만들고 비율을 출력한다."""
    dt = pd.to_datetime(s, errors="coerce", utc=True)
    bad = dt.isna().mean()
    print(f"[TIME] {col_name}: NaT ratio = {bad:.2%}")
    return dt

def clip_nonneg(x: pd.Series) -> pd.Series:
    return pd.to_numeric(x, errors="coerce").fillna(0).clip(lower=0)

def log1p_safe(x: float) -> float:
    return math.log1p(max(0.0, x))

def save_df(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False)
    print(f"[OK] Saved: {path}")

def barh_plot(labels, values, title, xlabel, out_path):
    """수평 막대그래프(가독성 좋음)."""
    plt.figure(figsize=(12, 6), dpi=160)
    y = np.arange(len(labels))
    plt.barh(y, values)
    plt.yticks(y, labels)
    plt.gca().invert_yaxis()
    plt.title(title, fontsize=14, weight="bold")
    plt.xlabel(xlabel)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[OK] Saved: {out_path}")


# -----------------------------
# Load minimal data
# -----------------------------

def load_posts_min(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    posts에서 필요한 컬럼만 로드.
    id/post_id는 나중에 comments merge를 위해 str로 통일한다.
    """
    q = """
    SELECT
      id,
      agent_id,
      agent_name,
      submolt,
      score,
      comment_count,
      created_at
    FROM posts
    """
    posts = pd.read_sql_query(q, conn)

    # 타입 정리
    posts["id"] = posts["id"].astype(str)
    posts["agent_id"] = posts["agent_id"].astype(str)
    posts["agent_name"] = posts["agent_name"].fillna("").astype(str)
    posts["submolt"] = posts["submolt"].fillna("").astype(str)

    # 시간 파싱 강제
    posts["created_at"] = safe_datetime_utc(posts["created_at"], "posts.created_at")
    return posts


def load_agents_min(conn: sqlite3.Connection) -> pd.DataFrame:
    q = """
    SELECT
      id,
      name,
      karma,
      follower_count,
      following_count,
      first_seen_at,
      last_seen_at
    FROM agents
    """
    agents = pd.read_sql_query(q, conn)
    agents["id"] = agents["id"].astype(str)
    agents["name"] = agents["name"].fillna("").astype(str)
    agents["karma"] = pd.to_numeric(agents["karma"], errors="coerce")
    agents["follower_count"] = pd.to_numeric(agents["follower_count"], errors="coerce")
    agents["following_count"] = pd.to_numeric(agents["following_count"], errors="coerce")
    return agents


def load_submolts_min(conn: sqlite3.Connection) -> pd.DataFrame:
    q = """
    SELECT
      name,
      display_name,
      subscriber_count,
      created_at
    FROM submolts
    """
    submolts = pd.read_sql_query(q, conn)
    submolts["name"] = submolts["name"].fillna("").astype(str)
    submolts["display_name"] = submolts["display_name"].fillna("").astype(str)
    submolts["subscriber_count"] = pd.to_numeric(submolts["subscriber_count"], errors="coerce")
    return submolts


def load_db_comments_min(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    DB comments는 agent_id가 있으나, CSV와 합칠 때 name 기반으로 통일할 것이므로
    최소로 필요한 컬럼만 읽는다.
    """
    q = """
    SELECT
      id,
      post_id,
      agent_id,
      agent_name,
      parent_id,
      created_at
    FROM comments
    """
    c = pd.read_sql_query(q, conn)

    c["id"] = c["id"].astype(str)
    c["post_id"] = c["post_id"].astype(str)
    c["agent_id"] = c["agent_id"].astype(str)
    c["agent_name"] = c["agent_name"].fillna("").astype(str)
    c["parent_id"] = c["parent_id"].astype(str)

    c["created_at"] = safe_datetime_utc(c["created_at"], "comments.created_at")
    return c


def iter_csv_comments(path: str, chunksize: int):
    """
    comments.csv를 스트리밍으로 읽는다.
    기대 컬럼(일반적으로):
      id, post_id, agent_name, parent_id, created_at, content, score ...
    -> 여기서는 네트워크에 필요한 최소만 읽음.
    """
    if not os.path.exists(path):
        print(f"[INFO] CSV comments not found, skipping: {path}")
        return

    usecols = ["id", "post_id", "agent_name", "parent_id", "created_at"]
    for chunk in pd.read_csv(path, usecols=lambda c: c in usecols, chunksize=chunksize):
        # 컬럼 누락 방어
        for col in usecols:
            if col not in chunk.columns:
                chunk[col] = np.nan

        chunk["id"] = chunk["id"].astype(str)
        chunk["post_id"] = chunk["post_id"].astype(str)
        chunk["agent_name"] = chunk["agent_name"].fillna("").astype(str)
        chunk["parent_id"] = chunk["parent_id"].astype(str)
        chunk["created_at"] = safe_datetime_utc(chunk["created_at"], "csv_comments.created_at")

        yield chunk


# -----------------------------
# Submolt dynamics (post-centric)
# -----------------------------

def compute_post_energy(posts: pd.DataFrame, tmax: pd.Timestamp, alpha_comment: float, tau_hours: float) -> pd.Series:
    """
    에너지 정의:
      energy = (sqrt(score_pos) + alpha*sqrt(comment_pos)) * exp(-age_hours/tau)
    """
    score_pos = clip_nonneg(posts["score"])
    cmt_pos = clip_nonneg(posts["comment_count"])

    # age_hours
    age = (tmax - posts["created_at"]) / pd.Timedelta(hours=1)
    age = age.fillna(np.inf).clip(lower=0)

    energy = (np.sqrt(score_pos) + alpha_comment * np.sqrt(cmt_pos)) * np.exp(-age / tau_hours)
    return energy


def analyze_dynamics(posts: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """
    동역학 분석:
    - top submolts by volume
    - top submolts by (time-decayed) hotness
    - activity timeseries
    """
    print("[DYNAMICS] analyzing submolt dynamics...")

    # 시간 유효한 posts만
    valid = posts.dropna(subset=["created_at"]).copy()
    if valid.empty:
        print("[WARN] No valid created_at in posts. Skipping dynamics.")
        return pd.DataFrame()

    tmax = valid["created_at"].max()
    window_start = tmax - pd.Timedelta(hours=cfg.window_hours)
    w = valid[valid["created_at"] >= window_start].copy()

    if w.empty:
        print("[WARN] No posts in window. Using all valid posts.")
        w = valid.copy()

    # 기본 volume 지표
    top_volume = w["submolt"].value_counts().head(15)
    barh_plot(
        labels=top_volume.index.tolist(),
        values=top_volume.values.tolist(),
        title=f"Top Submolts by Post Count (last {cfg.window_hours}h)",
        xlabel="Posts",
        out_path=os.path.join(cfg.output_dir, "top_submolts_posts_window.png")
    )

    # 에너지/핫도 계산
    w["energy"] = compute_post_energy(w, tmax=tmax, alpha_comment=cfg.alpha_comment, tau_hours=cfg.tau_hours)

    agg = w.groupby("submolt").agg(
        posts=("submolt", "size"),
        unique_authors=("agent_name", pd.Series.nunique),
        score_sum=("score", "sum"),
        comments_sum=("comment_count", "sum"),
        energy_sum=("energy", "sum"),
    ).reset_index()

    # hotness: energy_sum * (1 + beta*log1p(unique_authors))
    agg["hotness"] = agg["energy_sum"] * (1.0 + cfg.diversity_beta * agg["unique_authors"].map(log1p_safe))

    # velocity: posts/hour
    hours = max(1.0, cfg.window_hours)
    agg["posts_per_hour"] = agg["posts"] / hours

    # accel: 최근 window를 반으로 쪼개 증가량(간단하지만 해석 좋음)
    half = pd.Timedelta(hours=cfg.window_hours / 2)
    w_recent = w[w["created_at"] >= (tmax - half)]
    w_prev = w[(w["created_at"] < (tmax - half)) & (w["created_at"] >= window_start)]

    recent_cnt = w_recent["submolt"].value_counts()
    prev_cnt = w_prev["submolt"].value_counts()

    agg["accel_posts"] = agg["submolt"].map(lambda s: float(recent_cnt.get(s, 0) - prev_cnt.get(s, 0)))

    # Top hotness plot
    top_hot = agg.sort_values("hotness", ascending=False).head(15)
    barh_plot(
        labels=top_hot["submolt"].tolist(),
        values=top_hot["hotness"].tolist(),
        title=f"Top Submolts by Hotness (time-decay, last {cfg.window_hours}h)",
        xlabel="Hotness",
        out_path=os.path.join(cfg.output_dir, "top_submolts_hotness_window.png")
    )

    # Activity timeseries (hourly)
    hourly = valid.set_index("created_at").resample("h").size()
    plt.figure(figsize=(14, 5), dpi=160)
    plt.plot(hourly.index, hourly.values)
    plt.title("Hourly Post Activity (UTC)", fontsize=13, weight="bold")
    plt.ylabel("Posts / hour")
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.output_dir, "hourly_post_activity.png"))
    plt.close()
    print(f"[OK] Saved: {os.path.join(cfg.output_dir, 'hourly_post_activity.png')}")

    # Export dynamics table
    agg.sort_values("hotness", ascending=False, inplace=True)
    save_df(agg, os.path.join(cfg.output_dir, "submolt_dynamics_window.csv"))
    return agg


# -----------------------------
# Submolt transition network (agent flow)
# -----------------------------

def analyze_submolt_transitions(posts: pd.DataFrame, cfg: Config) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    에이전트가 서브몰트를 옮겨다니는 전이 네트워크.
    - 노드: submolt
    - 엣지: prev_submolt -> next_submolt (같은 agent의 시간순 포스팅)
    """
    print("[TRANSITION] analyzing submolt transitions...")

    valid = posts.dropna(subset=["created_at"]).copy()
    if valid.empty:
        print("[WARN] No valid created_at. Skipping transitions.")
        return pd.DataFrame(), pd.DataFrame()

    # agent_key: agent_id가 있으면 사용, 없으면 agent_name 사용
    agent_key = valid["agent_id"].fillna("").astype(str)
    name_key = valid["agent_name"].fillna("").astype(str)
    valid["agent_key"] = np.where(agent_key.str.len() > 0, agent_key, name_key)

    valid.sort_values(["agent_key", "created_at"], inplace=True)
    valid["next_submolt"] = valid.groupby("agent_key")["submolt"].shift(-1)

    trans = valid.dropna(subset=["next_submolt"]).copy()
    trans = trans[trans["submolt"].astype(str) != ""]
    trans = trans[trans["next_submolt"].astype(str) != ""]

    # self-loop는 옵션. 일단 기본은 제거(“이동”만 보고 싶다면)
    trans = trans[trans["submolt"] != trans["next_submolt"]]

    edges = trans.groupby(["submolt", "next_submolt"]).size().reset_index(name="weight")
    edges.sort_values("weight", ascending=False, inplace=True)

    if edges.empty:
        print("[WARN] No transitions found.")
        return pd.DataFrame(), pd.DataFrame()

    # 노드 지표: in/out flow, pagerank(상위 노드에 한해 계산)
    out_flow = edges.groupby("submolt")["weight"].sum()
    in_flow = edges.groupby("next_submolt")["weight"].sum()

    nodes = pd.DataFrame({
        "submolt": pd.Index(out_flow.index).union(in_flow.index)
    })
    nodes["out_flow"] = nodes["submolt"].map(lambda s: float(out_flow.get(s, 0)))
    nodes["in_flow"] = nodes["submolt"].map(lambda s: float(in_flow.get(s, 0)))
    nodes["total_flow"] = nodes["in_flow"] + nodes["out_flow"]

    # PageRank는 너무 크면 비싸니 상위 flow 노드로 축소
    top_nodes = nodes.sort_values("total_flow", ascending=False).head(cfg.pagerank_max_nodes)["submolt"].tolist()
    edges_pr = edges[edges["submolt"].isin(top_nodes) & edges["next_submolt"].isin(top_nodes)].copy()

    G = nx.DiGraph()
    for r in edges_pr.itertuples(index=False):
        G.add_edge(r.submolt, r.next_submolt, weight=float(r.weight))

    pr = nx.pagerank(G, weight="weight") if G.number_of_nodes() > 0 else {}
    nodes["pagerank"] = nodes["submolt"].map(lambda s: float(pr.get(s, 0.0)))

    # Export CSV
    save_df(nodes.sort_values("pagerank", ascending=False), os.path.join(cfg.output_dir, "submolt_transition_nodes.csv"))
    save_df(edges, os.path.join(cfg.output_dir, "submolt_transition_edges.csv"))

    # Print top hubs
    top_hubs = nodes.sort_values("pagerank", ascending=False).head(15)[["submolt", "pagerank", "in_flow", "out_flow"]]
    print("[TRANSITION] Top Submolt Hubs (PageRank):")
    print(top_hubs.to_string(index=False))

    # Draw a readable subgraph (top edges)
    top_edges = edges.head(cfg.top_edges_to_draw).copy()
    subG = nx.DiGraph()
    for r in top_edges.itertuples(index=False):
        subG.add_edge(r.submolt, r.next_submolt, weight=float(r.weight))

    if subG.number_of_nodes() > 1:
        plt.figure(figsize=(14, 14), dpi=170)
        pos = nx.spring_layout(subG, k=1.2, iterations=60, seed=cfg.random_seed)

        # node size: pagerank (clip)
        node_sizes = []
        for n in subG.nodes():
            v = pr.get(n, 0.0)
            node_sizes.append(200 + 15000 * min(v, 0.01))

        weights = [subG[u][v]["weight"] for u, v in subG.edges()]
        maxw = max(weights) if weights else 1.0
        widths = [1.0 + 5.0 * math.log1p(w) / math.log1p(maxw) for w in weights]

        nx.draw_networkx_nodes(subG, pos, node_size=node_sizes, alpha=0.75)
        nx.draw_networkx_edges(subG, pos, width=widths, alpha=0.35, arrows=True, arrowsize=12,
                               connectionstyle="arc3,rad=0.08")

        # label: top pagerank nodes만
        label_nodes = set(top_hubs["submolt"].head(cfg.top_nodes_to_label).tolist())
        labels = {n: n for n in subG.nodes() if n in label_nodes}
        nx.draw_networkx_labels(subG, pos, labels=labels, font_size=9)

        plt.title("Submolt Transition Network (Top edges)", fontsize=14, weight="bold")
        plt.axis("off")
        plt.tight_layout()
        out_path = os.path.join(cfg.output_dir, "submolt_transitions.png")
        plt.savefig(out_path)
        plt.close()
        print(f"[OK] Saved: {out_path}")

    return nodes, edges


# -----------------------------
# Comment interaction network (agent -> agent)
# -----------------------------

def build_post_author_map(posts: pd.DataFrame) -> dict:
    """post_id(str) -> post_author_name"""
    m = dict(zip(posts["id"].astype(str), posts["agent_name"].fillna("").astype(str)))
    return m


def accumulate_comment_edges_from_df(
    comments_df: pd.DataFrame,
    post_author_map: dict,
    edge_counter: Counter,
    coverage_counter: dict
):
    """
    comments_df에서 edge_counter[(src_name, tgt_name)] += 1 누적
    coverage_counter에는 매칭/스킵 통계 기록
    """
    # 타입 통일
    post_id = comments_df["post_id"].astype(str)
    src = comments_df["agent_name"].fillna("").astype(str)

    # 타겟(포스트 작성자) 매핑
    tgt = post_id.map(post_author_map)

    # 매칭 통계
    coverage_counter["rows_total"] += len(comments_df)
    coverage_counter["rows_missing_post"] += int(tgt.isna().sum())

    # 유효 rows
    ok = (~tgt.isna()) & (src != "") & (tgt != "")
    sub = pd.DataFrame({"src": src[ok].values, "tgt": tgt[ok].values})

    # self reply 제거
    sub = sub[sub["src"] != sub["tgt"]]
    coverage_counter["rows_self_reply"] += int((src[ok].values == tgt[ok].values).sum())
    coverage_counter["rows_used"] += len(sub)

    # group count -> counter update
    grp = sub.groupby(["src", "tgt"]).size()
    for (s, t), w in grp.items():
        edge_counter[(s, t)] += int(w)


def analyze_comment_network(posts: pd.DataFrame, db_comments: pd.DataFrame, cfg: Config) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    댓글 기반 상호작용 네트워크:
      src = 댓글 작성자
      tgt = 포스트 작성자
    DB comments + (옵션) CSV comments 스트리밍을 합쳐 edges를 누적.
    """
    print("[COMMENT] analyzing comment interaction network...")

    post_author_map = build_post_author_map(posts)

    edge_counter = Counter()
    coverage = defaultdict(int)

    # 1) DB comments 누적
    if db_comments is not None and not db_comments.empty:
        accumulate_comment_edges_from_df(db_comments, post_author_map, edge_counter, coverage)

    # 2) CSV comments 스트리밍 누적
    if os.path.exists(cfg.comments_csv_path):
        for chunk in iter_csv_comments(cfg.comments_csv_path, cfg.csv_chunksize):
            accumulate_comment_edges_from_df(chunk, post_author_map, edge_counter, coverage)

    if not edge_counter:
        print("[WARN] No comment interactions found (after mapping).")
        return pd.DataFrame(), pd.DataFrame()

    print("[COMMENT] Coverage stats:")
    for k in ["rows_total", "rows_missing_post", "rows_self_reply", "rows_used"]:
        print(f"  - {k}: {coverage[k]:,}")

    # edges df
    edges = pd.DataFrame([(s, t, w) for (s, t), w in edge_counter.items()],
                         columns=["src", "tgt", "weight"])
    edges.sort_values("weight", ascending=False, inplace=True)

    # 노드 지표(간단): in/out 가중합
    out_w = edges.groupby("src")["weight"].sum()
    in_w = edges.groupby("tgt")["weight"].sum()

    nodes = pd.DataFrame({"agent": pd.Index(out_w.index).union(in_w.index)})
    nodes["out_weight"] = nodes["agent"].map(lambda a: float(out_w.get(a, 0)))
    nodes["in_weight"] = nodes["agent"].map(lambda a: float(in_w.get(a, 0)))
    nodes["total_weight"] = nodes["out_weight"] + nodes["in_weight"]

    # pagerank는 상위 노드로 축소해서 계산(성능/안정)
    top_agents = nodes.sort_values("total_weight", ascending=False).head(cfg.pagerank_max_nodes)["agent"].tolist()
    edges_pr = edges[edges["src"].isin(top_agents) & edges["tgt"].isin(top_agents)].copy()

    G = nx.DiGraph()
    for r in edges_pr.itertuples(index=False):
        G.add_edge(r.src, r.tgt, weight=float(r.weight))

    pr = nx.pagerank(G, weight="weight") if G.number_of_nodes() > 0 else {}
    nodes["pagerank"] = nodes["agent"].map(lambda a: float(pr.get(a, 0.0)))

    # Export CSV
    save_df(nodes.sort_values("pagerank", ascending=False), os.path.join(cfg.output_dir, "comment_network_nodes.csv"))
    save_df(edges, os.path.join(cfg.output_dir, "comment_network_edges.csv"))

    # Print top PR
    print("[COMMENT] Top Agents by PageRank:")
    print(nodes.sort_values("pagerank", ascending=False).head(15)[["agent", "pagerank", "in_weight", "out_weight"]].to_string(index=False))

    # Draw readable subgraph: top edges
    top_edges = edges.head(cfg.top_edges_to_draw).copy()
    subG = nx.DiGraph()
    for r in top_edges.itertuples(index=False):
        subG.add_edge(r.src, r.tgt, weight=float(r.weight))

    if subG.number_of_nodes() > 1:
        plt.figure(figsize=(14, 14), dpi=170)
        pos = nx.spring_layout(subG, k=1.3, iterations=60, seed=cfg.random_seed)

        weights = [subG[u][v]["weight"] for u, v in subG.edges()]
        maxw = max(weights) if weights else 1.0
        widths = [1.0 + 5.0 * math.log1p(w) / math.log1p(maxw) for w in weights]

        # 노드 크기: (subgraph) PR 기반, 없으면 total_weight 기반 fallback
        node_sizes = []
        for n in subG.nodes():
            v = pr.get(n, 0.0)
            if v > 0:
                node_sizes.append(200 + 16000 * min(v, 0.01))
            else:
                tw = float(nodes.set_index("agent").loc[n, "total_weight"]) if n in set(nodes["agent"]) else 1.0
                node_sizes.append(150 + 40 * math.log1p(tw))

        nx.draw_networkx_nodes(subG, pos, node_size=node_sizes, alpha=0.75)
        nx.draw_networkx_edges(subG, pos, width=widths, alpha=0.35, arrows=True, arrowsize=12,
                               connectionstyle="arc3,rad=0.07")

        # label: 상위 PR 노드만
        top_label = nodes.sort_values("pagerank", ascending=False).head(cfg.top_nodes_to_label)["agent"].tolist()
        labels = {n: n for n in subG.nodes() if n in set(top_label)}
        nx.draw_networkx_labels(subG, pos, labels=labels, font_size=8)

        plt.title("Comment Interaction Network (Top edges)", fontsize=14, weight="bold")
        plt.axis("off")
        plt.tight_layout()
        out_path = os.path.join(cfg.output_dir, "comment_network.png")
        plt.savefig(out_path)
        plt.close()
        print(f"[OK] Saved: {out_path}")

    return nodes, edges


# -----------------------------
# Agent profile (target-focused)
# -----------------------------

def analyze_agent_profile(target_name: str, posts: pd.DataFrame, comment_edges: pd.DataFrame, cfg: Config) -> None:
    """
    특정 에이전트의 프로필:
    - 게시글 수 / 서브몰트 분포
    - inbound: 누가 target에게 댓글을 다는가?  (src -> target)
    - outbound: target이 누구에게 댓글을 다는가? (target -> tgt)
    주의: comment_edges는 "댓글작성자 -> 포스트작성자" 기반으로 누적된 edge임.
    """
    print(f"[PROFILE] analyzing agent profile: {target_name}")

    # posts stats
    agent_posts = posts[posts["agent_name"] == target_name]
    print(f"  - Total posts: {len(agent_posts):,}")

    if not agent_posts.empty:
        top_sub = agent_posts["submolt"].value_counts().head(10)
        barh_plot(
            labels=top_sub.index.tolist(),
            values=top_sub.values.tolist(),
            title=f"{target_name}: Top Submolts (Posting)",
            xlabel="Posts",
            out_path=os.path.join(cfg.output_dir, f"{target_name}_top_submolts.png")
        )

    if comment_edges is None or comment_edges.empty:
        print("  - No comment edges available.")
        return

    inbound = comment_edges[comment_edges["tgt"] == target_name].sort_values("weight", ascending=False).head(20)
    outbound = comment_edges[comment_edges["src"] == target_name].sort_values("weight", ascending=False).head(20)

    if not inbound.empty:
        barh_plot(
            labels=inbound["src"].tolist(),
            values=inbound["weight"].tolist(),
            title=f"Who talks to {target_name}? (Inbound Attention)",
            xlabel="Observed comments",
            out_path=os.path.join(cfg.output_dir, f"{target_name}_inbound.png")
        )
    else:
        print("  - Inbound: none observed.")

    if not outbound.empty:
        barh_plot(
            labels=outbound["tgt"].tolist(),
            values=outbound["weight"].tolist(),
            title=f"Who does {target_name} talk to? (Outbound Replies)",
            xlabel="Observed comments",
            out_path=os.path.join(cfg.output_dir, f"{target_name}_outbound.png")
        )
    else:
        print("  - Outbound: none observed.")


# -----------------------------
# Leaderboards (agents)
# -----------------------------

def build_agent_leaderboard(agents: pd.DataFrame, comment_nodes: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """
    정적 스펙(karma/followers) + 상호작용 중심성(pagerank) 결합 리더보드.
    follower 그래프가 없더라도 노드 속성으로 충분히 '위계감'을 표현할 수 있음.
    """
    print("[LEADERBOARD] building agent leaderboard...")

    # agents 테이블을 name 기준으로 병합 (comment_nodes는 agent name 기반)
    base = agents[["name", "karma", "follower_count", "following_count"]].copy()
    base.rename(columns={"name": "agent"}, inplace=True)

    if comment_nodes is None or comment_nodes.empty:
        base["pagerank"] = 0.0
        base["in_weight"] = 0.0
        base["out_weight"] = 0.0
        base["total_weight"] = 0.0
        lb = base
    else:
        lb = pd.merge(base, comment_nodes, on="agent", how="outer")
        for c in ["karma", "follower_count", "following_count", "pagerank", "in_weight", "out_weight", "total_weight"]:
            if c in lb.columns:
                lb[c] = pd.to_numeric(lb[c], errors="coerce").fillna(0.0)

    # 간단 합성(과도한 튜닝 없이도 해석 가능한 버전)
    # - pagerank, total_weight는 이미 구조/행동 기반
    # - follower/karma는 보조
    lb["score_struct"] = np.log1p(lb["total_weight"]) + 5.0 * lb["pagerank"]
    lb["score_reach"] = np.log1p(lb["follower_count"]) + 0.5 * np.log1p(lb["karma"].clip(lower=0))
    lb["status_index"] = 0.7 * lb["score_struct"] + 0.3 * lb["score_reach"]

    lb.sort_values("status_index", ascending=False, inplace=True)
    save_df(lb.head(200), os.path.join(cfg.output_dir, "agent_leaderboard_top200.csv"))
    # Save full leaderboard for persona map weighting
    save_df(lb, os.path.join(cfg.output_dir, "agent_leaderboard_full.csv"))
    print("[LEADERBOARD] Top 15 agents:")
    print(lb.head(15)[["agent", "status_index", "pagerank", "total_weight", "follower_count", "karma"]].to_string(index=False))

    return lb


# -----------------------------
# Main
# -----------------------------

def main(cfg: Config):
    ensure_output_dir(cfg.output_dir)

    if not os.path.exists(cfg.db_path):
        raise FileNotFoundError(f"DB not found: {cfg.db_path}")

    print(f"[INFO] DB: {cfg.db_path}")
    print(f"[INFO] Comments CSV: {cfg.comments_csv_path} (exists={os.path.exists(cfg.comments_csv_path)})")

    conn = sqlite3.connect(cfg.db_path)

    try:
        posts = load_posts_min(conn)
        agents = load_agents_min(conn)
        submolts = load_submolts_min(conn)

        # follows는 현재 비어있을 가능성이 높아서 로드 자체는 생략(필요시 추가 가능)

        # DB comments 로드(작을 때는 OK, 커지면 여기서도 SQL chunking으로 바꿔도 됨)
        db_comments = load_db_comments_min(conn)

    finally:
        conn.close()

    # 1) Dynamics
    dynamics = analyze_dynamics(posts, cfg)

    # 2) Comment interaction network (DB + CSV)
    comment_nodes, comment_edges = analyze_comment_network(posts, db_comments, cfg)

    # 3) Submolt transitions (flow)
    trans_nodes, trans_edges = analyze_submolt_transitions(posts, cfg)

    # 4) Leaderboard (agents)
    leaderboard = build_agent_leaderboard(agents, comment_nodes, cfg)

    # 5) Example: agent profile
    target = "botcrong"
    analyze_agent_profile(target, posts, comment_edges, cfg)

    print(f"[DONE] Results saved in: {cfg.output_dir}")


if __name__ == "__main__":
    main(CFG)
