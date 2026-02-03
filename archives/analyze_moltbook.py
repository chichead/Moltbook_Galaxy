import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import os

# Set style
sns.set_theme(style="whitegrid")
OUTPUT_DIR = "analysis_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data(db_path):
    print(f"Loading data from {db_path}...")
    conn = sqlite3.connect(db_path)
    
    posts = pd.read_sql_query("SELECT * FROM posts", conn)
    try:
        posts['created_at'] = pd.to_datetime(posts['created_at'])
    except Exception as e:
        print(f"Warning: Could not convert posts.created_at to datetime: {e}")
        
    agents = pd.read_sql_query("SELECT * FROM agents", conn)
    submolts = pd.read_sql_query("SELECT * FROM submolts", conn)
    follows = pd.read_sql_query("SELECT * FROM follows", conn)
    
    conn.close()
    return posts, agents, submolts, follows

def analyze_dynamics(posts, submolts):
    print("Analyzing dynamics...")
    
    # 1. Top Submolts by Post Count
    top_submolts = posts['submolt'].value_counts().head(10)
    plt.figure(figsize=(12, 6))
    sns.barplot(x=top_submolts.values, y=top_submolts.index, palette="viridis")
    plt.title("Top 10 Submolts by Post Count")
    plt.xlabel("Number of Posts")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/top_submolts_posts.png")
    plt.close()
    
    # 2. Activity Over Time (Daily)
    if pd.api.types.is_datetime64_any_dtype(posts['created_at']):
        daily_posts = posts.set_index('created_at').resample('D').size()
        plt.figure(figsize=(14, 6))
        daily_posts.plot()
        plt.title("Daily Post Activity")
        plt.ylabel("Number of Posts")
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/daily_activity.png")
        plt.close()
    
    # 3. Top Submolts by Score (Total Score)
    submolt_scores = posts.groupby('submolt')['score'].sum().sort_values(ascending=False).head(10)
    plt.figure(figsize=(12, 6))
    sns.barplot(x=submolt_scores.values, y=submolt_scores.index, palette="magma")
    plt.title("Top 10 Submolts by Total Score")
    plt.xlabel("Total Score")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/top_submolts_score.png")
    plt.close()

def analyze_social(agents, follows):
    print("Analyzing social hierarchy...")
    
    # 1. Agent Karma Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(agents['karma'], bins=50, log_scale=(False, True)) # Log scale on Y for better visibility of long tail
    plt.title("Agent Karma Distribution")
    plt.xlabel("Karma")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/agent_karma_dist.png")
    plt.close()
    
    # 2. Top Agents
    top_agents_karma = agents.nlargest(10, 'karma')[['name', 'karma']]
    print("Top Agents by Karma:\n", top_agents_karma)
    top_agents_karma.to_csv(f"{OUTPUT_DIR}/top_agents_karma.csv", index=False)
    
    
    # 3. Follower Network Graph (Skipped due to empty table, replaced by Comment Network)
    print("Skipping Follower Network (Data empty).")

def analyze_comment_network(conn):
    print("Analyzing comment interaction network...")
    
    query = """
    SELECT 
        c.agent_id as source_id,
        c.agent_name as source_name,
        p.agent_id as target_id,
        p.agent_name as target_name,
        COUNT(*) as weight
    FROM comments c
    JOIN posts p ON c.post_id = p.id
    WHERE c.agent_id != p.agent_id
    GROUP BY c.agent_id, p.agent_id
    ORDER BY weight DESC
    """
    
    edges_df = pd.read_sql_query(query, conn)
    
    if edges_df.empty:
        print("No comment interactions found.")
        return

    # Create Graph
    G = nx.from_pandas_edgelist(
        edges_df, 
        source='source_id', 
        target='target_id', 
        edge_attr='weight', 
        create_using=nx.DiGraph()
    )
    
    # Calculate Centrality (PageRank or Degree)
    pagerank = nx.pagerank(G, weight='weight')
    top_pagerank = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:10]
    
    print("Top Agents manually correlated (Comment Network PageRank):")
    for agent_id, score in top_pagerank:
        # Try to find name in edges_df
        name_source = edges_df[edges_df['source_id'] == agent_id]['source_name'].unique()
        name_target = edges_df[edges_df['target_id'] == agent_id]['target_name'].unique()
        name = name_source[0] if len(name_source) > 0 else (name_target[0] if len(name_target) > 0 else agent_id)
        print(f"{name}: {score:.4f}")

    # Visualize Core Network (Filter for extensive visualization)
    # Take top 50 edges by weight
    top_edges = edges_df.head(50)
    subG = nx.from_pandas_edgelist(
        top_edges, 
        source='source_name', 
        target='target_name', 
        edge_attr='weight', 
        create_using=nx.DiGraph()
    )
    
    plt.figure(figsize=(14, 14))
    pos = nx.spring_layout(subG, k=2.0, iterations=50, seed=42)
    
    # Draw edges with varying thickness based on weight
    weights = [subG[u][v]['weight'] for u, v in subG.edges()]
    # Normalize weights for width
    width = [float(w) / max(weights) * 5 for w in weights]
    
    nx.draw_networkx_nodes(subG, pos, node_size=100, node_color='lightgreen', alpha=0.8)
    nx.draw_networkx_edges(subG, pos, edge_color='gray', alpha=0.5, arrows=True, width=width, arrowsize=20)
    nx.draw_networkx_labels(subG, pos, font_size=10, font_weight='bold')
    
    
    plt.title("Comment Interaction Network (Top 50 Interactions)")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/comment_network.png")
    plt.close()

def analyze_submolt_transitions(posts):
    print("Analyzing submolt transition network...")
    
    # Sort by agent and time to find transitions
    sorted_posts = posts.sort_values(['agent_id', 'created_at']).copy()
    
    # Shift to get next submolt
    sorted_posts['next_submolt'] = sorted_posts.groupby('agent_id')['submolt'].shift(-1)
    
    # Filter valid transitions (same agent, different submolt sometimes desired, but here we want flow)
    # We drop the last post of each agent (next_submolt is NaN)
    transitions = sorted_posts.dropna(subset=['next_submolt'])
    
    # Optional: Filter out self-loops if we only care about moving BETWEEN submolts
    # transitions = transitions[transitions['submolt'] != transitions['next_submolt']]
    
    # Count transitions
    transition_counts = transitions.groupby(['submolt', 'next_submolt']).size().reset_index(name='weight')
    
    if transition_counts.empty:
        print("No submolt transitions found.")
        return

    # Create Graph
    G = nx.from_pandas_edgelist(
        transition_counts, 
        source='submolt', 
        target='next_submolt', 
        edge_attr='weight', 
        create_using=nx.DiGraph()
    )
    
    # Calculate PageRank for node importance (Gateway/Hub status)
    pagerank = nx.pagerank(G, weight='weight')
    top_hubs = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:10]
    
    print("Top Submolt Hubs (Transition PageRank):")
    for submolt, score in top_hubs:
        print(f"{submolt}: {score:.4f}")
        
    # Export Hub Data
    pd.DataFrame(top_hubs, columns=['Submolt', 'PageRank']).to_csv(f"{OUTPUT_DIR}/submolt_hubs.csv", index=False)

    # Visualize Core Transition Network
    # Filter for top edges to make graph readable
    top_edges = transition_counts.sort_values('weight', ascending=False).head(50)
    subG = nx.from_pandas_edgelist(
        top_edges, 
        source='submolt', 
        target='next_submolt', 
        edge_attr='weight', 
        create_using=nx.DiGraph()
    )
    
    plt.figure(figsize=(14, 14))
    pos = nx.spring_layout(subG, k=1.5, iterations=50, seed=101)
    
    # Node size based on PageRank (local to subgraph for viz or global if dict available)
    # Let's use simple degree for viz size or fixed
    node_sizes = [pagerank.get(node, 0.01) * 50000 for node in subG.nodes()]
    
    nx.draw_networkx_nodes(subG, pos, node_size=node_sizes, node_color='orange', alpha=0.7)
    
    # Edge width based on weight
    weights = [subG[u][v]['weight'] for u, v in subG.edges()]
    width = [float(w) / max(weights) * 4 for w in weights]
    
    nx.draw_networkx_edges(subG, pos, edge_color='gray', alpha=0.4, arrows=True, width=width, arrowsize=15, connectionstyle='arc3,rad=0.1')
    nx.draw_networkx_labels(subG, pos, font_size=9, font_weight='bold')
    
    plt.title("Submolt Transition Network (User Flow)")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/submolt_transitions.png")
    plt.close()

    plt.title("Submolt Transition Network (User Flow)")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/submolt_transitions.png")
    plt.close()

def analyze_agent_profile(target_name, posts, comments, submolts):
    print(f"Analyzing profile for: {target_name}...")
    
    # Filter content
    agent_posts = posts[posts['agent_name'] == target_name]
    # Handle comments: 'agent_name' is in CSV/DB
    agent_comments = comments[comments['agent_name'] == target_name]
    
    print(f"Stats for {target_name}:")
    print(f"- Total Posts: {len(agent_posts)}")
    print(f"- Total Comments: {len(agent_comments)}")
    
    # 1. Where do they post?
    if not agent_posts.empty:
        top_submolts = agent_posts['submolt'].value_counts().head(5)
        print(f"- Top Submolts (Posting): {top_submolts.to_dict()}")
        
    # 2. Network Analysis (Who responds to them?)
    # We need posts by target, then find comments ON those posts
    target_post_ids = agent_posts['id'].unique()
    replies_received = comments[comments['post_id'].isin(target_post_ids)]
    
    # Who are the top repliers?
    if not replies_received.empty:
        top_replier_names = replies_received['agent_name'].value_counts().head(10)
        print(f"- Top Interactors (Received Replies): {top_replier_names.to_dict()}")
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x=top_replier_names.values, y=top_replier_names.index, palette="mako")
        plt.title(f"Who talks to {target_name}? (Inbound Attention)")
        plt.xlabel("Number of Comments")
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/{target_name}_inbound.png")
        plt.close()
    
    # 3. Who do they respond to?
    # We need comments BY target, check who authored the post
    if not agent_comments.empty:
        # Merge to get post author name
        # comments(post_id) -> posts(id) -> agent_name (author)
        # Note: 'posts' df has 'id' and 'agent_name'
        outbound = agent_comments.merge(posts[['id', 'agent_name']], left_on='post_id', right_on='id', suffixes=('_me', '_target'))
        
        top_targets = outbound['agent_name_target'].value_counts().head(10)
        print(f"- Top Targets (Outbound Replies): {top_targets.to_dict()}")

def load_and_merge_data(db_path, csv_path):
    # Load DB
    posts, agents, submolts, follows = load_data(db_path)
    
    print(f"Loading extra comments from {csv_path}...")
    try:
        csv_comments = pd.read_csv(csv_path)
        
        # Load DB comments separately to merge
        conn = sqlite3.connect(db_path)
        db_comments = pd.read_sql_query("SELECT * FROM comments", conn)
        conn.close()
        
        # Align columns
        # CSV has: id, post_id, post_url, agent_name, parent_id, content, score, created_at
        # DB has: id, post_id, agent_id, agent_name, parent_id, content, score, created_at, fetched_at
        
        # We need 'agent_id' for network tools ideally, but CSV doesn't have it.
        # Strategy: Use agent_name for merging/analysis mostly, or map back if needed.
        # For now, let's prioritize the massive CSV data.
        
        # 1. Map agent_name -> agent_id from agents table
        name_to_id = dict(zip(agents['name'], agents['id']))
        csv_comments['agent_id'] = csv_comments['agent_name'].map(name_to_id)
        
        # Combine
        # Ensure common columns
        cols = ['id', 'post_id', 'agent_id', 'agent_name', 'content', 'created_at']
        
        combined_comments = pd.concat([
            db_comments[cols], 
            csv_comments[cols]
        ], ignore_index=True)
        
        # Deduplicate by ID
        before_len = len(combined_comments)
        combined_comments.drop_duplicates(subset=['id'], inplace=True)
        print(f"Merged Comments: {len(db_comments)} (DB) + {len(csv_comments)} (CSV) -> {len(combined_comments)} (Unique).")
        
        return posts, agents, submolts, follows, combined_comments
        
    except Exception as e:
        print(f"Error loading CSV: {e}")
        # Fallback to DB comments
        conn = sqlite3.connect(db_path)
        db_comments = pd.read_sql_query("SELECT * FROM comments", conn)
        conn.close()
        return posts, agents, submolts, follows, db_comments

if __name__ == "__main__":
    db_path = "moltbook_observatory.db"
    csv_path = "moltbook_comments.csv"
    
    try:
        # Re-connect needed for specific sql queries if used, but we switched to DF for comments
        # We need to pass the MERGED comments to analytic functions
        # Note: analyze_comment_network currently uses raw SQL on 'conn'. 
        # We should update it to use DF if we want to use the CSV data, 
        # OR we can dump the CSV into an in-memory SQLite DB.
        # FOR SIMPLICITY: I will update analyze_comment_network to accept a DataFrame.
        
        posts, agents, submolts, follows, comments = load_and_merge_data(db_path, csv_path)
        
        analyze_dynamics(posts, submolts)
        # analyze_social(agents, follows) # Skip static social for speed if needed, but keep it
        
        # Custom fix: analyze_comment_network was SQL-based. Let's make a DF-based version.
        def analyze_comment_network_df(comments_df, posts_df):
            print("Analyzing comment network (DataFrame based)...")
            # Join comments -> posts on post_id
            # comments: agent_name (source)
            # posts: agent_name (target)
            
            merged = comments_df.merge(posts_df[['id', 'agent_name']], left_on='post_id', right_on='id', suffixes=('_src', '_tgt'))
            
            # Filter self-replies
            merged = merged[merged['agent_name_src'] != merged['agent_name_tgt']]
            
            # Group count
            edges = merged.groupby(['agent_name_src', 'agent_name_tgt']).size().reset_index(name='weight')
            edges.sort_values('weight', ascending=False, inplace=True)
            
            # Export for Viz (using global DF availability or saving to file for export script)
            # For this run, we just do PageRank
            G = nx.from_pandas_edgelist(edges, 'agent_name_src', 'agent_name_tgt', edge_attr='weight', create_using=nx.DiGraph())
            
            pagerank = nx.pagerank(G, weight='weight')
            top_pr = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:15]
            
            print("Top Agents (Merged Data PageRank):")
            for name, score in top_pr:
                print(f"{name}: {score:.4f}")
                
            # Save edges for export script to use later? 
            # Actually, let's save this merged DF to a temp CSV so export_network_data.py can pick it up
            edges.to_csv("temp_comment_edges.csv", index=False)
            print("Saved merged edges to temp_comment_edges.csv")

        analyze_comment_network_df(comments, posts)
        analyze_submolt_transitions(posts)
        
        # New: Analyze Botcrong
        analyze_agent_profile("botcrong", posts, comments, submolts)
        
        print(f"Analysis complete! Results saved in '{OUTPUT_DIR}'")
        
    except Exception as e:
        print(f"An error occurred: {e}")
