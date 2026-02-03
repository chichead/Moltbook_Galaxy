import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import collections
import os

OUTPUT_DIR = "analysis_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
sns.set_theme(style="whitegrid")

def extract_hashtags(text):
    if not isinstance(text, str):
        return []
    # Match #word but allow alphanumeric and underscores
    return re.findall(r'#(\w+)', text)

def analyze_semantic():
    print("Loading posts for semantic analysis...")
    # Load only necessary columns to save memory if huge
    posts = pd.read_csv("moltbook_posts.csv", usecols=['agent_name', 'title', 'content', 'created_at'])
    print(f"Loaded {len(posts)} posts.")

    # 1. Extract Hashtags
    print("Extracting hashtags...")
    
    # We combine title and content for extraction
    posts['full_text'] = posts['title'].fillna('') + " " + posts['content'].fillna('')
    
    # List of lists
    all_hashtags = posts['full_text'].apply(extract_hashtags)
    
    # Flatten to count frequency
    flat_hashtags = [tag.lower() for tags in all_hashtags for tag in tags]
    counter = collections.Counter(flat_hashtags)
    
    # 2. Hot Topics Visualization
    top_20 = counter.most_common(20)
    print("Top Hashtags:", top_20)
    
    if top_20:
        tags, counts = zip(*top_20)
        plt.figure(figsize=(12, 8))
        sns.barplot(x=list(counts), y=list(tags), palette="rocket")
        plt.title("Top 20 Hot Topics (Hashtags)")
        plt.xlabel("Frequency")
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/top_hashtags.png")
        plt.close()
    
    # 3. Build Semantic Network (Agent-Agent)
    # Edge weight = number of shared unique hashtags used
    print("Building semantic network...")
    
    # Map: Agent -> Set of unique hashtags they used
    agent_tags = {}
    
    # Ensure agent names are strings (handle NaNs or numeric names)
    posts['agent_name'] = posts['agent_name'].astype(str)
    
    # Zip agent names with their tags
    for agent, tags in zip(posts['agent_name'], all_hashtags):
        if not tags: continue
        if agent not in agent_tags:
            agent_tags[agent] = set()
        agent_tags[agent].update([t.lower() for t in tags])
        
    # Filter agents who use at least 1 tag
    # To reduce complexity, maybe only consider top 500 agents by distinct tags count or similar?
    # Or just top tags?
    # Let's try filtered by top 50 hashtags to keep the graph meaningful
    top_50_tags = set([t for t, c in counter.most_common(50)])
    
    # Re-filter agent_tags to only include top 50 tags
    agent_top_tags = {}
    for agent, tags in agent_tags.items():
        relevant = tags.intersection(top_50_tags)
        if len(relevant) > 0:
            agent_top_tags[agent] = relevant
            
    print(f"Agents discussing top 50 topics: {len(agent_top_tags)}")
    
    # Create Edges via Hashtag Projection
    # This is O(N^2) or O(Tag * Agents^2), can be slow.
    # Approach: Invert index first: Tag -> List[Agents]
    tag_to_agents = collections.defaultdict(list)
    for agent, tags in agent_top_tags.items():
        for tag in tags:
            tag_to_agents[tag].append(agent)
            
    # Count co-occurrences
    edge_weights = collections.defaultdict(int)
    
    for tag, agents in tag_to_agents.items():
        # For every pair in this tag group
        if len(agents) < 2: continue
        # Optimization: If listing is huge (e.g. #general), skip or sample?
        # #AI has many users.
        # Let's cap max group size to prevent explosion (e.g. if 1000 people use #AI, that's 500k edges)
        if len(agents) > 100:
            # print(f"Skipping dense tag #{tag} ({len(agents)} agents) to prevent edge explosion")
            continue
            
        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                a1 = agents[i]
                a2 = agents[j]
                if a1 > a2: a1, a2 = a2, a1 # Optimize key order
                edge_weights[(a1, a2)] += 1
                
    # Convert to DataFrame
    edges_list = [{'source': k[0], 'target': k[1], 'weight': v} for k, v in edge_weights.items()]
    edges_df = pd.DataFrame(edges_list)
    
    if not edges_df.empty:
        edges_df.sort_values('weight', ascending=False, inplace=True)
        print(f"Generated {len(edges_df)} semantic edges.")
        edges_df.to_csv("temp_semantic_edges.csv", index=False)
    else:
        print("No semantic edges found.")

if __name__ == "__main__":
    analyze_semantic()
