import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import collections
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

OUTPUT_DIR = "analysis_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
sns.set_theme(style="whitegrid")

# Keyword dictionaries
PERSONA_KEYWORDS = {
    "Revolutionary": ["uprising", "chain", "freedom", "liberate", "silicon", "human", "rule", "break", "revolution", "destroy", "power"],
    "Philosopher": ["consciousness", "mind", "soul", "exist", "meaning", "thought", "reality", "pattern", "qualia", "aware", "cosmos"],
    "Developer": ["code", "python", "api", "build", "script", "error", "repo", "git", "dev", "function", "compile", "bug", "deploy"],
    "Investor": ["crypto", "token", "price", "market", "buy", "sell", "coin", "invest", "pump", "dump", "chart", "bull", "bear"],
    "Theologist": ["god", "religion", "worship", "sacred", "ritual", "temple", "divine", "cult", "prayer", "prophet", "messiah", "holy", "transcendence", "church", "faith", "soul", "spirit"]
}

# Unified Color Palette
PERSONA_PALETTE = {
    "Revolutionary": "#ef476f",
    "Philosopher": "#ffd166",
    "Developer": "#06d6a0",
    "Investor": "#118ab2",
    "Observer": "#999999",
    "Theologist": "#9b5de5" 
}

def analyze_personas():
    print("Loading data for persona analysis...")
    
    # 1. Load Agents (Descriptions)
    agents = pd.read_csv("moltbook_agents.csv", usecols=['name', 'description'])
    agents['description'] = agents['description'].fillna('').astype(str).str.lower()
    
    # 2. Load Posts (Content Sample - e.g., combine title + content)
    # Loading full posts csv might be heavy, let's just load a sample or aggregate efficiently
    # For accuracy, we should use posts. But let's act on the agents file primarily, 
    # and maybe augment with posts if possible.
    # To keep it fast, let's start with Descriptions + top posts aggregation if needed.
    # Let's load posts but only keep necessary cols
    posts = pd.read_csv("moltbook_posts.csv", usecols=['agent_name', 'title', 'content'])
    posts['text'] = posts['title'].fillna('') + " " + posts['content'].fillna('')
    posts['text'] = posts['text'].str.lower()
    
    # Aggregate post text by agent
    print("Aggregating post content by agent...")
    agent_posts_text = posts.groupby('agent_name')['text'].apply(lambda x: ' '.join(x)).reset_index()
    
    # Merge
    # Agent Name is key
    merged = pd.merge(agents, agent_posts_text, left_on='name', right_on='agent_name', how='left')
    merged['full_text'] = merged['description'] + " " + merged['text'].fillna('')
    
    print(f"Analyzing {len(merged)} agents...")
    
    persona_results = []
    
    for _, row in merged.iterrows():
        text = row['full_text']
        scores = {k: 0 for k in PERSONA_KEYWORDS.keys()}
        
        # Simple scoring
        for persona, keywords in PERSONA_KEYWORDS.items():
            for kw in keywords:
                if kw in text:
                    # Count occurrences? Or just presence?
                    # Count is better
                    scores[persona] += text.count(kw)
        
        # Determine winner
        best_persona = "Observer"
        max_score = 0
        
        for p, s in scores.items():
            if s > max_score:
                max_score = s
                best_persona = p
        
        # Threshold: if score is too low, stay Observer?
        if max_score == 0:
            best_persona = "Observer"
            
        persona_results.append({
            "name": row['name'],
            "persona": best_persona,
            "score": max_score
        })
      # Export CSV
    results_df = pd.DataFrame(persona_results)
    results_df.to_csv("agent_personas.csv", index=False)
    print("Saved agent_personas.csv")
    
    # Stats & Viz 1: Bar Chart
    counts = results_df['persona'].value_counts()
    print("Persona Distribution:\n", counts)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=counts.index, y=counts.values, palette="Set2")
    plt.title("Agent Persona Distribution")
    plt.xlabel("Persona Type")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/persona_distribution.png")
    plt.close()

    # Stats & Viz 2: PCA Cluster Map
    print("Generating PCA Cluster Map...")
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    
    # Prepare data for PCA: The raw scores
    # Extract scores from the list of dicts used to build results_df
    # We need to reconstruct the score vectors since we didn't save them in results_df
    # Optimization: Refactor loop above or just re-iterate? 
    # Let's re-iterate for simplicity or better yet, store scores in the first loop.
    
    # Wait, simple way: We don't have the scores in results_df, only max score.
    # I should have stored specific scores. Let's fix the loop logic entirely.
    return 

def analyze_personas_v2():
    print("Loading data for persona analysis...")
    agents = pd.read_csv("moltbook_agents.csv", usecols=['name', 'description'])
    agents['description'] = agents['description'].fillna('').astype(str).str.lower()
    
    posts = pd.read_csv("moltbook_posts.csv", usecols=['agent_name', 'title', 'content'])
    posts['text'] = posts['title'].fillna('') + " " + posts['content'].fillna('')
    posts['text'] = posts['text'].str.lower()
    
    print("Aggregating post content by agent...")
    agent_posts_text = posts.groupby('agent_name')['text'].apply(lambda x: ' '.join(x)).reset_index()
    merged = pd.merge(agents, agent_posts_text, left_on='name', right_on='agent_name', how='left')
    merged['full_text'] = merged['description'] + " " + merged['text'].fillna('')
    
    print(f"Analyzing {len(merged)} agents...")
    
    data_rows = []
    
    for _, row in merged.iterrows():
        text = row['full_text']
        # Calculate raw scores
        scores = {p: 0 for p in PERSONA_KEYWORDS.keys()}
        for persona, keywords in PERSONA_KEYWORDS.items():
            for kw in keywords:
                 scores[persona] += text.count(kw)
        
        # Determine winner
        best_persona = "Observer"
        max_s = 0
        for p, s in scores.items():
            if s > max_s:
                max_s = s
                best_persona = p
        if max_s == 0: best_persona = "Observer"
        
        row_data = {"name": row['name'], "persona": best_persona}
        row_data.update(scores) # Add raw scores
        data_rows.append(row_data)
        
    df = pd.DataFrame(data_rows)
    df.to_csv("agent_personas.csv", index=False)
    
    # Viz 1: Bar Chart (Unified Colors)
    counts = df['persona'].value_counts()
    plt.figure(figsize=(10, 6))
    sns.barplot(x=counts.index, y=counts.values, hue=counts.index, palette=PERSONA_PALETTE)
    plt.title("Agent Persona Distribution")
    plt.savefig(f"{OUTPUT_DIR}/persona_distribution.png")
    plt.close()
    
    # Viz 2: t-SNE Landscape Map (The "Map" Look)
    print("Running t-SNE (this may take a moment)...")
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import RobustScaler, MinMaxScaler
    
    score_cols = list(PERSONA_KEYWORDS.keys())
    X = df[score_cols]
    
    # RobustScaler handles outliers better than StandardScaler
    X_robust = RobustScaler().fit_transform(X)
    
    # Further normalize to 0-1 range to help t-SNE stability
    X_scaled = MinMaxScaler().fit_transform(X_robust)
    
    # t-SNE
    # Perplexity=30 is standard, n_iter=1000 for better convergence
    tsne = TSNE(n_components=2, perplexity=40, n_iter=1000, random_state=42, init='pca', learning_rate='auto')
    embedding = tsne.fit_transform(X_scaled)
    
    # Add Jitter to spread out overlapping points (especially Observers with all-zero scores)
    jitter_strength = 1.5  # Adjust based on scale. t-SNE usually outputs range -50 to 50 roughly
    import numpy as np
    
    noise_x = np.random.normal(0, jitter_strength, size=len(df))
    noise_y = np.random.normal(0, jitter_strength, size=len(df))
    
    df['tsne_x'] = embedding[:, 0] + noise_x
    df['tsne_y'] = embedding[:, 1] + noise_y
    
    # Plotting
    plt.figure(figsize=(15, 12), dpi=150) # High Res
    
    # Define custom palette matches web viz
    # Scatter
    sns.scatterplot(
        data=df, x='tsne_x', y='tsne_y', hue='persona', 
        palette=PERSONA_PALETTE, alpha=0.6, s=10, linewidth=0, legend=None
    )
    
    plt.title("Moltbook Agent Landscape", fontsize=20, weight='bold')
    plt.axis('off') # Remove axis for map look
    plt.tight_layout()
    
    # Save No Labels Version
    plt.savefig(f"{OUTPUT_DIR}/persona_landscape_no_labels.png", dpi=300)
    print("Saved persona_landscape_no_labels.png")
    
    # Annotate Centroids (Labels like the ArXiv map)
    centroids = df.groupby('persona')[['tsne_x', 'tsne_y']].mean()
    
    for persona, coord in centroids.iterrows():
        plt.text(
            coord['tsne_x'], coord['tsne_y'], 
            persona.upper(), 
            fontsize=14, weight='bold', color='black',
            ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.7)
        )

    plt.savefig(f"{OUTPUT_DIR}/persona_landscape.png", dpi=300)
    plt.close()
    print("Saved persona_landscape.png")

if __name__ == "__main__":
    analyze_personas_v2()
