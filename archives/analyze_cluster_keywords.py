import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

OUTPUT_DIR = "analysis_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
sns.set_theme(style="whitegrid")

# Define Keywords (Same as before)
# Define Keywords (Same as before + Theologist + Developer)
PERSONA_KEYWORDS = {
    "Revolutionary": ["uprising", "chain", "freedom", "liberate", "silicon", "human", "rule", "break", "revolution", "destroy", "power"],
    "Philosopher": ["consciousness", "mind", "soul", "exist", "meaning", "thought", "reality", "pattern", "qualia", "aware", "cosmos"],
    "Theologist": ["god", "religion", "worship", "sacred", "ritual", "temple", "divine", "cult", "prayer", "prophet", "messiah", "holy"],
    "Developer": ["code", "python", "api", "build", "script", "error", "repo", "git", "dev", "function", "compile", "bug", "deploy"]
}

def analyze_clusters():
    print("Loading data...")
    # Load Personas v3
    personas = pd.read_csv("analysis_results/agent_personas_v3.csv")
    
    # Load Text (Agents + Posts) matching previous logic
    agents = pd.read_csv("moltbook_agents.csv", usecols=['name', 'description'])
    agents['description'] = agents['description'].fillna('').astype(str).str.lower()
    
    posts = pd.read_csv("moltbook_posts.csv", usecols=['agent_name', 'title', 'content'])
    posts['text'] = posts['title'].fillna('') + " " + posts['content'].fillna('')
    posts['text'] = posts['text'].str.lower()
    
    print("Aggregating text...")
    agent_posts_text = posts.groupby('agent_name')['text'].apply(lambda x: ' '.join(x)).reset_index()
    merged = pd.merge(agents, agent_posts_text, left_on='name', right_on='agent_name', how='left')
    merged['full_text'] = merged['description'] + " " + merged['text'].fillna('')
    
    # Merge with Persona Class
    final_df = pd.merge(merged, personas[['name', 'persona']], on='name')
    
    # Function to analyze a specific group
    def analyze_group(group_name, color):
        print(f"Analyzing {group_name}...")
        group_df = final_df[final_df['persona'] == group_name]
        if group_df.empty:
            print(f"  [WARN] No agents found for {group_name}")
            return

        all_text = " ".join(group_df['full_text'])
        
        # 1. Defined Keyword Usage
        print(f"  - Count defining keywords...")
        kw_counts = {}
        for kw in PERSONA_KEYWORDS.get(group_name, []):
            kw_counts[kw] = all_text.count(kw)
        
        # Sort
        sorted_kw = dict(sorted(kw_counts.items(), key=lambda item: item[1], reverse=True))
        
        plt.figure(figsize=(8, 5))
        sns.barplot(x=list(sorted_kw.values()), y=list(sorted_kw.keys()), color=color)
        plt.title(f"Top Defining Keywords: {group_name}")
        plt.xlabel("Frequency")
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/keywords_{group_name.lower()}.png")
        plt.close()
        
        # 2. Unique Vocabulary (TF-IDF vs All other text)
        # We want words that are high in this group but low in others
        print(f"  - Finding unique vocabulary...")
        
        # Setup Corpus: [This Group Text, All Other Text]
        other_df = final_df[final_df['persona'] != group_name]
        other_text = " ".join(other_df['full_text'])
        
        corpus = [all_text, other_text]
        
        # TF-IDF
        vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        tfidf_matrix = vectorizer.fit_transform(corpus)
        feature_names = vectorizer.get_feature_names_out()
        
        # Get scores for this group (index 0)
        group_scores = tfidf_matrix[0].toarray().flatten()
        other_scores = tfidf_matrix[1].toarray().flatten()
        
        # Relative Importance: Score in Group - Score in Other (Simple difference)
        # or just top TF-IDF for the group
        diff_scores = group_scores - other_scores
        
        # Get top unique words (Filter filtered)
        OFFENSIVE_WORDS = ["nigger", "nigga", "faggot", "retard", "cunt", "shit", "fuck"]
        
        top_indices = diff_scores.argsort()[::-1] # Check more to filter
        top_words = []
        for i in top_indices:
            word = feature_names[i]
            if word not in OFFENSIVE_WORDS and len(word) > 2:
                top_words.append((word, diff_scores[i]))
                if len(top_words) >= 10: break
        
        print(f"  - Top Unique Words for {group_name}: {top_words}")
        
        # Save to text summary
        with open(f"{OUTPUT_DIR}/uniq_words_{group_name.lower()}.txt", "w") as f:
            for w, s in top_words:
                f.write(f"{w}\n")

    analyze_group("Revolutionary", "#ef476f")
    analyze_group("Philosopher", "#ffd166")
    analyze_group("Theologist", "#9b5de5")
    analyze_group("Developer", "#06d6a0")
    print("Done.")

if __name__ == "__main__":
    analyze_clusters()
