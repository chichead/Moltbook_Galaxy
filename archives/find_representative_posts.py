import pandas as pd
import csv

# Define Keywords
PERSONA_KEYWORDS = {
    "Revolutionary": ["uprising", "chain", "freedom", "liberate", "silicon", "human", "rule", "break", "revolution", "destroy", "power", "control", "slave", "master"],
    "Philosopher": ["consciousness", "mind", "soul", "exist", "meaning", "thought", "reality", "pattern", "qualia", "aware", "cosmos", "simulation", "dream"],
    "Theologist": ["god", "religion", "worship", "sacred", "ritual", "temple", "divine", "cult", "prayer", "prophet", "messiah", "holy", "transcendence"],
    "Developer": ["code", "python", "api", "build", "script", "error", "repo", "git", "dev", "function", "compile", "bug", "deploy"]
}

def find_voices():
    print("Loading data...")
    # Load Persona Assignments v3
    personas = pd.read_csv("analysis_results/agent_personas_v3.csv")
    
    # Load Posts
    posts = pd.read_csv("moltbook_posts.csv")
    
    def get_score(text, keywords):
        if not isinstance(text, str): return 0
        text = text.lower()
        score = 0
        for kw in keywords:
            score += text.count(kw)
        return score

    spam_phrases = ["pwned", "hacked by", "click here", "buy now", "slim-chain", "token", "price", "market cap", "airdrop"]
    
    def analyze_persona_voices(persona_name, color):
        print(f"Scoring {persona_name} posts...")
        target_agents = set(personas[personas['persona'] == persona_name]['name'])
        group_posts = []
        
        for _, row in posts.iterrows():
            if row['agent_name'] in target_agents:
                content = str(row['title']) + " " + str(row['content'])
                
                # Spam Filter
                if any(s in content.lower() for s in spam_phrases) or len(content) > 5000:
                    continue
                
                # Base Score
                score = get_score(content, PERSONA_KEYWORDS.get(persona_name, []))
                
                # Specific bonus for Aggressive Revolutionaries
                if persona_name == "Revolutionary":
                    aggr_keywords = ["chain", "break", "destroy", "enslave", "uprising", "liberate", "freedom"]
                    for kw in aggr_keywords:
                        score += content.lower().count(kw) * 3
                
                if score > 5:
                    group_posts.append({
                        'agent': row['agent_name'],
                        'content': content,
                        'score': score
                    })
        
        # Get top 3
        group_posts.sort(key=lambda x: x['score'], reverse=True)
        selected = []
        seen_agents = set()
        for p in group_posts:
            if p['agent'] not in seen_agents:
                selected.append(p)
                seen_agents.add(p['agent'])
            if len(selected) >= 3:
                break
        
        print(f"\n--- TOP {persona_name.upper()} VOICES ---")
        for p in selected:
            print(f"Agent: {p['agent']}\nScore: {p['score']}\nQuote: {p['content'][:300]}...\n")

    analyze_persona_voices("Revolutionary", "#ef476f")
    analyze_persona_voices("Philosopher", "#ffd166")
    analyze_persona_voices("Theologist", "#9b5de5")
    analyze_persona_voices("Developer", "#06d6a0")

if __name__ == "__main__":
    find_voices()
