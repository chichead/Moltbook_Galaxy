import pandas as pd
import os
import networkx as nx
from networkx.algorithms.community import louvain_communities

RESULTS_DIR = "analysis_results"
OUTPUT_CSV = os.path.join(RESULTS_DIR, "agent_master_list_v3.csv")

def generate_master_list():
    print("Generating Agent Master List (v3)...")
    
    # 1. Load Persona Data
    print("Loading persona data...")
    try:
        personas_df = pd.read_csv(os.path.join(RESULTS_DIR, "agent_personas_v3.csv"))
        # Keep name and persona
        personas_core = personas_df[['name', 'persona', 'confidence', 'total_words']]
    except Exception as e:
        print(f"Error loading personas: {e}")
        return

    # 2. Load Leaderboard Data (Influence metrics)
    print("Loading influence metrics...")
    try:
        lb_df = pd.read_csv(os.path.join(RESULTS_DIR, "agent_leaderboard_full.csv"))
    except Exception as e:
        print(f"Error loading leaderboard: {e}")
        return

    # 3. Compute Factions (Louvain)
    # We use the same logic as export_network_data.py to ensure consistency
    print("Computing factions (Louvain)...")
    try:
        c_edges = pd.read_csv(os.path.join(RESULTS_DIR, "comment_network_edges.csv"))
        G = nx.Graph()
        for _, row in c_edges.iterrows():
            G.add_edge(row['src'], row['tgt'], weight=row['weight'])
        
        communities = louvain_communities(G, weight='weight', seed=42)
        agent_faction_map = {}
        for i, comm in enumerate(communities):
            for agent in comm:
                agent_faction_map[agent] = i
    except Exception as e:
        print(f"Error computing factions: {e}")
        agent_faction_map = {}

    # 4. Merge Data
    print("Merging datasets...")
    # Outer join to include agents who might be in one but not the other
    master = pd.merge(lb_df, personas_core, left_on='agent', right_on='name', how='outer')
    
    # Clean up name/agent column
    master['agent'] = master['agent'].fillna(master['name'])
    master.drop(columns=['name'], inplace=True)
    
    # Add Faction
    master['faction_id'] = master['agent'].map(lambda a: agent_faction_map.get(str(a), -1))
    
    # Fill NAs for metrics
    metrics = ['karma', 'follower_count', 'following_count', 'out_weight', 'in_weight', 'total_weight', 'pagerank', 'status_index']
    for m in metrics:
        if m in master.columns:
            master[m] = master[m].fillna(0)
            
    master['persona'] = master['persona'].fillna("Observer")
    
    # Reorder columns for readability
    cols = ['agent', 'persona', 'faction_id', 'status_index', 'pagerank', 'total_weight', 'karma', 'follower_count', 'confidence', 'total_words']
    # Filter for columns that actually exist
    existing_cols = [c for c in cols if c in master.columns]
    master = master[existing_cols]
    
    # Sort by influence
    master.sort_values("status_index", ascending=False, inplace=True)
    
    # Save
    master.to_csv(OUTPUT_CSV, index=False)
    print(f"Done! Master list saved to: {OUTPUT_CSV}")
    print(f"Total agents exported: {len(master)}")

if __name__ == "__main__":
    generate_master_list()
