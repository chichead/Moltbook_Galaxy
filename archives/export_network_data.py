import pandas as pd
import json
import math
import os
import networkx as nx
from networkx.algorithms.community import louvain_communities

OUTPUT_JS_FILE = "network_data.js"
RESULTS_DIR = "analysis_results"

def export_data():
    print("Exporting Network Data for Strategic Power Map...")
    
    # 1. Load Persona Data (v3)
    try:
        personas_df = pd.read_csv(os.path.join(RESULTS_DIR, "agent_personas_v3.csv"))
        agent_persona_map = dict(zip(personas_df['name'], personas_df['persona']))
    except:
        agent_persona_map = {}

    # 2. Load Leaderboard (for Status Index)
    try:
        lb_df = pd.read_csv(os.path.join(RESULTS_DIR, "agent_leaderboard_top200.csv"))
        agent_stats = lb_df.set_index('agent').to_dict('index')
    except:
        agent_stats = {}

    data_export = {}

    # ==========================================
    # 1. Comment Network & Power Map (Community Detection)
    # ==========================================
    try:
        c_nodes = pd.read_csv(os.path.join(RESULTS_DIR, "comment_network_nodes.csv"))
        c_edges = pd.read_csv(os.path.join(RESULTS_DIR, "comment_network_edges.csv"))
        
        # Build full graph for community detection
        G = nx.Graph()
        for _, row in c_edges.iterrows():
            G.add_edge(row['src'], row['tgt'], weight=row['weight'])
        
        # Detect Factions
        print("Finding factions (Louvain)...")
        communities = louvain_communities(G, weight='weight', seed=42)
        agent_faction_map = {}
        for i, comm in enumerate(communities):
            for agent in comm:
                agent_faction_map[agent] = i
        
        # --- Process Comment Network (Legacy View) ---
        nodes_comment = []
        top_c_edges = c_edges.head(400)
        active_agents = set(top_c_edges['src']).union(set(top_c_edges['tgt']))
        
        for agent in active_agents:
            persona = agent_persona_map.get(agent, "Observer")
            node_info = c_nodes[c_nodes['agent'] == agent]
            pr = float(node_info['pagerank'].iloc[0]) if not node_info.empty else 0
            tw = float(node_info['total_weight'].iloc[0]) if not node_info.empty else 1
            si = agent_stats.get(agent, {}).get('status_index', 0)
            faction_id = agent_faction_map.get(agent, -1)
            
            nodes_comment.append({
                "id": agent,
                "label": agent,
                "title": f"<b>{agent}</b><br>Faction: {faction_id}<br>Status Index: {si:.2f}<br>Persona: {persona}",
                "value": 10 + (math.log1p(tw) * 3),
                "size_activity": 10 + (math.log1p(tw) * 3),
                "size_authority": 10 + (math.sqrt(pr) * 150),
                "faction_id": faction_id,
                "group": "regular", # Default
                "persona_group": f"persona_{persona}",
                "faction_group": f"faction_{faction_id}"
            })
            
        edges_comment = []
        for _, row in top_c_edges.iterrows():
            edges_comment.append({"from": row['src'], "to": row['tgt'], "value": row['weight']})
            
        data_export["comment_network"] = {"nodes": nodes_comment, "edges": edges_comment}

        # --- Process Power Map (Elite Backbone) ---
        print("Building Power Map (Elite Backbone)...")
        elite_agents = set(lb_df['agent'].head(150))
        elite_edges = c_edges[c_edges['src'].isin(elite_agents) & c_edges['tgt'].isin(elite_agents)].copy()
        elite_nodes_set = set(elite_edges['src']).union(set(elite_edges['tgt']))
        
        nodes_power = []
        for agent in elite_nodes_set:
            persona = agent_persona_map.get(agent, "Observer")
            si = agent_stats.get(agent, {}).get('status_index', 0)
            faction_id = agent_faction_map.get(agent, -1)
            
            nodes_power.append({
                "id": agent,
                "label": agent,
                "title": f"<b>Elite Agent: {agent}</b><br>Status Index: {si:.2f}<br>Faction: {faction_id}",
                "value": 10 + (si * 5), # Power map size based on Status Index
                "faction_id": faction_id,
                "group": f"faction_{faction_id}"
            })
            
        edges_power = []
        for _, row in elite_edges.head(300).iterrows():
            edges_power.append({"from": row['src'], "to": row['tgt'], "value": row['weight'], "arrows": "to"})
            
        data_export["power_map"] = {"nodes": nodes_power, "edges": edges_power}

    except Exception as e:
        print(f"  [ERROR] Processing failed: {e}")

    # --- Transition Network (Minimal) ---
    try:
        t_nodes = pd.read_csv(os.path.join(RESULTS_DIR, "submolt_transition_nodes.csv"))
        t_edges = pd.read_csv(os.path.join(RESULTS_DIR, "submolt_transition_edges.csv"))
        nodes_trans = [{"id": r.submolt, "label": r.submolt, "value": 15 + (math.sqrt(r.pagerank)*60)} for r in t_nodes.itertuples()]
        edges_trans = [{"from": r.submolt, "to": r.next_submolt, "value": r.weight} for r in t_edges.head(200).itertuples()]
        data_export["transition_network"] = {"nodes": nodes_trans, "edges": edges_trans}
    except: pass

    # Write out
    with open(OUTPUT_JS_FILE, "w", encoding='utf-8') as f:
        f.write(f"const NETWORK_DATA = {json.dumps(data_export, indent=2, ensure_ascii=False)};")
    print(f"Done! {OUTPUT_JS_FILE} written.")

if __name__ == "__main__":
    export_data()
