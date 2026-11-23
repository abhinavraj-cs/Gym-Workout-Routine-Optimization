import streamlit as st
import heapq
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

# Page Configuration
st.set_page_config(page_title="AI Gym Routine Optimizer", page_icon="⚡", layout="wide")

# High-contrast, clean modern dark UI with header bar fix
st.markdown("""
<style>
    /* Fix Streamlit Header & Top White Bar */
    header[data-testid="stHeader"] {
        background-color: #0b0f19 !important;
    }
    
    /* Dark Theme Base */
    .stApp {
        background-color: #0b0f19;
        color: #f1f5f9;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }

    /* Sidebar Styling & Input Visibility Fix */
    section[data-testid="stSidebar"] {
        background-color: #111827 !important;
        border-right: 1px solid #1f2937;
    }
    section[data-testid="stSidebar"] p, 
    section[data-testid="stSidebar"] label, 
    section[data-testid="stSidebar"] span {
        color: #f3f4f6 !important;
    }
    
    /* Fix Selectbox / Dropdown text readability */
    div[data-baseweb="select"] {
        background-color: #1e293b !important;
        border-radius: 8px !important;
    }
    div[data-baseweb="select"] * {
        color: #ffffff !important;
    }
    
    /* Header Card */
    .header-card {
        background: linear-gradient(135deg, #1e1e38 0%, #111827 100%);
        border: 1px solid #374151;
        border-radius: 14px;
        padding: 22px 28px;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    }
    .header-title {
        font-size: 2.1rem;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 4px;
        letter-spacing: -0.5px;
    }
    .header-desc {
        color: #9ca3af;
        font-size: 0.95rem;
    }

    /* Metric Counters */
    .metric-container {
        display: flex;
        gap: 16px;
        margin-bottom: 24px;
    }
    .metric-tile {
        flex: 1;
        background: #151d30;
        border: 1px solid #2d3748;
        border-radius: 12px;
        padding: 16px 20px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }
    .metric-num {
        font-size: 2rem;
        font-weight: 800;
        color: #38bdf8;
    }
    .metric-lbl {
        font-size: 0.85rem;
        font-weight: 600;
        color: #94a3b8;
        text-transform: uppercase;
        margin-top: 4px;
    }

    /* Step Routine Cards */
    .routine-step {
        background: #161f30;
        border-left: 5px solid #0284c7;
        border-top: 1px solid #243049;
        border-right: 1px solid #243049;
        border-bottom: 1px solid #243049;
        border-radius: 12px;
        padding: 18px 24px;
        margin-bottom: 14px;
        transition: transform 0.15s ease, border-color 0.15s ease;
    }
    .routine-step:hover {
        border-left-color: #38bdf8;
        background: #1a253a;
        transform: translateX(3px);
    }
    .step-head {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 8px;
    }
    .step-name {
        font-size: 1.25rem;
        font-weight: 700;
        color: #ffffff;
    }
    .step-tag {
        background: #1e293b;
        color: #38bdf8;
        border: 1px solid #0284c7;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
    }
    .step-body {
        font-size: 0.92rem;
        color: #cbd5e1;
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        align-items: center;
    }
    .badge-hit {
        background: #064e3b;
        color: #34d399;
        border: 1px solid #059669;
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 0.82rem;
        font-weight: 600;
    }

    /* Custom Toggle Button */
    div.stButton > button {
        background: linear-gradient(135deg, #0284c7 0%, #2563eb 100%) !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 10px 22px !important;
        box-shadow: 0 4px 14px rgba(2, 132, 199, 0.4) !important;
    }
    div.stButton > button:hover {
        background: linear-gradient(135deg, #0ea5e9 0%, #1d4ed8 100%) !important;
        box-shadow: 0 6px 18px rgba(14, 165, 233, 0.5) !important;
    }
</style>
""", unsafe_allow_html=True)

# 1. Station Attributes & Muscle Taxonomy
STATIONS = {
    0: ("Warm-up activities", 5, "Warm-up", ["warmup"]),
    1: ("Skipping rope", 16, "Cardio", ["cardio", "calves"]),
    2: ("Exercise bike", 10, "Cardio", ["cardio", "quads"]),
    3: ("Treadmill", 12, "Cardio", ["cardio", "stamina"]),
    4: ("Step Mill", 14, "Cardio", ["cardio", "glutes", "quads"]),
    5: ("Dumbbell Press / Curl", 9, "Arms/Chest", ["middle_chest", "biceps", "triceps"]),
    6: ("Barbell Bench Press", 10, "Chest", ["middle_chest", "lower_chest"]),
    7: ("Cable-Crossover", 8, "Chest", ["lower_chest", "inner_chest"]),
    8: ("Pulling bars (Pull-ups/Lat)", 10, "Back", ["lats", "upper_back", "biceps"]),
    9: ("Incline Bench Press", 8, "Chest", ["upper_chest", "front_delts"]),
    10: ("Leg Press Machine", 8, "Legs", ["quads", "glutes", "calves"]),
    11: ("Climbing Rope", 5, "Back/Core", ["lats", "forearms", "abs"]),
    12: ("Hammer Strength Press", 4, "Chest/Back", ["upper_chest", "middle_chest", "lats"]),
    13: ("Stretching", 0, "Cooldown", ["cooldown"])
}

BASE_GRAPH = {
    0: {1: 20, 2: 25, 3: 22, 4: 28},
    1: {5: 35, 6: 30},
    2: {7: 28},
    3: {8: 24, 9: 26},
    4: {9: 26},
    5: {10: 25},
    6: {10: 25},
    7: {11: 20},
    8: {11: 20},
    9: {12: 22},
    10: {13: 15},
    11: {13: 15},
    12: {13: 15},
    13: {}
}

INTER_TRANSITIONS = {
    5: {8: 20, 9: 20},
    6: {7: 18, 8: 22},
    7: {9: 20, 10: 22},
    8: {9: 20, 12: 22},
    9: {10: 24, 11: 20},
}

H = {node: data[1] for node, data in STATIONS.items()}

# State-Space Multi-Goal Search
def run_constrained_search(algo_type, selected_sub_muscles):
    start = 0
    goal = 13
    
    available_sub_targets = set()
    for n in range(1, 13):
        for sub in STATIONS[n][3]:
            if sub in selected_sub_muscles:
                available_sub_targets.add(sub)

    init_unvisited = frozenset(available_sub_targets)
    counter = 0
    frontier = []
    
    # Priority based on formal AI search formulas
    if algo_type == "UCS":
        init_prio = 0
    elif algo_type == "Greedy":
        init_prio = H[start]
    else:  # A*
        init_prio = H[start] + (len(init_unvisited) * 15)
    
    heapq.heappush(frontier, (init_prio, counter, start, init_unvisited, [start], 0))
    
    best_cost = {}
    nodes_expanded = 0
    nodes_generated = 1
    best_fallback = None

    while frontier:
        prio, _, curr, unvisited, path, g = heapq.heappop(frontier)

        state_key = (curr, unvisited) if algo_type != "Greedy" else (curr, len(path))
        if state_key in best_cost and best_cost[state_key] <= g:
            continue
        best_cost[state_key] = g

        if curr == goal:
            if len(unvisited) == 0 or algo_type == "Greedy":
                return {
                    "path": path,
                    "total_g": g,
                    "nodes_expanded": nodes_expanded,
                    "nodes_generated": nodes_generated,
                    "covered_targets": list(available_sub_targets - unvisited),
                    "missing_targets": list(unvisited)
                }
            if best_fallback is None or len(unvisited) < len(best_fallback["missing_targets"]) or (len(unvisited) == len(best_fallback["missing_targets"]) and g < best_fallback["total_g"]):
                best_fallback = {
                    "path": path,
                    "total_g": g,
                    "nodes_expanded": nodes_expanded,
                    "nodes_generated": nodes_generated,
                    "covered_targets": list(available_sub_targets - unvisited),
                    "missing_targets": list(unvisited)
                }

        nodes_expanded += 1

        all_neighbors = dict(BASE_GRAPH.get(curr, {}))
        if curr in INTER_TRANSITIONS:
            for k, v in INTER_TRANSITIONS[curr].items():
                if k not in all_neighbors:
                    all_neighbors[k] = v

        for next_node, base_step_cost in all_neighbors.items():
            if next_node in path and next_node != goal:
                continue

            next_subs = set(STATIONS[next_node][3])
            hits = next_subs.intersection(unvisited)
            
            step_cost = base_step_cost
            if hits:
                step_cost = max(5, base_step_cost - (len(hits) * 12))
            elif next_node not in [goal, 0]:
                step_cost = base_step_cost + 10

            new_g = g + step_cost
            new_unvisited = frozenset(unvisited - hits)
            nodes_generated += 1
            counter += 1

            if algo_type == "UCS":
                eval_f = new_g
            elif algo_type == "Greedy":
                eval_f = H[next_node]
            else:  # A*
                remaining_penalty = len(new_unvisited) * 18
                eval_f = new_g + H[next_node] + remaining_penalty

            heapq.heappush(frontier, (eval_f, counter, next_node, new_unvisited, path + [next_node], new_g))

    return best_fallback

# --- SIDEBAR CONTROLS ---
st.sidebar.markdown("## 🎯 Custom Workout Plan")
st.sidebar.caption("Select your targeted muscle groups & sub-heads:")

muscle_hierarchy = {
    "Chest": ["upper_chest", "middle_chest", "lower_chest", "inner_chest"],
    "Back": ["lats", "upper_back", "forearms"],
    "Legs": ["quads", "glutes", "calves"],
    "Cardio & Stamina": ["cardio", "stamina"],
    "Arms & Core": ["biceps", "triceps", "abs"]
}

selected_groups = st.sidebar.multiselect(
    "1. Primary Muscle Groups:",
    options=list(muscle_hierarchy.keys()),
    default=["Chest", "Legs"]
)

chosen_sub_muscles = []
if selected_groups:
    st.sidebar.markdown("---")
    st.sidebar.markdown("##### 2. Specific Sub-Part Selection:")
    for group in selected_groups:
        sub_opts = muscle_hierarchy[group]
        chosen = st.sidebar.multiselect(
            f"{group} Heads/Areas:",
            options=sub_opts,
            default=sub_opts[:1]
        )
        chosen_sub_muscles.extend(chosen)

st.sidebar.markdown("---")
selected_algo = st.sidebar.selectbox(
    "3. Search Algorithm:",
    options=["A* Search", "Uniform Cost Search (UCS)", "Greedy Best-First", "Compare All 3"],
    index=3
)

# Graph Toggle Session State
if "show_graph" not in st.session_state:
    st.session_state.show_graph = False

def toggle_graph():
    st.session_state.show_graph = not st.session_state.show_graph

# --- MAIN CONTENT VIEW ---
st.markdown("""
<div class="header-card">
    <div class="header-title">⚡ Intelligent Gym Routine Optimizer</div>
    <div class="header-desc">Multi-Goal State-Space Search & Heuristic-Driven Gym Path Sequencing</div>
</div>
""", unsafe_allow_html=True)

# Run Algorithm
if selected_algo != "Compare All 3":
    algo_key = "A*" if "A*" in selected_algo else ("UCS" if "UCS" in selected_algo else "Greedy")
    result = run_constrained_search(algo_key, chosen_sub_muscles)
    active_path = result["path"]

    # Metrics Display
    st.markdown(f"""
    <div class="metric-container">
        <div class="metric-tile">
            <div class="metric-num">{result['total_g']} min</div>
            <div class="metric-lbl">Total Session Active Time</div>
        </div>
        <div class="metric-tile">
            <div class="metric-num">{len(result['path'])}</div>
            <div class="metric-lbl">Total Routine Steps</div>
        </div>
        <div class="metric-tile">
            <div class="metric-num">{result['nodes_expanded']}</div>
            <div class="metric-lbl">Nodes Expanded</div>
        </div>
        <div class="metric-tile">
            <div class="metric-num">{result['nodes_generated']}</div>
            <div class="metric-lbl">Nodes Generated</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Top Bar: Routine Header + Toggle Button
    top_col, btn_col = st.columns([3, 1])
    with top_col:
        st.subheader("📋 Tailored Step-by-Step Exercise Routine")
    with btn_col:
        btn_label = "✖ Hide Graph View" if st.session_state.show_graph else "📊 View State-Space Graph"
        st.button(btn_label, on_click=toggle_graph, use_container_width=True)

    # Render Exercise Routine cleanly
    for idx, node in enumerate(result["path"], 1):
        name, h_val, cat, subs = STATIONS[node]
        hit_matches = set(subs).intersection(chosen_sub_muscles)
        hit_html = "".join([f'<span class="badge-hit">🎯 {m.replace("_", " ").title()}</span>' for m in hit_matches])
        status_lbl = "Session Warmup" if node == 0 else ("Final Stretching & Cool-down" if node == 13 else "Active Station")

        step_card_html = f"""<div class="routine-step">
<div class="step-head">
<span class="step-name">Step {idx}: {name}</span>
<span class="step-tag">{cat} • Node {node}</span>
</div>
<div class="step-body">
<span>Phase: <b>{status_lbl}</b></span>
<span>• Recovery Heuristic: <b>{h_val} mins</b></span>
{hit_html}
</div>
</div>"""
        st.markdown(step_card_html, unsafe_allow_html=True)

    if result.get("missing_targets"):
        st.warning(f"⚠️ Note: In a single gym session, {', '.join(result['missing_targets'])} could not be covered due to station transition safety bounds.")

else:
    # Compare All 3 Algorithms
    st.subheader("⚖️ Comparative Algorithm Benchmark")
    comparison_data = []
    results_map = {}
    for algo in ["UCS", "Greedy", "A*"]:
        r = run_constrained_search(algo, chosen_sub_muscles)
        results_map[algo] = r
        path_str = " ➔ ".join([f"{STATIONS[n][0]} ({n})" for n in r["path"]])
        comparison_data.append({
            "Algorithm": algo,
            "Total Steps": len(r["path"]),
            "Active Time (mins)": f"{r['total_g']} min",
            "Nodes Expanded": r["nodes_expanded"],
            "Nodes Generated": r["nodes_generated"],
            "Generated Path": path_str
        })
    st.dataframe(pd.DataFrame(comparison_data), use_container_width=True)
    active_path = results_map["A*"]["path"]

    st.markdown("<br>", unsafe_allow_html=True)
    btn_label = "✖ Hide Graph View" if st.session_state.show_graph else "📊 View State-Space Graph"
    st.button(btn_label, on_click=toggle_graph)

# Render Graph when Button is Active
if st.session_state.show_graph:
    st.markdown("---")
    st.markdown("### 🗺️ State-Space Graph Traversal")
    st.caption("Active optimal sequence highlighted in glowing cyan along the gym transition topology.")

    fig, ax = plt.subplots(figsize=(13, 6))
    fig.patch.set_facecolor('#0b0f19')
    ax.set_facecolor('#0b0f19')

    G = nx.DiGraph()
    for u in BASE_GRAPH:
        for v in BASE_GRAPH[u]:
            G.add_edge(u, v)
    for u in INTER_TRANSITIONS:
        for v in INTER_TRANSITIONS[u]:
            G.add_edge(u, v)

    pos = {
        0: (0, 3),
        1: (2, 5), 2: (2, 3), 3: (2, 1), 4: (2, -1),
        5: (4, 5.5), 6: (4, 4.5), 7: (4, 3), 8: (4, 1.5), 9: (4, 0),
        10: (6, 5), 11: (6, 2.5), 12: (6, 0),
        13: (8, 3)
    }

    edge_colors = []
    edge_widths = []
    for u, v in G.edges():
        if u in active_path and v in active_path and active_path.index(v) == active_path.index(u) + 1:
            edge_colors.append("#38bdf8")
            edge_widths.append(3.2)
        else:
            edge_colors.append("#1e293b")
            edge_widths.append(1.0)

    node_colors = ["#0284c7" if n in active_path else "#1e293b" for n in G.nodes()]
    node_borders = ["#ffffff" if n in active_path else "#475569" for n in G.nodes()]

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1150, edgecolors=node_borders, linewidths=1.8, ax=ax)
    labels = {n: f"{n}\\nh={H[n]}" for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, font_color="#ffffff", font_weight="bold", ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=edge_widths, arrowsize=16, arrowstyle="-|>", ax=ax)

    plt.axis("off")
    st.pyplot(fig)
    