# ⚡ AI Gym Routine Optimizer

An intelligent gym workout route optimizer built with **Streamlit**, **NetworkX**, and **Matplotlib**. It models gym equipment stations and transitions as a state-space graph and finds optimal exercise sequences using formal AI search algorithms:
- **Uniform Cost Search (UCS)**
- **Greedy Best-First Search**
- **A\* Search**

---

## 🎯 Key Features

- **Multi-Goal State-Space Search**: Augments search state as `(current_node, frozenset(unvisited_targeted_submuscles))` to plan complete workout routines covering selected muscle heads (Chest, Back, Legs, Cardio, Arms/Core).
- **Realistic Gym Topology**: Includes transitions between workout stations with recovery heuristic values ($h(n)$) and calorie burn / active time costs ($c(u, v)$).
- **Comparative Algorithm Benchmark**: Compare UCS, Greedy Best-First, and A\* side-by-side with metrics:
  - Total Active Time (minutes)
  - Number of Steps
  - Nodes Expanded & Generated
  - Full Workout Sequencing
- **State-Space Graph Visualization**: Interactive graph powered by NetworkX and Matplotlib showing active optimal paths, station nodes, and heuristic values.
- **Sleek Modern UI**: Custom high-contrast dark theme with step badges, recovery time indicators, and responsive controls.

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10+ installed

### 2. Installation
Clone the repository and install the dependencies:
```bash
git clone <repository-url>
cd FAI
pip install -r requirements.txt
```

### 3. Run the App
Launch the Streamlit app:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🏋️ Gym Station Graph Model

| Node | Station | Category | Muscle Groups |
|:---:|:---|:---|:---|
| 0 | Warm-up activities | Warm-up | Warmup |
| 1 | Skipping rope | Cardio | Cardio, Calves |
| 2 | Exercise bike | Cardio | Cardio, Quads |
| 3 | Treadmill | Cardio | Cardio, Stamina |
| 4 | Step Mill | Cardio | Cardio, Glutes, Quads |
| 5 | Dumbbell Press / Curl | Arms/Chest | Middle Chest, Biceps, Triceps |
| 6 | Barbell Bench Press | Chest | Middle Chest, Lower Chest |
| 7 | Cable-Crossover | Chest | Lower Chest, Inner Chest |
| 8 | Pulling bars (Pull-ups/Lat) | Back | Lats, Upper Back, Biceps |
| 9 | Incline Bench Press | Chest | Upper Chest, Front Delts |
| 10 | Leg Press Machine | Legs | Quads, Glutes, Calves |
| 11 | Climbing Rope | Back/Core | Lats, Forearms, Abs |
| 12 | Hammer Strength Press | Chest/Back | Upper Chest, Middle Chest, Lats |
| 13 | Stretching | Cooldown | Cooldown |
