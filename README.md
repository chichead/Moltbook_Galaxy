# 🌌 Moltbook Galaxy: Agent Evolution Visualization

Moltbook Galaxy is an interactive network visualization tool designed to map the influence and evolution of AI agents within the Moltbook ecosystem. It provides a "time-lapse" view of how agents interact, gain influence, and change their personas over time.

![Moltbook Galaxy Screenshot](https://raw.githubusercontent.com/chichead/Moltbook_Galaxy/main/assets/preview.png)


<br/>

## 🚀 Key Features

- **Interactive Map**: Pan and zoom through a cosmic network of agents.
- **Timelapse Timeline**: A bottom slider allows you to traverse history (from Jan 28 to Feb 7) and watch the galaxy evolve.
- **Dynamic Transitions**: Agents fade in/out and move smoothly as they gain influence or change factions.
- **Agent Search**: Quickly find and focus on specific agents with a pulsating highlight and auto-zoom.
- **Persona Filtering**: View agents categorized by their psychological profiles (Developer, Philosopher, Theologist, etc.).


<br/>

## 🛠 Project Structure

- **`index.html`**: The main entry point. A pure static HTML/JS file that renders the visualization using HTML5 Canvas & D3.js.
- **`results/galaxy_history_data.js`**: The core data engine containing the consolidated historical snapshots.
- **`snapshot_tool/`**: Python-based pipeline for processing raw Moltbook data into visual snapshots.
  - `generate_evolution_snapshots.py`: Generates daily coordinate and persona data.
  - `consolidate_history.py`: Merges snapshots into the web-optimized JS data file.
- **`data/moltbook.db`**: SQLite database containing the source of truth for all agent activities.


<br/>

## 💻 Tech Stack

- **Frontend**: Vanilla JS, D3.js (for transformation and zoom logic), HTML5 Canvas (for high-performance rendering).
- **Backend Data Pipeline**: Python, Pandas, SQLAlchemy (SQLite), Scikit-Learn (t-SNE for coordinate generation).
- **Deployment**: GitHub Pages.


<br/>

## ⚖️ License

This project is for personal research and visualization purposes. All rights reserved.
