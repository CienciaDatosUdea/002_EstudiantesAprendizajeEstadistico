import pickle
import os
import pandas as pd

DATA_PATH = os.path.join(
    os.path.dirname(__file__),
    "winning_episodes.pkl"
)

with open(DATA_PATH, "rb") as f:
    data = pickle.load(f)

rows = []

#Turn the raw data into an organized dict
for i in range(len(data)):
    episode = data[i]
    for t in range(len(episode)):
        step = episode[t]
        snapshot, action = step
        row = {"EpisodeID": i, "Timestep": t, "Action": action,
               "pacman_x": snapshot["pacman_pos"][0], "pacman_y": snapshot["pacman_pos"][1],
               "ghost_positions": snapshot["ghost_positions"], "ghost_scared_timers": snapshot["ghost_scared_timers"],
               "food_grid": snapshot["food_grid"], "wall_grid": snapshot["wall_grid"],
               "capsules": snapshot["capsules"], "score": snapshot["score"], 
               "legal_actions": snapshot["legal_actions"],}
        rows.append(row)

df = pd.DataFrame(rows)