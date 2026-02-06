# ml/supervisedAgents.py
from pacman import Directions
from game import Agent
import os
import sys
import pickle

#Code fixed by deepseek, was having a problem with the imports for pacman.py to read

class SupervisedAgent(Agent):

    def __init__(self):
        super().__init__()
        self.clf = None
        self.distance = None
        self.scared = None
        self._model_loaded = False
        self.calculate_food_features = None
        self.extract_snapshot = None
        self.pd = None
    
    def _load_model(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        from feature_cleaning import distance, scared, calculate_food_features
        from HumanAgents import extract_snapshot
        import pandas as pd

        #Store for use in other methods
        self.distance = distance
        self.scared = scared
        self.calculate_food_features = calculate_food_features
        self.extract_snapshot = extract_snapshot
        self.pd = pd

        try:
            model_path = os.path.join(os.path.dirname(__file__), 'trained_model.pkl')
            with open(model_path, 'rb') as f:
                self.clf = pickle.load(f)
            print("Pre-trained model loaded successfully!")
            self._model_loaded = True
        except Exception as e:
            print(f"Error loading model: {e}")
        if self._model_loaded:
            return
    
    # Keep your exact functions but add self parameter
    def good_row(self, row):
        #ghost info
        ghost1_scared = self.scared(row["ghost_scared_timers"][0])
        ghost2_scared = self.scared(row["ghost_scared_timers"][1])
        ghost1_dist = self.distance(row["ghost_positions"][0][0], row["ghost_positions"][0][1], row["pacman_x"], row["pacman_y"])
        ghost2_dist = self.distance(row["ghost_positions"][1][0], row["ghost_positions"][1][1], row["pacman_x"], row["pacman_y"])

        #food info
        food_left, food_distance = self.calculate_food_features(row, row["pacman_x"], row["pacman_y"])

        if len(row["capsules"]) > 0:
            cap_x, cap_y = row["capsules"][0]
            capsule_distance = self.distance(cap_x, cap_y, row["pacman_x"], row["pacman_y"])
        else:
            capsule_distance = 1e10

        clean = {"pacman_x": row["pacman_x"], "pacman_y": row["pacman_y"],
                 "ghost1_dist": ghost1_dist, "ghost1_scared": ghost1_scared, "ghost2_dist": ghost2_dist, "ghost2_scared": ghost2_scared,
                 "dist_nearest_food": food_distance, "dist_nearest_capsule": capsule_distance, "num_food_left": food_left,
                 "score": row["score"]}
        return self.pd.DataFrame([clean])
    
    def get_row(self, snapshot):
        row = {"pacman_x": snapshot["pacman_pos"][0], "pacman_y": snapshot["pacman_pos"][1],
               "ghost_positions": snapshot["ghost_positions"], "ghost_scared_timers": snapshot["ghost_scared_timers"],
               "food_grid": snapshot["food_grid"], "wall_grid": snapshot["wall_grid"],
               "capsules": snapshot["capsules"], "score": snapshot["score"], 
               "legal_actions": snapshot["legal_actions"]}
        return self.good_row(row)

    def getAction(self, state):
        #Load model on first call
        if not self._model_loaded:
            self._load_model()
        
        legal = state.getLegalPacmanActions()

        snapshot = self.extract_snapshot(state)
        row = self.get_row(snapshot)

        action = self.clf.predict(row)[0]
        if action not in legal:
            return Directions.STOP
        return action