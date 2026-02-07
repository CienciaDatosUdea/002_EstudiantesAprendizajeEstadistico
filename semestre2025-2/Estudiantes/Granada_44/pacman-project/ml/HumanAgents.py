from game import Agent
from keyboardAgents import KeyboardAgent
import pickle
import os

DATA_PATH = os.path.join(
    os.path.dirname(__file__),
    "winning_episodes.pkl"
)

def extract_snapshot(state):
    # Positions
    pacman_pos = state.getPacmanPosition()
    ghost_positions = state.getGhostPositions()

    #Ghost timers
    ghost_states = state.getGhostStates()
    scared_timers = [ghost.scaredTimer for ghost in ghost_states]

    #Food grid to list of 0/1
    food_grid = state.getFood()
    food_2d = [
        [1 if food_grid[x][y] else 0 for y in range(food_grid.height)]
        for x in range(food_grid.width)
    ]

    #Wall grid to list of 0/1
    wall_grid = state.getWalls()
    walls_2d = [
        [1 if wall_grid[x][y] else 0 for y in range(wall_grid.height)]
        for x in range(wall_grid.width)
    ]

    #Capsules
    capsules = state.getCapsules()

    #Score
    score = state.getScore()

    #Legal actions
    legal_actions = state.getLegalActions()

    return {
        "pacman_pos": pacman_pos,
        "ghost_positions": ghost_positions,
        "ghost_scared_timers": scared_timers,
        "food_grid": food_2d,
        "wall_grid": walls_2d,
        "capsules": capsules,
        "score": score,
        "legal_actions": legal_actions,
    }

#Same as keyboard agents but returns the rows from winning games!
class humanAgent(Agent):
    def __init__(self, **args):
        self.state_actions = []
        self.episode_saved = False
        self.keyboard_agent = KeyboardAgent(**args)

    def getAction(self, state):
        action = self.keyboard_agent.getAction(state)

        snapshot = extract_snapshot(state)
        self.state_actions.append((snapshot, action))

        next_state = state.generateSuccessor(0, action)

        if next_state.isWin() and not self.episode_saved:
            with open(DATA_PATH, "rb") as f:
                dataset = pickle.load(f)

            dataset.append(self.state_actions)

            with open(DATA_PATH, "wb") as f:
                pickle.dump(dataset, f)

            self.episode_saved = True

        if next_state.isWin() or next_state.isLose():
            self.state_actions = []
            self.episode_saved = False

        return action