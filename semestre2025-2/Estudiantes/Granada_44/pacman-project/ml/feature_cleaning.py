from build_dataframe import rows
import pandas as pd

def distance(xg, yg, xp, yp):
    dist = abs((xp -xg)) + abs((yp - yg))
    return dist

def scared(timer1):
    if timer1>0:
        scared1 = 1
    else:
        scared1 = 0
    return scared1

def calculate_food_features(row, pacman_x, pacman_y):
    food_grid = row["food_grid"]
    food_count = 0
    min_food_dist = None

    for x in range(len(food_grid)):
        for y in range(len(food_grid[x])):
            if food_grid[x][y] == 1:
                food_count += 1
                dist = abs(pacman_x - x) + abs(pacman_y - y)
                if min_food_dist is None or dist < min_food_dist:
                    min_food_dist = dist

    if min_food_dist is None:
        min_food_dist = None

    return food_count, min_food_dist

#useful features for training
def good_row(row):
    #ghost info
    ghost1_scared = scared(row["ghost_scared_timers"][0])
    ghost2_scared = scared(row["ghost_scared_timers"][1])
    ghost1_dist = distance(row["ghost_positions"][0][0], row["ghost_positions"][0][1], row["pacman_x"], row["pacman_y"])
    ghost2_dist = distance(row["ghost_positions"][1][0], row["ghost_positions"][1][1], row["pacman_x"], row["pacman_y"])

    #food info
    food_left, food_distance = calculate_food_features(row, row["pacman_x"], row["pacman_y"])

    if len(row["capsules"]) > 0:
        cap_x, cap_y = row["capsules"][0]
        capsule_distance = distance(cap_x, cap_y, row["pacman_x"], row["pacman_y"])
    else:
        capsule_distance = 1e10

    clean = {"pacman_x": row["pacman_x"], "pacman_y": row["pacman_y"],
             "ghost1_dist": ghost1_dist, "ghost1_scared": ghost1_scared, "ghost2_dist": ghost2_dist, "ghost2_scared": ghost2_scared,
             "dist_nearest_food": food_distance, "dist_nearest_capsule": capsule_distance, "num_food_left": food_left,
             "score": row["score"], "Action": row["Action"]}
    return clean

clean_rows = []

for row in rows:
    clean = good_row(row)
    clean_rows.append(clean)

clean_df = pd.DataFrame(clean_rows)