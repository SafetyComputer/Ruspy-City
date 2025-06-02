import json
import os
import numpy as np


def convert_move_to_index(dest, wall):
    # dest is a tuple (x, y), wall is an int (0-3)

    match wall:
        case 0:  # UP
            return dest[0] + (dest[1] - 1) * 7
        case 1:  # DOWN
            return dest[0] + dest[1] * 7 + 42
        case 2:  # LEFT
            return dest[0] - 1 + dest[1] * 6 + 84
        case 3:  # RIGHT
            return dest[0] + dest[1] * 6 + 126
        case _:
            raise ValueError(
                f"Invalid wall value: {wall}. Must be 0, 1, 2, or 3.")


def convert_index_to_move(index):
    if 0 <= index < 42:  # UP wall
        x = index % 7
        y = index // 7 + 1
        wall = 0
    elif 42 <= index < 84:  # DOWN wall
        index -= 42
        x = index % 7
        y = index // 7
        wall = 1
    elif 84 <= index < 126:  # LEFT wall
        index -= 84
        x = index % 6 + 1
        y = index // 6
        wall = 2
    elif 126 <= index < 168:  # RIGHT wall
        index -= 126
        x = index % 6
        y = index // 6
        wall = 3
    else:
        raise ValueError(f"Invalid index: {index}")

    return (x, y), wall


def convert_json_to_dict(json_file_path):
    result = []

    with open(json_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                data = json.loads(line.strip())
                if isinstance(data, dict):
                    result.append(data)
                else:
                    print(f"Skipping line: {line.strip()} (not a dictionary)")
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e} in line: {line.strip()}")

    return result


def convert_board_to_feature_planes(blue_pos, green_pos, hor_wall, ver_wall, blue_turn):
    feature_planes = np.zeros((5, 7, 7), dtype=np.int8)

    feature_planes[0, blue_pos[0], blue_pos[1]] = 1
    feature_planes[1, green_pos[0], green_pos[1]] = 1

    for x in range(7):
        for y in range(6):
            if hor_wall[y][x]:
                feature_planes[2, y, x] = 1

    for x in range(6):
        for y in range(7):
            if ver_wall[y][x]:
                feature_planes[3, y, x] = 1

    feature_planes[4, :, :] = blue_turn
    return feature_planes


def data_augmentation(feature_planes, best_move_index, eval):
    # Augment the feature planes by flipping and rotating
    # returns np.ndarray of shape (8, 5, 7, 7) + # np.ndarray of shape (8, 1) for best_move and eval
    augmented_planes = np.zeros((8, 5, 7, 7), dtype=np.int8)
    augmented_best_move = np.zeros((8,), dtype=np.int32)
    augmented_eval = np.ones((8,), dtype=np.float32) * \
                     eval  # Assuming eval is a float

    # Base case - original orientation
    augmented_planes[0] = feature_planes
    augmented_best_move[0] = best_move_index

    # Rotate 90 degrees
    rotated_90 = np.rot90(feature_planes, k=-1, axes=(1, 2)).copy()
    rotated_90[[2, 3]] = rotated_90[[3, 2]]
    rotated_90[3, :, 0: 6] = rotated_90[3, :, 1: 7]
    rotated_90[3, :, 6] = 0
    augmented_planes[1] = rotated_90
    augmented_best_move[1] = transform_move_index(best_move_index, "rot90")

    # Rotate 180 degrees
    rotated_180 = np.rot90(feature_planes, k=-2, axes=(1, 2)).copy()
    rotated_180[3, :, 0: 6] = rotated_180[3, :, 1: 7]
    rotated_180[3, :, 6] = 0
    rotated_180[2, 0: 6, :] = rotated_180[2, 1: 7, :]
    rotated_180[2, 6, :] = 0
    augmented_planes[2] = rotated_180
    augmented_best_move[2] = transform_move_index(best_move_index, "rot180")

    # Rotate 270 degrees
    rotated_270 = np.rot90(feature_planes, k=-3, axes=(1, 2)).copy()
    rotated_270[[2, 3]] = rotated_270[[3, 2]]
    rotated_270[2, 0: 6, :] = rotated_270[2, 1: 7, :]
    rotated_270[2, 6, :] = 0
    augmented_planes[3] = rotated_270
    augmented_best_move[3] = transform_move_index(best_move_index, "rot270")

    # Flip horizontally
    flipped_h = np.flip(feature_planes, axis=2).copy()
    flipped_h_rot0 = flipped_h.copy()
    flipped_h_rot0[3, :, 0: 6] = flipped_h_rot0[3, :, 1: 7]
    flipped_h_rot0[3, :, 6] = 0
    augmented_planes[4] = flipped_h_rot0
    augmented_best_move[4] = transform_move_index(best_move_index, "flip_h")

    # Flip horizontally + rotate 90
    flipped_h_rot90 = np.rot90(flipped_h, k=-1, axes=(1, 2)).copy()
    flipped_h_rot90[[2, 3]] = flipped_h_rot90[[3, 2]]
    flipped_h_rot90[2, 0: 6, :] = flipped_h_rot90[2, 1: 7, :]
    flipped_h_rot90[2, 6, :] = 0
    flipped_h_rot90[3, :, 0: 6] = flipped_h_rot90[3, :, 1: 7]
    flipped_h_rot90[3, :, 6] = 0
    augmented_planes[5] = flipped_h_rot90
    augmented_best_move[5] = transform_move_index(
        best_move_index, "flip_h_rot90")

    # Flip horizontally + rotate 180
    flipped_h_rot180 = np.rot90(flipped_h, k=-2, axes=(1, 2)).copy()
    flipped_h_rot180[2, 0: 6, :] = flipped_h_rot180[2, 1: 7, :]
    flipped_h_rot180[2, 6, :] = 0
    augmented_planes[6] = flipped_h_rot180
    augmented_best_move[6] = transform_move_index(
        best_move_index, "flip_h_rot180")

    # Flip horizontally + rotate 270
    flipped_h_rot270 = np.rot90(flipped_h, k=-3, axes=(1, 2)).copy()
    augmented_planes[7] = flipped_h_rot270
    augmented_best_move[7] = transform_move_index(
        best_move_index, "flip_h_rot270")

    return augmented_planes, augmented_best_move, augmented_eval


def transform_wall_direction(wall, transformation):
    # 0: UP, 1: DOWN, 2: LEFT, 3: RIGHT

    if transformation == "rot90":
        match wall:
            case 0:
                return 3
            case 1:
                return 2
            case 2:
                return 0
            case 3:
                return 1
    elif transformation == "rot180":
        match wall:
            case 0:
                return 1
            case 1:
                return 0
            case 2:
                return 3
            case 3:
                return 2
    elif transformation == "rot270":
        match wall:
            case 0:
                return 2
            case 1:
                return 3
            case 2:
                return 1
            case 3:
                return 0

    elif transformation == "flip_h":
        match wall:
            case 0:
                return 0
            case 1:
                return 1
            case 2:
                return 3
            case 3:
                return 2

    elif transformation == "flip_h_rot90":
        match wall:
            case 0:
                return 3
            case 1:
                return 2
            case 2:
                return 1
            case 3:
                return 0

    elif transformation == "flip_h_rot180":
        match wall:
            case 0:
                return 1
            case 1:
                return 0
            case 2:
                return 2
            case 3:
                return 3

    elif transformation == "flip_h_rot270":
        match wall:
            case 0:
                return 2
            case 1:
                return 3
            case 2:
                return 0
            case 3:
                return 1


def transform_pos(pos, transformation):
    # pos is a tuple (x, y)
    x, y = pos

    if transformation == "rot90":
        return (6 - y, x)
    elif transformation == "rot180":
        return (6 - x, 6 - y)
    elif transformation == "rot270":
        return (y, 6 - x)
    elif transformation == "flip_h":
        return (6 - x, y)
    elif transformation == "flip_h_rot90":
        return (6 - y, 6 - x)
    elif transformation == "flip_h_rot180":
        return (x, 6 - y)
    elif transformation == "flip_h_rot270":
        return (y, x)

    # If no transformation is applied, return the original position
    return pos


def transform_move_index(index, transformation):
    # Convert index to move coordinates and wall
    (x, y), wall = convert_index_to_move(index)

    return convert_move_to_index(
        transform_pos((x, y), transformation),
        transform_wall_direction(wall, transformation)
    )


def parse_dict(data):
    blue_pos = data["blue_pos"]
    green_pos = data["green_pos"]
    hor_wall = data["hor_wall"]
    ver_wall = data["ver_wall"]
    blue_turn = data["blue_turn"]

    best_move_dest, best_move_wall = data["best_move"]
    # Clamp evaluation between -100 and 100
    evaluation = min(max(data["evaluation"], -100), 100)

    best_move_index = convert_move_to_index(best_move_dest, best_move_wall)
    feature_planes = convert_board_to_feature_planes(
        blue_pos, green_pos, hor_wall, ver_wall, blue_turn)
    augmented_planes, augmented_best_move, augmented_eval = data_augmentation(
        feature_planes, best_move_index, evaluation)

    return augmented_planes, augmented_best_move, augmented_eval


def main():
    # get all json files in the log directory
    log_dir = "./log"  # Adjust this path to your logs directory
    json_files = [os.path.join(log_dir, f)
                  for f in os.listdir(log_dir) if f.endswith('.json')]
    results = []

    for file_path in json_files:
        print(f"Processing {file_path}...")
        result = convert_json_to_dict(file_path)
        print(f"Found {len(result)} valid JSON objects in {file_path}")
        results.extend(result)

    print(len(results))

    planes, best_moves, evals = [], [], []

    for data in results:
        augmented_planes, augmented_best_move, augmented_eval = parse_dict(
            data)
        planes.append(augmented_planes)
        best_moves.append(augmented_best_move)
        evals.append(augmented_eval)

    # concatenate all results
    planes = np.concatenate(planes, axis=0)
    best_moves = np.concatenate(best_moves, axis=0)
    evals = np.concatenate(evals, axis=0)

    print(f"Planes shape: {planes.shape}")
    print(f"Best moves shape: {best_moves.shape}")
    print(f"Evals shape: {evals.shape}")

    # Save the results to a .npz file
    output_file = "./data/processed_data.npz"
    np.savez(output_file, planes=planes, best_moves=best_moves, evals=evals)

    # # load the data back
    # loaded_data = np.load(output_file)
    # planes, best_moves, evals = loaded_data['planes'], loaded_data['best_moves'], loaded_data['evals']


if __name__ == "__main__":
    main()
    # planes = np.zeros((5, 7, 7), dtype=np.int8)
    # planes[2, 0:6, :] = 1
    # planes[3, :, 0:6] = 2
    #
    # augmented_planes, _, _ = data_augmentation(planes, 0, 0.5)
    #
    # print(augmented_planes[:, [2, 3]])