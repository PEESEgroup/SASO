import itertools

import numpy as np


def search_best_salt(target_vec, salt_features_dict):
    best = None
    min_dist = float("inf")
    for name, vector in salt_features_dict.items():
        distance = np.linalg.norm(vector - target_vec)
        if distance < min_dist:
            best = name
            min_dist = distance
    return best, min_dist


def search_best_2salt(target_vec, salt_features_dict):
    best = None
    min_dist = float("inf")
    names = list(salt_features_dict)
    for salt_1, salt_2 in itertools.combinations(names, 2):
        for ratio_1 in np.linspace(0, 1, 21):
            ratio_2 = 1 - ratio_1
            vector = (
                ratio_1 * salt_features_dict[salt_1]
                + ratio_2 * salt_features_dict[salt_2]
            )
            distance = np.linalg.norm(vector - target_vec)
            if distance < min_dist:
                best = (salt_1, ratio_1, salt_2, ratio_2)
                min_dist = distance
    return best, min_dist


def search_best_2solvent(target_vec, solvent_features_dict):
    best = None
    min_dist = float("inf")
    names = list(solvent_features_dict)
    for solvent_1, solvent_2 in itertools.combinations(names, 2):
        for ratio_1 in np.linspace(0, 1, 21):
            ratio_2 = 1 - ratio_1
            vector = (
                ratio_1 * solvent_features_dict[solvent_1]
                + ratio_2 * solvent_features_dict[solvent_2]
            )
            distance = np.linalg.norm(vector - target_vec)
            if distance < min_dist:
                best = (solvent_1, ratio_1, solvent_2, ratio_2)
                min_dist = distance
    return best, min_dist


def search_best_3solvent(target_vec, solvent_features_dict):
    best = None
    min_dist = float("inf")
    names = list(solvent_features_dict)
    for solvent_1, solvent_2, solvent_3 in itertools.combinations(names, 3):
        for ratio_1 in np.linspace(0, 1, 11):
            for ratio_2 in np.linspace(0, 1 - ratio_1, 11):
                ratio_3 = 1 - ratio_1 - ratio_2
                vector = (
                    ratio_1 * solvent_features_dict[solvent_1]
                    + ratio_2 * solvent_features_dict[solvent_2]
                    + ratio_3 * solvent_features_dict[solvent_3]
                )
                distance = np.linalg.norm(vector - target_vec)
                if distance < min_dist:
                    best = (
                        solvent_1,
                        ratio_1,
                        solvent_2,
                        ratio_2,
                        solvent_3,
                        ratio_3,
                    )
                    min_dist = distance
    return best, min_dist
