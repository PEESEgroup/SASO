import numpy as np
import itertools



def search_best_salt(target_vec, salt_features_dict):
    best = None
    min_dist = float('inf')
    names = list(salt_features_dict.keys())

    for s1 in names:
        vec = salt_features_dict[s1]
        dist = np.linalg.norm(vec - target_vec)
        if dist < min_dist:
            best = s1
            min_dist = dist
    return best, min_dist


def search_best_2salt(target_vec, salt_features_dict):
    best = None
    min_dist = float('inf')
    names = list(salt_features_dict.keys())

    for s1, s2 in itertools.combinations(names, 2):
        for r in np.linspace(0, 1, 21):
            r1, r2 = r, 1 - r
            vec = r1 * salt_features_dict[s1] + r2 * salt_features_dict[s2]
            dist = np.linalg.norm(vec - target_vec)
            if dist < min_dist:
                best = (s1, r1, s2, r2)
                min_dist = dist
    return best, min_dist


def search_best_2solvent(target_vec, solvent_features_dict):
    best = None
    min_dist = float('inf')
    names = list(solvent_features_dict.keys())

    for s1, s2 in itertools.combinations(names, 2):
        for r in np.linspace(0, 1, 21):
            r1, r2 = r, 1 - r
            vec = r1 * solvent_features_dict[s1] + r2 * solvent_features_dict[s2]
            dist = np.linalg.norm(vec - target_vec)
            if dist < min_dist:
                best = (s1, r1, s2, r2)
                min_dist = dist
    return best, min_dist


def search_best_3solvent(target_vec, solvent_features_dict):
    best = None
    min_dist = float('inf')
    names = list(solvent_features_dict.keys())

    for s1, s2, s3 in itertools.combinations(names, 3):
        for r1 in np.linspace(0, 1, 11):
            for r2 in np.linspace(0, 1 - r1, 11):
                r3 = 1 - r1 - r2
                vec = (
                    r1 * solvent_features_dict[s1] +
                    r2 * solvent_features_dict[s2] +
                    r3 * solvent_features_dict[s3]
                )
                dist = np.linalg.norm(vec - target_vec)
                if dist < min_dist:
                    best = (s1, r1, s2, r2, s3, r3)
                    min_dist = dist
    return best, min_dist


# best, min_dist = search_best_2solvent([0.61676943, -8.083972], solvent_features_dict)

# print(best, min_dist)
