import torch
from model_r import CVAEWithRouting as CVAE
from descriptor_utils import ElectrolyteDataset
import pandas as pd
from smiles import salt_features_dict, solvent_features_dict
from search_salt_solvent import search_best_salt, search_best_2salt, search_best_2solvent, search_best_3solvent
import warnings
warnings.filterwarnings("ignore")


df = pd.read_csv("compressed_new.csv")
solvent_feature_dict = solvent_features_dict
salt_feature_dict = salt_features_dict
dataset = ElectrolyteDataset(df, solvent_feature_dict, salt_feature_dict)

X, Y = dataset.get_features_and_targets()
cond_dim, out_dim = X.shape[1], Y.shape[1]
model = CVAE(cond_dim, out_dim)
model.load_state_dict(torch.load("1-cvae.pt"))
model.eval()


results = []
for i in range(1000):
    k_val = 20.0
    c = torch.tensor([[k_val]]).float()
    z = torch.randn(1, 16)

    with torch.no_grad():
        y = model.decode(z, c).numpy()

    T_c, cats, salts, solv = dataset.decode(y)


    salt_best, min_best = search_best_salt([salts[0][0], salts[0][1]], salt_features_dict)
    solvent_best, min_dist = search_best_2solvent([solv[0][0], solv[0][1]], solvent_features_dict)

    entry = {
        "T": T_c[0][0],
        "c": T_c[0][1],
        "c units": cats[0][0],
        "solvent ratio type": cats[0][1],
        "salt": salt_best,
        "solvent_1": solvent_best[0],
        "ratio_1": solvent_best[1],
        "solvent_2": solvent_best[2],
        "ratio_2": solvent_best[3],
        "k": k_val
    }
    results.append(entry)

pd.DataFrame(results).to_csv("output_20.0.csv", index=False)
    # print(T_c, cats, salt_best, solvent_best)

