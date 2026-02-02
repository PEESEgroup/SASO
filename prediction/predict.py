import pandas as pd
from feature import salt_properties, solvent_properties, c_unit_encode, solvent_ratio_type_code
import numpy as np
import torch
from route_model import create_model


def load_model(model, path='1-model.pth'):
    model.load_state_dict(torch.load(path))
    model.eval()
    print(f"Model loaded from {path}")
    return model

def predict(model, x1, x2, x3):
    model.eval()
    with torch.no_grad():
        outputs = model(x1, x2, x3)
    return outputs


if __name__ == "__main__":
    df = pd.read_csv("output_1.0.csv")
    generated_features = []

    salt_features = []
    solvent_features = []
    condition_features = []
    conductivity_target = []

    for _, row in df.iterrows():
        # Salt feature
        lithium_salt = row['salt']
        salt_vector = salt_properties[lithium_salt]
        salt_features.append(salt_vector)

        # Solvent feature
        ratio_1 = row['ratio_1']
        ratio_2 = row['ratio_2']
        #ratio_3 = row['ratio_3']
        #ratio_4 = row['ratio_4']

        solvent_1 = solvent_properties[row['solvent_1']]
        solvent_1_feature = [i*ratio_1 for i in solvent_1]
        solvent_2 = solvent_properties[row['solvent_2']]
        solvent_2_feature = [i * ratio_2 for i in solvent_2]
        #solvent_3 = solvent_properties[row['solvent_3']]
        #solvent_3_feature = [i * ratio_3 for i in solvent_3]
        #solvent_4 = solvent_properties[row['solvent_4']]
        #solvent_4_feature = [i * ratio_4 for i in solvent_4]

        #solvent_vector = [solvent_1_feature[i] + solvent_2_feature[i] + solvent_3_feature[i] + solvent_4_feature[i] for i in range(14)]
        solvent_vector = [solvent_1_feature[i] + solvent_2_feature[i] for i in range(14)]
        solvent_features.append(solvent_vector)

        condition_vector = []
        condition_vector.append(row['T'])
        condition_vector += c_unit_encode[row['c units']]
        condition_vector += solvent_ratio_type_code[row['solvent ratio type']]
        condition_vector.append(row['c'])
        condition_features.append(condition_vector)

    salt_features = np.array(salt_features)
    solvent_features = np.array(solvent_features)
    condition_features = np.array(condition_features)


    # print(salt_features.shape, solvent_features.shape, condition_features.shape)

    # np.save("salt_features.npy", salt_features)
    # np.save("solvent_features.npy", solvent_features)
    # np.save("condition_features.npy", condition_features)


    model = load_model(create_model(), path="5-model.pth")

    x_1 = torch.from_numpy(salt_features).float() # 14
    x_2 = torch.from_numpy(solvent_features).float() # 14
    x_3 = torch.from_numpy(condition_features).float() # 6


    predictions = predict(model, x_1, x_2, x_3)
    for i in predictions:
        print(i.item())

