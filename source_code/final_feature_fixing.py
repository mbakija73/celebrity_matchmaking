#this file is for fixing up the features so they are binary variables and can be used for analysis
import numpy as np
import pandas as pd
import json
import os

node_features_path = "npy_data/node_features.npy"
pure_node_features_path = "npy_data/pure_node_features.npy"
final_node_features_path = "npy_data/final_node_features.npy"


final_features = [
    # birth year and height features
    "birth_year", "height", "height_NA",
    # Zodiac features
    "zodiac_Pisces", "zodiac_NA", "zodiac_Taurus", "zodiac_Gemini", "zodiac_Scorpio",
    "zodiac_Sagittarius", "zodiac_Aries", "zodiac_Cancer", "zodiac_Aquarius",
    "zodiac_Libra", "zodiac_Virgo", "zodiac_Leo", "zodiac_Capricorn",
    # Religion features
    "rel_NA", "rel_Christian", "rel_Muslim", "rel_Jewish", "rel_Hindu",
    "rel_Buddhism", "rel_Sikh", "rel_Irreligion", "rel_Other",
    # Ethnicity features
    "eth_NA", "eth_Black", "eth_Asian", "eth_OTHER", "eth_Middle Eastern",
    "eth_White", "eth_Asian/Indian", "eth_Multiracial", "eth_Hispanic",
    # Continent features
    "cont_North_America", "cont_South_America", "cont_Europe", "cont_Africa",
    "cont_Asia", "cont_Oceania", "cont_NA",
    # Occupation features
    "occ_Music", "occ_Sports", "occ_Writing", "occ_Science", "occ_Humanties",
    "occ_Health", "occ_Buisness", "occ_Acting", "occ_Film and TV production",
    "occ_Blue Collar", 'occ_"Influencer"', "occ_Politics", "occ_Artist",
    "occ_Dance", "occ_Fashion", "occ_Entertainer", "occ_Law", "occ_Other", "occ_NA"
]


def expand_coded_features(pure_node_features, node_features):

    #load the column names
    columns = list(node_features[list(node_features.keys())[0]].keys())
    print(columns)
    expanded_features = []
 
    con_df = pd.read_csv("feature_codings/continents_coded.csv")
    rel_df = pd.read_csv("feature_codings/religions_coded.csv")

    for i, node_id in enumerate(node_features.keys()):
        original_row = list(pure_node_features[i])
        expanded_row = []
        # ['birth_year','height','height_NA','zodiac'(1-13), 'religion'(1-9),'ethnicity'(1-9),'continent_of_nationality'(1-7),'occupation'(1-19)]
        # birth_year = 0
        # height 1 
        # height NA 2
        # zodiac 3-16
        # religion 17 - 25
        # ethnicity 26 - 34
        # continent_of_nationality 35 - 41
        # occupation 42 - 60
        #60 total features


        expanded_row.append(int(original_row[0]))
        if(original_row[1] == -1):
            expanded_row.append(0)
            expanded_row.append(1)
        else: 
            expanded_row.append(float(original_row[1]))
            expanded_row.append(0)

        #zodiac
        if(original_row[2] == -1):
            expanded_row.extend(np.eye(1, 13, 1, dtype=int).flatten().tolist())
        else:
            expanded_row.extend(np.eye(1, 13, int(original_row[2]) - 1, dtype=int).flatten().tolist())
        #religion
        if(original_row[3] == -1):
            expanded_row.extend(np.eye(1, 9, 0, dtype=int).flatten().tolist())
        else:
            
            religion_code = rel_df.loc[rel_df["religion_coded"] == int(original_row[3]), "rel_group_coded"].values[0]
            expanded_row.extend(np.eye(1, 9, religion_code - 1, dtype=int).flatten().tolist())
        #ethnicity
        if(original_row[4] == -1):
            expanded_row.extend(np.eye(1, 9, 0, dtype=int).flatten().tolist())
        else:
            expanded_row.extend(np.eye(1, 9, int(original_row[4])- 1, dtype=int).flatten().tolist())
        #continent_of_nationality
        if(original_row[5] == -1):
            expanded_row.extend(np.eye(1, 7, 6, dtype=int).flatten().tolist())
        else:
            continent_code = con_df.loc[con_df["nationality_coded"] == int(original_row[5]), "continent_coded"].values[0]
            expanded_row.extend(np.eye(1, 7, continent_code - 1, dtype=int).flatten().tolist())
        #occupation
        if(original_row[5] == 0 or original_row[5] == -1):
            expanded_row.extend(np.eye(1, 19, 18, dtype=int).flatten().tolist())
        else:
            expanded_row.extend(np.eye(1, 19, int(original_row[6]) - 1, dtype=int).flatten().tolist())

        expanded_features.append(expanded_row)

    print(np.array(expanded_features, dtype=np.float32).shape)
    return np.array(expanded_features, dtype=np.float32)

def main():
    # Load node features
    node_features = np.load(node_features_path, allow_pickle=True).item()
    pure_node_features = np.load(pure_node_features_path)

    final_features = expand_coded_features(pure_node_features, node_features)
    np.save(final_node_features_path, final_features)


if __name__ == "__main__":
    main()




