#deal with missing data, create node index to key, fix edge attributes and node features to be more useable
import numpy as np
import random
import os
import json

node_features_path = 'npy_data/node_features.npy'
edge_attributes_path = 'npy_data/edge_attributes.npy'
info_dir = 'info'  # Directory containing JSON files

# Output paths
pure_node_features_path = 'npy_data/pure_node_features.npy'
pure_edge_list_path = 'npy_data/pure_edge_list.npy'
medium_edge_list_path = 'npy_data/medium_edge_list.npy'
strict_edge_list_path = 'npy_data/strict_edge_list.npy'
pure_edge_attributes_path = 'npy_data/pure_edge_attributes.npy'
medium_edge_attributes_path = 'npy_data/medium_edge_attributes.npy'
strict_edge_attributes_path = 'npy_data/strict_edge_attributes.npy'
node_index_to_key_path = 'npy_data/node_index_to_key.json'


def process_node_features(node_features_path):
    
    node_features = np.load(node_features_path, allow_pickle=True).item()

    nodes = list(node_features.keys())
    features_list = []
    #keep track of column names
    column_names = []

    #median year for birthdays (calculated previously; replace no birth with)
    median_birth_year = 1974

    #create a mapping of node index to node key
    node_index_to_key = {idx: key for idx, key in enumerate(nodes)}

    for node_id in nodes:
        features = node_features[node_id]
        numeric_features = []
        birthday_missing = 0  

        for key, value in features.items():
            if key == "birthday":
                #some birthdays not transformed to year earlier accidently, convert here
                if value == 'NA' or not value.isdigit():
                    numeric_features.append(median_birth_year)
                    birthday_missing = 1 
                else:
                    numeric_features.append(float(value))
            else:
                if value == 'NA':
                    numeric_features.append(-1)  #replace all 'NA' with -1
                else:
                    numeric_features.append(float(value))

            if key not in column_names:
                column_names.append(key)

        #add birthday missing feature
        numeric_features.append(birthday_missing)
        if "birthday_missing" not in column_names:
            column_names.append("birthday_missing")

        features_list.append(numeric_features)

    #convert to a npy
    features_matrix = np.array(features_list, dtype=np.float32)

    #print column names
    print("Column Names in Order:")
    for idx, name in enumerate(column_names):
        print(f"{idx}: {name}")

    return nodes, features_matrix, node_index_to_key



#function to process edge attributes
def process_edge_attributes(edge_attributes_path, edge_list, nodes):
    #load edge attributes
    edge_attributes = np.load(edge_attributes_path, allow_pickle=True).item()

    #map edge attributes back to a numeric array
    edge_attribute_matrix = []
    node_index_map = {node_id: idx for idx, node_id in enumerate(nodes)}

    for edge in edge_list:
        node1, node2 = edge
        node1_id = nodes[node1]
        node2_id = nodes[node2]

        #check if the edge exists in the attributes dictionary
        if (node1_id, node2_id) in edge_attributes:
            attributes = edge_attributes[(node1_id, node2_id)]
            numeric_attributes = [
                attributes.get("start_months_since_1900", -1),
                attributes.get("end_months_since_1900", -1),
                attributes.get("length_in_months", -1),
                attributes.get("relationship_type", -1),
            ]
        else:
            #default attributes for missing edges
            numeric_attributes = [-1, -1, -1, -1]

        edge_attribute_matrix.append(numeric_attributes)

    return np.array(edge_attribute_matrix, dtype=np.float32)

#function to validate random nodes
def validate_random_nodes(node_index_to_key, edge_lists, info_dir):
    random_nodes = random.sample(range(len(node_index_to_key)), 5)

    for node in random_nodes:
        node_name = node_index_to_key[node]
        print(f"\nNode #{node} ({node_name}) relationships:")

        for edge_type, edge_list in edge_lists.items():
            print(f"  Checking {edge_type} edge list:")
            found_relationship = False

            for edge in edge_list:
                if node in edge:
                    partner_node = edge[1] if edge[0] == node else edge[0]
                    partner_name = node_index_to_key[partner_node]
                    print(f"    Connected to: Node #{partner_node} ({partner_name})")
                    found_relationship = True

                    # Validate using original json
                    partner_info_path = os.path.join(info_dir, f"{partner_name}.json")
                    if os.path.exists(partner_info_path):
                        with open(partner_info_path, 'r', encoding='utf-8') as file:
                            partner_data = json.load(file)
                            relationships = partner_data.get("relationships", [])
                            if any(r.get("href", "").endswith(node_name) for r in relationships):
                                print(f"    Relationship found in {partner_name}.json")
                            else:
                                print(f"    Relationship NOT found in {partner_name}.json")
            if not found_relationship:
                print(f"    No relationships found in {edge_type} edge list.")


#fix birthdays that were missed and not converted to just year (ex Sep 1947)
def analyze_birthdays_raw(node_features_path):
    
    node_features = np.load(node_features_path, allow_pickle=True).item()
    birthdays = []
    for node_id, features in node_features.items():
        birthday = features.get("birthday", "NA")
        if birthday.isdigit():  #check if it's a valid year
            birthdays.append(int(birthday))
        else:
            birthdays.append(-1)

    #convert to a npy for analysis
    birthdays = np.array(birthdays)

    #filter out missing values (-1 indicates missing)
    valid_birthdays = birthdays[birthdays > 0]

    #calculate statistics
    mean_birthday = valid_birthdays.mean() if len(valid_birthdays) > 0 else None
    median_birthday = np.median(valid_birthdays) if len(valid_birthdays) > 0 else None
    num_missing = len(birthdays) - len(valid_birthdays)

    #print analysis
    print("Birthday Analysis:")
    print(f"  Mean birthday (year): {mean_birthday}")
    print(f"  Median birthday (year): {median_birthday}")
    print(f"  Missing birthdays: {num_missing} out of {len(birthdays)}")


def main():
    #analyze_birthdays_raw(node_features_path)
    node_features = np.load(node_features_path, allow_pickle=True).item()
    nodes = list(node_features.keys())
    node_index_to_key = {idx: key for idx, key in enumerate(nodes)}

    # Validate relationships for 5 random nodes
    pure_edge_list = np.load(pure_edge_list_path)
    medium_edge_list = np.load(medium_edge_list_path)
    strict_edge_list = np.load(strict_edge_list_path)
    edge_lists = {
        "pure": pure_edge_list,
        "medium": medium_edge_list,
        "strict": strict_edge_list,
    }
    validate_random_nodes(node_index_to_key, edge_lists, info_dir)

if __name__ == "__main__":
    main()
