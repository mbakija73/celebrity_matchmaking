#save edge attributes and node features to a npy and get some graph centrality statistics out of curiousity
import os
import json
import networkx as nx
import numpy as np

info_dir = 'info'

#load all json data
def load_json_files(directory):
    data = {}
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            filepath = os.path.join(directory, filename)
            node_id = filename[:-5]
            with open(filepath, 'r', encoding='utf-8') as file:
                json_data = json.load(file)
                data[node_id] = json_data
    return data

#build a graph a networkX graph
def build_graph(json_data):
    G = nx.Graph() 

    # Add nodes
    for node_id, content in json_data.items():
        details = content.get("details", {})
        G.add_node(
            node_id,
            birthday=details.get("Birthday", "NA"),
            height=details.get("Height", "NA"),
            zodiac_sign=details.get("Zodiac Sign", "NA"),
            religion=details.get("Religion", "NA"),
            ethnicity=details.get("Ethnicity", "NA"),
            nationality=details.get("Nationality", "NA"),
            occupation=details.get("Occupation", "NA"),
        )

    print("Nodes added.")

    #add edges
    edge_attributes = {}
    for node_id, content in json_data.items():
        relationships = content.get("relationships", [])
        for relationship in relationships:
            #get relationship attributes
            partner_url = relationship.get("href", "")
            start_months = relationship.get("start_months_since_1900", -1)
            end_months = relationship.get("end_months_since_1900", -1)
            length_months = relationship.get("length_in_months", -1)
            relationship_type = relationship.get("relationship_type", 0)

            #check if the partner exists in the graph by matching the URL pattern
            for target_node_id in json_data:
                expected_url = f"https://www.whosdatedwho.com/dating/{target_node_id}"
                if partner_url == expected_url:
                    #add an edge with the new features
                    G.add_edge(
                        node_id,
                        target_node_id,
                        start_months_since_1900=start_months,
                        end_months_since_1900=end_months,
                        length_in_months=length_months,
                        relationship_type=relationship_type,
                    )
                    #save edge attributes for npy
                    edge_attributes[(node_id, target_node_id)] = {
                        "start_months_since_1900": start_months,
                        "end_months_since_1900": end_months,
                        "length_in_months": length_months,
                        "relationship_type": relationship_type,
                    }

    print("Edges added")
    return G, edge_attributes

#filter graph based on conditions (medium strict or strict)
def filter_graph(G, condition_func):
    filtered_graph = nx.Graph()
    filtered_graph.add_nodes_from(G.nodes(data=True))
    #add only edges that are allowed
    for u, v, data in G.edges(data=True):
        if condition_func(data):
            filtered_graph.add_edge(u, v, **data)
    return filtered_graph

#print top 10 nodes for centrality measures in each graph
def print_top_centralities(G, description):
    print(f"\nTop 10 Nodes by Centrality Measures ({description}):")

    #closeness centrality
    closeness = nx.closeness_centrality(G)
    top_closeness = sorted(closeness.items(), key=lambda x: x[1], reverse=True)[:10]
    print("Closeness Centrality:")
    for node, score in top_closeness:
        print(f"{node}: {score:.4f}")

    #degree centrality
    degree = nx.degree_centrality(G)
    top_degree = sorted(degree.items(), key=lambda x: x[1], reverse=True)[:10]
    print("\nDegree Centrality:")
    for node, score in top_degree:
        print(f"{node}: {score:.4f}")

    #betweenness centrality
    betweenness = nx.betweenness_centrality(G)
    top_betweenness = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:10]
    print("\nBetweenness Centrality:")
    for node, score in top_betweenness:
        print(f"{node}: {score:.4f}")

def save_numpy_data(G, filename_prefix):
    #save node features and adjacency matrix
    node_features, adj_matrix = save_graph_as_numpy(G)
    np.save(f'npy_data/{filename_prefix}_node_features.npy', node_features)
    np.save(f'npy_data/{filename_prefix}_adj_matrix.npy', adj_matrix)
    print(f"Saved: {filename_prefix}_node_features.npy and {filename_prefix}_adj_matrix.npy")

#save graph data into numpy-friendly structures
def save_graph_as_numpy(G):

    nodes = list(G.nodes(data=True))
    node_features = {node: attrs for node, attrs in nodes}

    #create an adjacency matrix
    adj_matrix = nx.adjacency_matrix(G).todense()
    return node_features, adj_matrix

def main():

    json_data = load_json_files(info_dir)

    #build the main graph
    graph, edge_attributes = build_graph(json_data)

    #save edge attributes to npy
    np.save("npy_data/edge_attributes.npy", edge_attributes)
    print("Edge attributes saved as 'edge_attributes.npy'.")
    save_numpy_data(graph, "npy_data/full_graph")

    #medium strict filter: Exclude edges with relationship_type {0, 1, 5}
    G_medium = filter_graph(graph, lambda data: data["relationship_type"] not in {0, 1, 5})
    save_numpy_data(G_medium, "npy_data/medium_graph")

    #strict filter: Exclude edges with relationship_type {0, 1, 5} and length == -1
    G_strict = filter_graph(graph, lambda data: data["relationship_type"] not in {0, 1, 5} and data["length_in_months"] != -1)
    save_numpy_data(G_strict, "npy_data/strict_graph")

    print_top_centralities(graph, "Full Graph")
    print_top_centralities(G_medium, "Medium Strict Graph")
    print_top_centralities(G_strict, "Strict Graph")

if __name__ == "__main__":
    main()


