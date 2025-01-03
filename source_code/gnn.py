#code to load features and create graphs then train a model for relationship (edge) prediction
import torch
import numpy as np
import networkx as nx
from torch_geometric.data import Data
import dgl
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm1d, BatchNorm2d
from dgl.nn import SAGEConv
import dgl.function as fn
import itertools
import numpy as np
import sklearn
from sklearn.metrics import roc_auc_score

#paths to use
node_features_path = 'npy_data/final_node_features.npy'
edge_list_path = 'npy_data/pure_edge_list.npy'
edge_attributes_path = 'npy_data/pure_edge_attributes.npy'

#using the cleaned pure node futures turns into tensor object
def analyze_node_features(node_features_path, verbose=False):

    node_features = np.load(node_features_path)
    num_nodes = node_features.shape[0]
    num_features = node_features.shape[1]

    x = torch.tensor(node_features, dtype=torch.float)
    if(verbose):
        print(f"Number of nodes: {num_nodes}")
        print(f"Number of features: {num_features}")
        print(f"Node features tensor shape: {x.shape}")

    return x


#pure edge list and pure edge attributes
def analyze_edge_list(edge_list_path,edge_attribute_path,verbose=False):
    
    #load edge list and attributes and return as tensors
    edge_list = np.load(edge_list_path)
    edge_list = torch.tensor(edge_list, dtype=torch.long)

    edge_attributes = np.load(edge_attributes_path)
    edge_attributes = torch.tensor(edge_attributes, dtype=torch.float)

    if(verbose==True):
        print(f"Number of edges: {edge_list.shape[0]}")

    return edge_list, edge_attributes


#NetworkX graph
def networkx_graph_info(edge_list, num_nodes):
    
    G = nx.Graph()
    G.add_edges_from(edge_list.tolist())

    #add isolated nodes
    G.add_nodes_from(range(num_nodes))

    #get average degree
    avg_degree = sum(dict(G.degree()).values()) / num_nodes
    num_subgraphs = nx.number_connected_components(G)

    print(f"Average node degree: {avg_degree:.2f}")
    print(f"Number of subgraphs: {num_subgraphs}")

    return G


#chosing negative samples: 1. could just do a negative random sample but would probably be easy to discern because
#chosing out of a completely random group is likely to make very unlikely parings if not impossible:
#ie Lebron James and William Shakespeare are never gonna be in a relationship
#so can we make a more challenging selection of random negative samples? 
#inductive random sampling from Towards Better Evaluation for Dynamic Link Prediction
#note to self if they break up after split year should that be reflected in the edge_attributes?

def training_test_split(edge_list, edge_attributes, split_year, alpha=None, negative_sampling="Random", verbose=False):

    #convert split_year to months since 1900
    split_months_since_1900 = (split_year - 1900) * 12
    
    training_edges = []
    test_edges = []
    training_attributes = []
    test_attributes = []

    #split edges based on start date
    for i, edge in enumerate(edge_list):  
        start_months = edge_attributes[i, 0]  #first attribute: start_months_since_1900
        if start_months == -1:  #save for later
            continue
        if start_months < split_months_since_1900: #if after 
            training_edges.append(edge)
            training_attributes.append(edge_attributes[i])
        else:
            test_edges.append(edge)
            test_attributes.append(edge_attributes[i])


    #random selection of nodes without start date
    if alpha is not None:  #handle edges with no start date (-1)
        no_start_edges = [i for i, attr in enumerate(edge_attributes) if attr[0] == -1]

        num_no_start_to_add = int(alpha * len(no_start_edges))
        selected_indices = np.random.choice(no_start_edges, num_no_start_to_add, replace=False)

        selected_set = set(selected_indices)
        for idx in selected_indices:
            test_edges.append(edge_list[idx, :])  
            test_attributes.append(edge_attributes[idx])

        for idx in no_start_edges:
            if idx not in selected_set:
                training_edges.append(edge_list[idx, :]) 
                training_attributes.append(edge_attributes[idx])

    training_edge_list = torch.stack(training_edges).T if training_edges else torch.empty(2, 0, dtype=torch.long)
    test_edge_list = torch.stack(test_edges).T if test_edges else torch.empty(2, 0, dtype=torch.long)
    
    training_edge_attributes = torch.stack(training_attributes) if training_attributes else torch.empty(0, edge_attributes.shape[1], dtype=torch.float)
    test_edge_attributes = torch.stack(test_attributes) if test_attributes else torch.empty(0, edge_attributes.shape[1], dtype=torch.float)
    #Gather negative edges
    num_nodes = edge_list.max() + 1
    num_training_negatives = training_edge_list.shape[1]
    num_test_negatives = test_edge_list.shape[1]

    def sample_negative_edges(num_negatives, existing_edges):
        negative_edges = set()
        while len(negative_edges) < num_negatives:
            u = np.random.randint(0, num_nodes)
            v = np.random.randint(0, num_nodes)
            if u != v and (u, v) not in existing_edges and (v, u) not in existing_edges:
                negative_edges.add((u, v))
        return list(negative_edges)

    existing_edges = set(map(tuple, edge_list.tolist()))

    if negative_sampling == "Random":
        training_negatives = sample_negative_edges(num_training_negatives, existing_edges)
        test_negatives = sample_negative_edges(num_test_negatives, existing_edges)

    elif negative_sampling == "Inductive":
        test_negative_subset = sample_negative_edges(num_test_negatives // 2, existing_edges)
        training_negatives = sample_negative_edges(
            num_training_negatives - len(test_negative_subset), existing_edges.union(test_negative_subset)
        ) + test_negative_subset
        test_negatives = sample_negative_edges(num_test_negatives, existing_edges.union(training_negatives))

    #convert to tensors
    training_negative_edges = torch.tensor(training_negatives).T if training_negatives else torch.empty(2, 0, dtype=torch.long)
    test_negative_edges = torch.tensor(test_negatives).T if test_negatives else torch.empty(2, 0, dtype=torch.long)

    #combine positive and negative edges
    final_training_edges = torch.cat([training_edge_list, training_negative_edges])
    final_test_edges = torch.cat([test_edge_list, test_negative_edges])

    if verbose:
        print("Training Positive Edges:", training_edge_list.shape[1])
        print("Training Negative Edges:", training_negative_edges.shape[1])
        print("Test Positive Edges:", test_edge_list.shape[1])
        print("Test Negative Edges:", test_negative_edges.shape[1])
        print("Final Training Edge List Shape:", final_training_edges.shape)
        print("Final Test Edge List Shape:", final_test_edges.shape)

    return final_training_edges, training_edge_attributes, final_test_edges, test_edge_attributes



#graph sage with batch normalization
class GraphSAGE(nn.Module):
    def __init__(self, in_feats, h_feats):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_feats, h_feats, 'lstm')
        self.bn1 = nn.BatchNorm1d(h_feats)  #BatchNorm 
        self.conv2 = SAGEConv(h_feats, h_feats, 'lstm')
        self.bn2 = nn.BatchNorm1d(h_feats)  #BatchNorm 

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = self.bn1(h)  #Applying BatchNorm
        h = F.relu(h)    

        h = self.conv2(g, h)
        h = self.bn2(h)  
        return h


class DotPredictor(nn.Module):
    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            g.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            return g.edata['score'][:, 0]


def compute_loss(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score])
    labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])])
    return F.binary_cross_entropy_with_logits(scores, labels)

def compute_auc(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score]).numpy()
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).numpy()
    return roc_auc_score(labels, scores)



def main():
    #Load 
    x = (analyze_node_features(node_features_path))
    edge_list, edge_attributes = analyze_edge_list(edge_list_path, edge_attributes_path)

    split_year = 2018
    alpha = 0.1 #random selection of nodes with no start date
    negative_sampling = "Random" 

    #training and test edges and attributes
    training_edges, training_attributes, test_edges, test_attributes = training_test_split(
        edge_list, edge_attributes, split_year, alpha=alpha, negative_sampling=negative_sampling, verbose=False
    )

    #split positive and negative edges into source(u) and destination(v)
    train_pos_u, train_pos_v = training_edges[0,:], training_edges[1,:]
    test_pos_u, test_pos_v = test_edges[0,:], test_edges[1,:]

    #create the positive DGL graphs, and full ones with features
    train_pos_g = dgl.graph((train_pos_u, train_pos_v), num_nodes=x.shape[0])
    test_pos_g = dgl.graph((test_pos_u, test_pos_v), num_nodes=x.shape[0])
    train_g = dgl.graph((train_pos_u, train_pos_v), num_nodes=x.shape[0])
    test_g = dgl.graph((test_pos_u, test_pos_v), num_nodes=x.shape[0])

    #add node features to the full graphs
    train_g.ndata['feat'] = x
    test_g.ndata['feat'] = x

    #add edge attributes
    train_g.edata['attr'] = training_attributes
    test_g.edata['attr'] = test_attributes
 
    #full graph for statistics
    g = dgl.merge([train_g, test_g])
    print("number of edges in entire graph", g.number_of_edges())
    print("number of edges in train graph", train_g.number_of_edges())

    #create negative DGL graphs
    train_neg_u, train_neg_v = training_edges[2,:], training_edges[3,:]
    test_neg_u, test_neg_v = test_edges[2,:], test_edges[3,:]
    train_neg_g = dgl.graph((train_neg_u, train_neg_v), num_nodes=x.shape[0])
    test_neg_g = dgl.graph((test_neg_u, test_neg_v), num_nodes=x.shape[0])

    #create GraphSage model and start training
    model = GraphSAGE(train_g.ndata['feat'].shape[1], 32)
   
    pred = DotPredictor()

    optimizer = torch.optim.Adam(itertools.chain(model.parameters(), pred.parameters()), lr=0.01)

    for e in range(500):
        #forward pass
        h = model(train_g, train_g.ndata['feat'])
        pos_score = pred(train_pos_g, h)
        neg_score = pred(train_neg_g, h)

        #compute loss
        loss = compute_loss(pos_score, neg_score)

        #backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if e % 10 == 0:
            print(f"Epoch {e}, Loss: {loss.item()}")

    #check with test graph
    with torch.no_grad():
        pos_score = pred(test_pos_g, h)
        neg_score = pred(test_neg_g, h)
        print('AUC', compute_auc(pos_score, neg_score))


if __name__ == "__main__":
    main()





