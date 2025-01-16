#model code from DGL
#  * Most of code from from https://github.com/dmlc/dgl/tree/master/examples/pytorch/deepergcn
#  * modified to work in context
 
import torch
import numpy as np
import networkx as nx
import dgl
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm1d, BatchNorm2d
from dgl.nn import SAGEConv
import dgl.function as fn
import numpy as np
from dgl.nn.pytorch.glob import AvgPooling
from dgl.nn.functional import edge_softmax

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

class DeeperGCN(nn.Module):
    def __init__(
        self,
        node_feat_dim,
        edge_feat_dim,
        hid_dim,
        out_dim,
        num_layers,
        dropout=0.0,
        beta=1.0,
        learn_beta=False,
        aggr="softmax",
        mlp_layers=1,
    ):
        super(DeeperGCN, self).__init__()

        self.num_layers = num_layers
        self.dropout = dropout
        self.gcns = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(self.num_layers):
            conv = GENConv(
                in_dim=hid_dim,
                out_dim=hid_dim,
                aggregator=aggr,
                beta=beta,
                learn_beta=learn_beta,
                mlp_layers=mlp_layers,
            )
            self.gcns.append(conv)
            self.norms.append(nn.BatchNorm1d(hid_dim, affine=True))  # Ensure this matches the hidden dimension

        self.pooling = AvgPooling()
        self.output = nn.Linear(hid_dim, out_dim)

    def forward(self, g, in_feat=None):
        with g.local_scope():
            # Extract node features directly from the graph
            if 'feat' not in g.ndata:
                raise ValueError("Graph does not have node features stored in 'ndata'. Ensure g.ndata['feat'] exists.")

            node_feats = g.ndata['feat']  # Node features, which are already in the correct format

            hv = node_feats  # Skip AtomEncoder, use features directly
            he = g.edata.get('feat', None)  # Optional: Get edge features if they exist

            for layer in range(self.num_layers):
                # Ensure the feature dimension passed to BatchNorm is consistent
                hv1 = self.norms[layer](hv)
                hv1 = F.relu(hv1)
                hv1 = F.dropout(hv1, p=self.dropout, training=self.training)
                # Only pass graph `g`, the GENConv layer will handle features internally
                hv = self.gcns[layer](g) + hv  # Call GENConv with just the graph

            h_g = self.pooling(g, hv)
            return self.output(h_g)


class GENConv(nn.Module):
   
    def __init__(
        self,
        in_dim,
        out_dim,
        aggregator="softmax",
        beta=1.0,
        learn_beta=False,
        p=1.0,
        learn_p=False,
        msg_norm=False,
        learn_msg_scale=False,
        mlp_layers=1,
        eps=1e-7,
    ):
        super(GENConv, self).__init__()

        self.aggr = aggregator
        self.eps = eps

        channels = [in_dim]
        for _ in range(mlp_layers - 1):
            channels.append(in_dim * 2)
        channels.append(out_dim)

        self.mlp = MLP(channels)
        self.msg_norm = MessageNorm(learn_msg_scale) if msg_norm else None

        self.beta = (
            nn.Parameter(torch.Tensor([beta]), requires_grad=True)
            if learn_beta and self.aggr == "softmax"
            else beta
        )
        self.p = (
            nn.Parameter(torch.Tensor([p]), requires_grad=True)
            if learn_p
            else p
        )


    #note ignores the edge attributes try to add back in later
    def forward(self, g, in_feat=None):
        with g.local_scope():
            node_feats = g.ndata['feat']  
            g.ndata["h"] = node_feats  

            g.update_all(fn.copy_u("h", "m"), fn.sum("m", "m"))

            if self.aggr == "softmax":
                g.ndata["m"] = F.relu(g.ndata["m"]) + self.eps
                g.ndata["a"] = edge_softmax(g, g.ndata["m"] * self.beta)
                g.update_all(
                    lambda edge: {"x": edge.data["m"] * edge.data["a"]},
                    fn.sum("x", "m"),
                )

            elif self.aggr == "power":
                minv, maxv = 1e-7, 1e1
                torch.clamp_(g.ndata["m"], minv, maxv)
                g.ndata["m"] = torch.pow(g.ndata["m"], self.p)
                g.update_all(fn.copy_u("m", "x"), fn.mean("x", "m"))

            else:
                raise NotImplementedError(
                    f"Aggregator {self.aggr} is not supported."
                )

            if self.msg_norm is not None:
                g.ndata["m"] = self.msg_norm(node_feats, g.ndata["m"])

            feats = node_feats + g.ndata["m"]
            return self.mlp(feats)

class MLP(nn.Sequential):

    def __init__(self, channels, act="relu", dropout=0.0, bias=True):
        layers = []

        for i in range(1, len(channels)):
            layers.append(nn.Linear(channels[i - 1], channels[i], bias))
            if i < len(channels) - 1:
                layers.append(nn.BatchNorm1d(channels[i], affine=True))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))

        super(MLP, self).__init__(*layers)
