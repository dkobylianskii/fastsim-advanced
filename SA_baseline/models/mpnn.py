import dgl
import dgl.function as fn

import torch
import torch.nn as nn


def build_layers(
    inputsize, outputsize, features, add_batch_norm=False, add_activation=None
):
    layers = []
    layers.append(nn.Linear(inputsize, features[0]))
    layers.append(nn.ReLU())
    for hidden_i in range(1, len(features)):
        if add_batch_norm:
            layers.append(nn.BatchNorm1d(features[hidden_i - 1]))
        layers.append(nn.Linear(features[hidden_i - 1], features[hidden_i]))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(features[-1], outputsize))
    if add_activation != None:
        layers.append(add_activation)
    return nn.Sequential(*layers)


class NodeNetwork(nn.Module):
    def __init__(self, inputsize, outputsize, layers):
        super().__init__()
        self.net = build_layers(3 * inputsize, outputsize, layers)

    def forward(self, x):
        inputs = torch.sum(x.mailbox["message"], dim=1)
        inputs = torch.cat(
            [inputs, x.data["features"], x.data["global_features"]], dim=1
        )

        output = self.net(inputs)
        output = output / (torch.norm(output, p="fro", dim=1, keepdim=True) + 1e-8)

        return {"features": output}


class MPNN(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        emb_model = self.config["embedding_model"]

        self.init_network = build_layers(
            emb_model["truth_inputsize"],
            emb_model["truth_hidden_size"],
            emb_model["truth_init_layers"],
            add_batch_norm=False,  # True
        )

        self.hidden_size = emb_model["truth_hidden_size"]

        self.n_iter = emb_model["n_iter"]
        self.node_update_networks = nn.ModuleList()
        for iter_i in range(self.n_iter):
            self.node_update_networks.append(
                NodeNetwork(
                    self.hidden_size, self.hidden_size, emb_model["truth_mpnn_layers"]
                )
            )

    def update_global_rep(self, g):
        global_rep = dgl.sum_nodes(g, "features", ntype="truth_particles")
        global_rep = global_rep / (
            torch.norm(global_rep, p="fro", dim=1, keepdim=True) + 1e-8
        )

        g.nodes["truth_particles"].data["global_features"] = dgl.broadcast_nodes(
            g, global_rep, ntype="truth_particles"
        )
        g.nodes["global_node"].data["global_features"] = global_rep

    def forward(self, g):
        g.nodes["truth_particles"].data["features"] = self.init_network(
            g.nodes["truth_particles"].data["features"]
        )
        self.update_global_rep(g)

        for iter_i in range(self.n_iter):
            g.update_all(
                fn.copy_u("features", "message"),
                self.node_update_networks[iter_i],
                etype="truth_to_truth",
            )
            self.update_global_rep(g)

        return g
