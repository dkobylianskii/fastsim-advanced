import dgl

import torch
import torch.nn as nn


def build_layers(
    inputsize,
    outputsize,
    features,
    add_batch_norm=False,
    add_activation=None,
    add_dropout=False,
):
    layers = []
    layers.append(nn.Linear(inputsize, features[0]))
    layers.append(nn.ReLU())
    for hidden_i in range(1, len(features)):
        if add_batch_norm:
            layers.append(nn.BatchNorm1d(features[hidden_i - 1]))
        layers.append(nn.Linear(features[hidden_i - 1], features[hidden_i]))

        layers.append(nn.ReLU())
        if add_dropout:
            layers.append(nn.Dropout(p=0.5))
    layers.append(nn.Linear(features[-1], outputsize))
    if add_activation != None:
        layers.append(add_activation)
    return nn.Sequential(*layers)


class SetSizeEncoder(nn.Module):
    def __init__(self, top_level_config):
        super().__init__()

    def forward(self, g):
        cells_global = dgl.mean_nodes(g, "global rep 0", ntype="cells")
        tracks_global = dgl.mean_nodes(g, "global rep 0", ntype="tracks")
        all_global = torch.cat([cells_global, tracks_global], dim=1)

        predicted_setsizes = self.predictor(all_global)

        g.nodes["global node"].data["set size prediction"] = predicted_setsizes

        return g
