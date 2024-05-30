import torch
import torch.nn as nn
from rms_norm import RMSNorm


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


def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.uniform_(m.weight)


class TrainableEltwiseLayer(nn.Module):
    def __init__(self, n):
        super().__init__()

        self.weights = nn.Parameter(
            torch.randn((1, n))
        )  # define the trainable parameter

    def forward(self, x):
        # assuming x of size (-1, n)
        return x * self.weights  # element-wise multiplication


class SlotAttention(nn.Module):
    def __init__(
        self,
        truth_input_size,
        fastsim_input_size,
        config,
    ):
        super().__init__()

        truth_skip_size = config["embedding_model"]["truth_inputsize"]
        self.config = config
        self.key = build_layers(
            truth_input_size + truth_skip_size,
            30,
            features=self.config["output_model"]["KQV_layers"],
        )
        self.query = build_layers(
            fastsim_input_size + truth_input_size,
            30,
            features=self.config["output_model"]["KQV_layers"],
        )
        self.values = build_layers(
            truth_input_size + truth_skip_size,
            fastsim_input_size,
            features=self.config["output_model"]["KQV_layers"],
        )

        self.gru = nn.GRUCell(fastsim_input_size + truth_skip_size, fastsim_input_size)

        self.layer_norm = nn.LayerNorm(fastsim_input_size)
        self.norm = 1 / torch.sqrt(torch.tensor([30.0]))

        self.mlp = nn.Sequential(
            nn.Linear(fastsim_input_size, 64),
            nn.ReLU(),
            nn.Linear(64, fastsim_input_size),
        )

        self.rmsnorm = RMSNorm(fastsim_input_size + truth_skip_size)
        self.lin_weights = TrainableEltwiseLayer(fastsim_input_size + truth_skip_size)

    def edge_function(self, edges):
        attention = torch.sum(edges.src["key"] * edges.dst["query"], dim=1) * self.norm

        # new skip connection
        values = torch.cat([edges.src["values"], edges.src["features_0"]], dim=1)
        edges.data["attention_weights"] = attention

        return {"attention": attention, "values": values}

    def edge_attention_function(self, edges):
        attention_weights = (
            torch.exp(edges.data["attention_weights"])
            / (edges.dst["exp_sum_attention"])
        )

        return {"attention_weights": attention_weights}

    def node_update(self, nodes):
        if self.config["output_model"]["attention_type"] == "softmax":
            attention_weights = torch.softmax(
                nodes.mailbox["attention"], dim=1
            ).unsqueeze(2)
            weighted_sum = torch.sum(attention_weights * nodes.mailbox["values"], dim=1)

        elif self.config["output_model"]["attention_type"] == "ReLA":
            attention_weights = nodes.mailbox["attention"].unsqueeze(2)

            weighted_sum = torch.sum(attention_weights * nodes.mailbox["values"], dim=1)
            weighted_sum = self.rmsnorm(weighted_sum) * nn.Sigmoid()(
                self.lin_weights(weighted_sum)
            )

        # attention_matrix = torch.cat((attention_weights.squeeze(2),1e-8*torch.ones(attention_weights.shape[0],200-attention_weights.shape[1]).to(attention_weights.device)),dim=1)
        # attention_matrix = attention_weights.view(attention_weights.shape[0],attention_weights.shape[1])
        new_node_features = nodes.data["features"] + self.mlp(
            self.layer_norm(self.gru(weighted_sum, nodes.data["features"]))
        )

        return {
            "features": new_node_features
        }  # , 'attention_weights': attention_matrix}

    def forward(self, g):
        self.norm = self.norm.to(g.device)

        nodes_inputs = torch.cat(
            [
                g.nodes["truth_particles"].data["features"],
                g.nodes["truth_particles"].data["features_0"],
            ],
            dim=1,
        )

        g.nodes["truth_particles"].data["key"] = self.key(nodes_inputs)
        g.nodes["truth_particles"].data["values"] = self.values(nodes_inputs)
        query_input = torch.cat(
            [
                g.nodes["fastsim_particles"].data["features"],
                g.nodes["fastsim_particles"].data["global_features"],
            ],
            dim=1,
        )
        g.nodes["fastsim_particles"].data["query"] = self.query(query_input)
        g.update_all(self.edge_function, self.node_update, etype="truth_to_fastsim")
