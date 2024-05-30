import dgl
import torch
import torch.nn as nn

from slotattention import SlotAttention
from torch.distributions import Normal


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


class TSPN(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        out_model = self.config["output_model"]

        self.mean = nn.Parameter(torch.zeros(out_model["z_size"]))
        self.sigma = nn.Parameter(torch.ones(out_model["z_size"]))

        self.m = Normal(self.mean, self.sigma)

        self.z_shape = out_model["z_size"]
        self.z_emb = torch.nn.Embedding(
            num_embeddings=out_model["max_particles"], embedding_dim=self.z_shape
        )

        self.slotattns = nn.ModuleList()
        for i in range(out_model["num_slotattn"]):
            self.slotattns.append(
                SlotAttention(
                    config["embedding_model"]["truth_hidden_size"],
                    self.z_shape,
                    self.config,
                )
            )

        self.set_size_embedding = torch.nn.Embedding(
            out_model["max_particles"], out_model["set_size_embedding"]
        )
        self.setsize_predictor = build_layers(
            config["embedding_model"]["truth_hidden_size"]
            + out_model["set_size_embedding"],
            out_model["max_particles"],
            features=out_model["set_size_prediction_layers"],
        )

        self.pt_eta_phi_net = build_layers(
            self.z_shape,
            3,
            features=out_model["ptetaphi_prediction_layers"],
            add_batch_norm=False,  # Need to check
        )

        self.class_net = build_layers(
            self.z_shape, outputsize=1, features=out_model["class_prediction_layers"]
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, g):
        if self.config["predict_set_size"]:
            input_set_size = torch.cat(
                [
                    g.nodes["global_node"].data["global_features"],
                    self.set_size_embedding(g.batch_num_nodes("truth_particles")),
                ],
                dim=1,
            )
            g.nodes["global_node"].data["set_size_pred"] = self.setsize_predictor(
                input_set_size
            )

        # will get it from set_size_prredictor later (for eval), for now we fix it for training stability
        n_fastsim_particles = g.batch_num_nodes("fastsim_particles")

        indices = torch.cat(
            [
                torch.linspace(0, N - 1, N, device=g.device).view(N).long()
                for N in n_fastsim_particles
            ],
            dim=0,
        )
        noise = torch.stack(
            [
                self.m.sample()
                for n in range(int(torch.sum(n_fastsim_particles).item()))
            ],
            dim=0,
        )
        Z = self.z_emb(indices) + noise
        g.nodes["fastsim_particles"].data["features"] = Z

        inputset_global = g.nodes["global_node"].data["global_features"]
        g.nodes["fastsim_particles"].data["global_features"] = dgl.broadcast_nodes(
            g, inputset_global, ntype="fastsim_particles"
        )

        for i, slotattn in enumerate(self.slotattns):
            slotattn(g)

        ndata = g.nodes["fastsim_particles"].data["features"]

        g.nodes["fastsim_particles"].data["pt_eta_phi_pred"] = self.pt_eta_phi_net(
            ndata
        )
        g.nodes["fastsim_particles"].data["class_pred"] = self.sigmoid(
            self.class_net(ndata)
        )
        return g

    def undo_scaling(self, particles):
        pass

    def infer(self, g):
        if self.config["predict_set_size"]:
            input_set_size = torch.cat(
                [
                    g.nodes["global_node"].data["global_features"],
                    self.set_size_embedding(g.batch_num_nodes("truth_particles")),
                ],
                dim=1,
            )
            g.nodes["global_node"].data["set_size_pred"] = self.setsize_predictor(
                input_set_size
            )
            n_fastsim_particles = torch.multinomial(
                torch.nn.Softmax()(g.nodes["global_node"].data["set_size_pred"]),
                1,
                replacement=True,
            ).squeeze(1)

        indices = torch.cat(
            [
                torch.linspace(0, N - 1, N, device=g.device).view(N).long()
                for N in n_fastsim_particles
            ],
            dim=0,
        )
        noise = torch.stack(
            [
                self.m.sample()
                for n in range(int(torch.sum(n_fastsim_particles).item()))
            ],
            dim=0,
        )
        Z = self.z_emb(indices) + noise

        n_truth_particles = g.batch_num_nodes(ntype="truth_particles").cpu()
        n_pflow_particles = g.batch_num_nodes(ntype="pflow_particles").cpu()

        pred_graph = []
        for i in range(n_truth_particles.size(0)):
            num_nodes_dict = {
                "truth_particles": n_truth_particles[i].cpu(),
                "fastsim_particles": n_fastsim_particles[i].cpu(),
                "pflow_particles": n_pflow_particles[i].cpu(),
                "global_node": 1,
            }

            truth_to_truth_edge_start = torch.arange(n_truth_particles[i]).repeat(
                n_truth_particles[i]
            )
            truth_to_truth_edge_end = torch.repeat_interleave(
                torch.arange(n_truth_particles[i]), n_truth_particles[i]
            )

            truth_to_fastsim_edge_start = torch.arange(n_truth_particles[i]).repeat(
                n_fastsim_particles[i]
            )
            truth_to_fastsim_edge_end = torch.repeat_interleave(
                torch.arange(n_fastsim_particles[i]), n_truth_particles[i]
            )

            data_dict = {
                ("truth_particles", "truth_to_truth", "truth_particles"): (
                    truth_to_truth_edge_start,
                    truth_to_truth_edge_end,
                ),
                ("truth_particles", "truth_to_fastsim", "fastsim_particles"): (
                    truth_to_fastsim_edge_start,
                    truth_to_fastsim_edge_end,
                ),
            }

            pred_graph.append(
                dgl.heterograph(data_dict, num_nodes_dict, device=g.device)
            )

        gp = dgl.batch(pred_graph)

        gp.nodes["truth_particles"].data["features"] = g.nodes["truth_particles"].data[
            "features"
        ]
        gp.nodes["truth_particles"].data["features_0"] = g.nodes[
            "truth_particles"
        ].data["features_0"]
        gp.nodes["truth_particles"].data["class"] = g.nodes["truth_particles"].data[
            "class"
        ]

        gp.nodes["pflow_particles"].data["pt"] = g.nodes["pflow_particles"].data["pt"]
        gp.nodes["pflow_particles"].data["eta"] = g.nodes["pflow_particles"].data["eta"]
        gp.nodes["pflow_particles"].data["phi"] = g.nodes["pflow_particles"].data["phi"]
        gp.nodes["pflow_particles"].data["class"] = g.nodes["pflow_particles"].data[
            "class"
        ]

        if self.config["per_event_scaling"]:
            gp.nodes["global_node"].data["pt_mean"] = g.nodes["global_node"].data[
                "mean_pt"
            ]
            gp.nodes["global_node"].data["pt_std"] = g.nodes["global_node"].data[
                "std_pt"
            ]
            gp.nodes["global_node"].data["eta_mean"] = g.nodes["global_node"].data[
                "mean_eta"
            ]
            gp.nodes["global_node"].data["eta_std"] = g.nodes["global_node"].data[
                "std_eta"
            ]
            gp.nodes["global_node"].data["phi_mean"] = g.nodes["global_node"].data[
                "mean_phi"
            ]
            gp.nodes["global_node"].data["phi_std"] = g.nodes["global_node"].data[
                "std_phi"
            ]

        gp.nodes["fastsim_particles"].data["features"] = Z

        inputset_global = g.nodes["global_node"].data["global_features"]
        gp.nodes["fastsim_particles"].data["global_features"] = dgl.broadcast_nodes(
            gp, inputset_global, ntype="fastsim_particles"
        )

        for i, slotattn in enumerate(self.slotattns):
            slotattn(gp)

        ndata = gp.nodes["fastsim_particles"].data["features"]

        gp.nodes["fastsim_particles"].data["pt_eta_phi_pred"] = self.pt_eta_phi_net(
            ndata
        )
        gp.nodes["fastsim_particles"].data["class_pred"] = self.sigmoid(
            self.class_net(ndata)
        )

        return gp
