import uproot
import numpy as np
from tqdm import tqdm

import gc

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Sampler
import itertools
import math

import dgl
import dgl.function as fn
import torch.nn.functional as F
from dgl import DGLGraph


def normalize(inp):
    for i in range(len(inp)):
        while inp[i] > np.pi:
            inp[i] = inp[i] - 2 * np.pi
        while inp[i] < -np.pi:
            inp[i] = inp[i] + 2 * np.pi
    return inp


def rescale(var, mean, std):
    return (var - mean) / std


def collate_graphs(samples):
    batched_graphs = dgl.batch(samples[0])
    return batched_graphs


class FastsimSampler(Sampler):
    def __init__(self, nevents, batch_size, shuffle, n_replica):
        super().__init__(nevents)

        self.batch_size = batch_size
        self.shuffle = shuffle

        self.index_to_batch = {}
        self.n_replica = n_replica

        self.n_batches = nevents // (self.n_replica * self.batch_size)
        for i in range(nevents // (self.n_replica * self.batch_size)):
            self.index_to_batch[i] = np.arange(
                i * self.batch_size * self.n_replica,
                i * self.batch_size * self.n_replica + self.batch_size * self.n_replica,
            )

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        if self.shuffle:
            batch_order = np.random.permutation(np.arange(self.n_batches))
        else:
            batch_order = np.arange(self.n_batches)
        for i in batch_order:
            yield self.index_to_batch[i]


class FastSimDataset(Dataset):
    def __init__(self, filename, config=None, reduce_ds=1.0, entry_start=0):
        super().__init__()
        self.config = config

        self.entry_start = entry_start

        self.file = uproot.open(filename)
        self.tree = self.file["event_tree"]

        self.var_transform = self.config["var_transform"]

        self.nevents = self.tree.num_entries

        if reduce_ds < 1.0 and reduce_ds > 0:
            self.nevents = int(self.nevents * reduce_ds)
        if reduce_ds >= 1.0:
            self.nevents = reduce_ds
        print(" we have ", self.nevents, " events")

        # self.init_label_dicts()
        self.init_variables_list()

        self.full_data_array = {}

        print(entry_start, self.nevents + entry_start)

        for var in tqdm(self.truth_variables):
            self.full_data_array[var] = self.tree[var].array(
                library="np",
                entry_stop=self.nevents + entry_start,
                entry_start=entry_start,
            )

            if var == "truth_pt":
                self.n_truth_particles = [len(x) for x in self.full_data_array[var]]

            # flatten the arrays
            self.full_data_array[var] = np.concatenate(self.full_data_array[var])

            if var == "truth_phi":
                self.full_data_array[var] = normalize(self.full_data_array[var])

        for var in tqdm(self.pflow_variables):
            self.full_data_array[var] = self.tree[var].array(
                library="np",
                entry_stop=self.nevents + entry_start,
                entry_start=entry_start,
            )

            if var == "pflow_pt":
                self.n_pflow_particles = [len(x) for x in self.full_data_array[var]]

            # flatten the arrays
            self.full_data_array[var] = np.concatenate(self.full_data_array[var])

            if var == "pflow_phi":
                self.full_data_array[var] = normalize(self.full_data_array[var])

        # transform variables and transform to tensors
        for var in tqdm(self.full_data_array.keys()):
            if config["per_event_scaling"] == False:
                if var in self.var_transform:
                    self.full_data_array[var] = (
                        self.full_data_array[var] - self.var_transform[var]["mean"]
                    ) / self.var_transform[var]["std"]

            if var == "pflow_pt" or var == "truth_pt":
                self.full_data_array[var] = np.log(self.full_data_array[var] * 1000)

            self.full_data_array[var] = torch.tensor(self.full_data_array[var])

        self.truth_cumsum = np.cumsum([0] + self.n_truth_particles)
        self.pflow_cumsum = np.cumsum([0] + self.n_pflow_particles)

        self.file.close()
        del self.tree
        gc.collect()

        print("done loading data")

    def get_single_item(self, idx):
        # n_truth_particles = self.n_truth_particles[idx]
        # n_pflow_particles = self.n_pflow_particles[idx]
        # n_fastsim_particles = n_pflow_particles

        truth_start, truth_end = self.truth_cumsum[idx], self.truth_cumsum[idx + 1]
        pflow_start, pflow_end = self.pflow_cumsum[idx], self.pflow_cumsum[idx + 1]

        truth_class = self.full_data_array["truth_class"][truth_start:truth_end].long()
        pflow_class = self.full_data_array["pflow_class"][pflow_start:pflow_end].long()
        pflow_class[pflow_class < 3] = 1  # charged
        pflow_class[pflow_class > 2] = 0  # neutral
        truth_class[truth_class < 3] = 1
        truth_class[truth_class > 2] = 0

        ### REMOVE LATER

        # pflow_pt    = self.full_data_array['pflow_pt'][pflow_start:pflow_end]

        if self.config["charged_only"]:
            truth_mask = truth_class > 0
            pflow_mask = pflow_class > 0
        else:
            truth_mask = truth_class > -1  # always the case
            pflow_mask = pflow_class > -1
            # pflow_mask = pflow_pt > ( (np.log(4000) -self.config['var_transform']['particle_pt']['mean']) / self.config['var_transform']['particle_pt']['std'] )

        truth_pt = self.full_data_array["truth_pt"][truth_start:truth_end][truth_mask]
        truth_eta = self.full_data_array["truth_eta"][truth_start:truth_end][truth_mask]
        truth_phi = self.full_data_array["truth_phi"][truth_start:truth_end][truth_mask]
        truth_class = truth_class[truth_mask]

        pflow_pt = self.full_data_array["pflow_pt"][pflow_start:pflow_end][pflow_mask]
        # pflow_pt    = pflow_pt[pflow_mask]
        pflow_eta = self.full_data_array["pflow_eta"][pflow_start:pflow_end][pflow_mask]
        pflow_phi = self.full_data_array["pflow_phi"][pflow_start:pflow_end][pflow_mask]
        pflow_class = pflow_class[pflow_mask]

        n_truth_particles = len(truth_pt)
        n_pflow_particles = len(pflow_pt)
        n_fastsim_particles = n_pflow_particles

        # print(n_truth_particles, n_pflow_particles)

        num_nodes_dict = {
            "truth_particles": n_truth_particles,
            "pflow_particles": n_pflow_particles,
            "fastsim_particles": n_fastsim_particles,
            "global_node": 1,
        }
        truth_to_truth_edge_start = torch.arange(n_truth_particles).repeat(
            n_truth_particles
        )
        truth_to_truth_edge_end = torch.repeat_interleave(
            torch.arange(n_truth_particles), n_truth_particles
        )

        truth_to_fastsim_edge_start = torch.arange(n_truth_particles).repeat(
            n_fastsim_particles
        )
        truth_to_fastsim_edge_end = torch.repeat_interleave(
            torch.arange(n_fastsim_particles), n_truth_particles
        )

        fastsim_to_truth_edge_start = torch.arange(n_fastsim_particles).repeat(
            n_truth_particles
        )
        fastsim_to_truth_edge_end = torch.repeat_interleave(
            torch.arange(n_truth_particles), n_fastsim_particles
        )

        pflow_to_fastsim_edge_start = torch.arange(n_pflow_particles).repeat(
            n_fastsim_particles
        )
        pflow_to_fastsim_edge_end = torch.repeat_interleave(
            torch.arange(n_fastsim_particles), n_pflow_particles
        )

        data_dict = {
            ("truth_particles", "truth_to_truth", "truth_particles"): (
                truth_to_truth_edge_start,
                truth_to_truth_edge_end,
            ),
            ("truth_particles", "truth_to_global", "global_node"): (
                torch.arange(n_truth_particles).int(),
                torch.zeros(n_truth_particles).int(),
            ),
            ("truth_particles", "truth_to_fastsim", "fastsim_particles"): (
                truth_to_fastsim_edge_start,
                truth_to_fastsim_edge_end,
            ),
            ("fastsim_particles", "fastsim_to_truth", "truth_particles"): (
                fastsim_to_truth_edge_start,
                fastsim_to_truth_edge_end,
            ),
            ("pflow_particles", "pflow_to_fastsim", "fastsim_particles"): (
                pflow_to_fastsim_edge_start,
                pflow_to_fastsim_edge_end,
            ),
        }
        g = dgl.heterograph(data_dict, num_nodes_dict)

        if self.config["per_event_scaling"]:
            mean_pt = torch.mean(truth_pt)
            mean_eta = torch.mean(truth_eta)
            mean_phi = torch.mean(truth_phi)
            std_pt = torch.std(truth_pt)
            std_eta = torch.std(truth_eta)
            std_phi = torch.std(truth_phi)
            # mean_pt = torch.tensor(self.config['var_transform']['particle_pt']['mean'])
            # std_pt = torch.tensor(self.config['var_transform']['particle_pt']['std'])

            if torch.isnan(std_pt) or std_pt == 0:
                std_pt = torch.tensor(1).double()
            if torch.isnan(std_eta) or std_eta == 0:
                std_eta = torch.tensor(0.1).double()
            if torch.isnan(std_phi) or std_phi == 0:
                std_phi = torch.tensor(0.1).double()

            g.nodes["truth_particles"].data["idx"] = torch.arange(n_truth_particles)
            g.nodes["truth_particles"].data["class"] = truth_class
            g.nodes["truth_particles"].data["pt"] = rescale(
                truth_pt, mean_pt, std_pt
            ).float()
            g.nodes["truth_particles"].data["eta"] = rescale(
                truth_eta, mean_eta, std_eta
            ).float()
            g.nodes["truth_particles"].data["phi"] = rescale(
                truth_phi, mean_phi, std_phi
            ).float()

            g.nodes["pflow_particles"].data["idx"] = torch.arange(n_pflow_particles)
            g.nodes["pflow_particles"].data["class"] = pflow_class
            g.nodes["pflow_particles"].data["pt"] = rescale(
                pflow_pt, mean_pt, std_pt
            ).float()
            g.nodes["pflow_particles"].data["eta"] = rescale(
                pflow_eta, mean_eta, std_eta
            ).float()
            g.nodes["pflow_particles"].data["phi"] = rescale(
                pflow_phi, mean_phi, std_phi
            ).float()

            g.nodes["global_node"].data["mean_pt"] = mean_pt.unsqueeze(0)
            g.nodes["global_node"].data["std_pt"] = std_pt.unsqueeze(0)
            g.nodes["global_node"].data["mean_eta"] = mean_eta.unsqueeze(0)
            g.nodes["global_node"].data["std_eta"] = std_eta.unsqueeze(0)
            g.nodes["global_node"].data["mean_phi"] = mean_phi.unsqueeze(0)
            g.nodes["global_node"].data["std_phi"] = std_phi.unsqueeze(0)

            if torch.isnan(g.nodes["truth_particles"].data["eta"]).any():
                print(idx)
                print(g.nodes["truth_particles"].data["eta"])
                print(truth_eta)
                print(mean_eta)
                print(std_eta)

            if torch.isnan(g.nodes["pflow_particles"].data["pt"]).any():
                print(std_pt)

            global_feat_0 = torch.cat(
                [
                    mean_pt.unsqueeze(0),
                    std_pt.unsqueeze(0),
                    mean_eta.unsqueeze(0),
                    std_eta.unsqueeze(0),
                ]
            ).float()
            g.nodes["truth_particles"].data["event_feat_0"] = dgl.broadcast_nodes(
                g, global_feat_0, ntype="truth_particles"
            )

        else:
            g.nodes["truth_particles"].data["idx"] = torch.arange(n_truth_particles)
            g.nodes["truth_particles"].data["class"] = truth_class
            g.nodes["truth_particles"].data["pt"] = truth_pt
            g.nodes["truth_particles"].data["eta"] = truth_eta
            g.nodes["truth_particles"].data["phi"] = truth_phi

            g.nodes["pflow_particles"].data["idx"] = torch.arange(n_pflow_particles)
            g.nodes["pflow_particles"].data["class"] = pflow_class
            g.nodes["pflow_particles"].data["pt"] = pflow_pt
            g.nodes["pflow_particles"].data["eta"] = pflow_eta
            g.nodes["pflow_particles"].data["phi"] = pflow_phi

        return g

    def __len__(self):
        return self.nevents

    # def __getitem__(self, idx):
    #     return self.get_single_item(idx)

    def __getitem__(self, idxs):
        samples = [self.get_single_item(x) for x in idxs]
        return samples

    def init_label_dicts(self):
        # photon : 0
        # neutral hadron: n, pion0, K0, Xi0, lambda: 1
        # charged hadron: p+-, K+-, pion+-, Xi+, Omega, Sigma : 2
        # electron : 3
        # muon : 4

        self.class_labels = {
            -3112: 2,
            3112: 2,
            3222: 2,
            -3222: 2,
            -3334: 2,
            3334: 2,
            -3122: 1,
            3122: 1,
            310: 1,
            3312: 2,
            -3312: 2,
            3322: 1,
            -3322: 1,
            2112: 1,
            321: 2,
            130: 1,
            -2112: 1,
            2212: 2,
            11: 3,
            -211: 2,
            13: 4,
            211: 2,
            -13: 4,
            -11: 3,
            22: 0,
            -2212: 2,
            -321: 2,
        }

    def init_variables_list(self):
        self.truth_variables = ["truth_pt", "truth_eta", "truth_phi", "truth_class"]

        self.pflow_variables = ["pflow_pt", "pflow_eta", "pflow_phi", "pflow_class"]
