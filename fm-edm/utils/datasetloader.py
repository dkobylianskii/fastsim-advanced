import gc

import numpy as np
import torch
import uproot
from torch.utils.data import Dataset
from tqdm import tqdm


###  0: charged hadrons
###  1: electrons
###  2: muons
###  3: neutral hadrons
###  4: photons
###  5: residual
### -1: neutrinos


def normalize(inp):
    return np.arctan2(np.sin(inp), np.cos(inp))


def rescale(var, mean, std):
    return (var - mean) / std


class FastSimDataset(Dataset):
    def __init__(self, filename, config=None, reduce_ds=1.0, entry_start=0):
        super().__init__()
        self.config = config

        self.file = uproot.open(filename)
        self.tree = self.file["event_tree"]

        self.max_particles = self.config["max_particles"]
        self.train_class = self.config.get("train_class", False)

        self.nevents = self.tree.num_entries
        self.entry_start = entry_start
        if reduce_ds < 1.0 and reduce_ds > 0:
            self.nevents = int(self.nevents * reduce_ds)
        if reduce_ds >= 1.0:
            self.nevents = reduce_ds
        print(" we have ", self.nevents, " events")

        self.init_variables_list()

        self.full_data_array = {}

        for var in tqdm(self.truth_variables):
            self.full_data_array[var] = self.tree[var].array(
                library="np",
                entry_stop=self.nevents + self.entry_start,
                entry_start=self.entry_start,
            )
            if var == "truth_pt":
                self.n_truth_particles = [len(x) for x in self.full_data_array[var]]

            # flatten the arrays
            self.full_data_array[var] = np.concatenate(
                self.full_data_array[var]
            )  # [mask]
            if var == "truth_phi":
                self.full_data_array[var] = normalize(self.full_data_array[var])

        for var in tqdm(self.pflow_variables):
            self.full_data_array[var] = self.tree[var].array(
                library="np",
                entry_stop=self.nevents + self.entry_start,
                entry_start=self.entry_start,
            )
            if var == "pflow_eta":
                self.n_pflow_particles = [len(x) for x in self.full_data_array[var]]

            self.full_data_array[var] = np.concatenate(self.full_data_array[var])
            if var == "pflow_eta":
                self.full_data_array[var] = np.clip(self.full_data_array[var], -3, 3)
            if var == "pflow_phi":
                self.full_data_array[var] = normalize(self.full_data_array[var])

        # transform variables and transform to tensors
        for var in tqdm(self.full_data_array.keys()):
            if var == "pflow_pt" or var == "truth_pt":
                self.full_data_array[var] = np.log(self.full_data_array[var] * 1000)
            self.full_data_array[var] = torch.tensor(self.full_data_array[var])

        self.truth_cumsum = np.cumsum([0] + self.n_truth_particles)
        self.pflow_cumsum = np.cumsum([0] + self.n_pflow_particles)
        self.file.close()

        del self.tree
        gc.collect()
        if self.config.get("charged_only", False):
            print("Use charged only")
        print("done loading data")

    def get_single_item(self, idx):
        n_truth_particles = self.n_truth_particles[idx]
        n_pflow_particles = self.n_pflow_particles[idx]

        truth_start, truth_end = self.truth_cumsum[idx], self.truth_cumsum[idx + 1]
        pflow_start, pflow_end = self.pflow_cumsum[idx], self.pflow_cumsum[idx + 1]

        truth_pt = self.full_data_array["truth_pt"][truth_start:truth_end]
        truth_eta = self.full_data_array["truth_eta"][truth_start:truth_end]
        truth_phi = self.full_data_array["truth_phi"][truth_start:truth_end]
        truth_class = self.full_data_array["truth_class"][truth_start:truth_end].long()

        pflow_pt = self.full_data_array["pflow_pt"][pflow_start:pflow_end]
        pflow_eta = self.full_data_array["pflow_eta"][pflow_start:pflow_end]
        pflow_phi = self.full_data_array["pflow_phi"][pflow_start:pflow_end]

        pflow_class = self.full_data_array["pflow_class"][pflow_start:pflow_end].long()

        if self.config.get("charged_only", False):
            truth_mask = truth_class < 3
            pflow_mask = pflow_class < 3
            truth_pt = truth_pt[truth_mask]
            truth_eta = truth_eta[truth_mask]
            truth_phi = truth_phi[truth_mask]
            truth_e = truth_e[truth_mask]
            truth_class = truth_class[truth_mask]

            pflow_pt = pflow_pt[pflow_mask]
            pflow_eta = pflow_eta[pflow_mask]
            pflow_phi = pflow_phi[pflow_mask]
            pflow_class = pflow_class[pflow_mask]

            n_truth_particles = truth_mask.sum()
            n_pflow_particles = pflow_mask.sum()

        # calculate mean, std
        truth_pt_mean = torch.mean(truth_pt)
        truth_eta_mean = torch.mean(truth_eta)
        truth_phi_mean = torch.mean(truth_phi)

        if n_truth_particles == 0:
            truth_pt_mean = 0
            truth_eta_mean = 0
            truth_phi_mean = 0

        if n_truth_particles < 2:
            truth_pt_std = torch.tensor(1).float()
            truth_eta_std = torch.tensor(0.1).float()
            truth_phi_std = torch.tensor(0.1).float()
        else:
            truth_pt_std = torch.std(truth_pt)
            truth_eta_std = torch.std(truth_eta)
            truth_phi_std = torch.std(truth_phi)
            if truth_pt_std == 0:
                truth_pt_std = torch.tensor(1).float()
            if truth_eta_std == 0:
                truth_eta_std = torch.tensor(0.1).float()
            if truth_phi_std == 0:
                truth_phi_std = torch.tensor(0.1).float()

        truth_data = torch.zeros(self.max_particles, 4)
        if self.train_class:
            pflow_data = torch.zeros(self.max_particles, 4)
        else:
            pflow_data = torch.zeros(self.max_particles, 3)

        truth_idx = torch.argsort(truth_pt, descending=True)
        pflow_idx = torch.argsort(pflow_pt, descending=True)

        truth_data[:n_truth_particles, 0] = rescale(
            truth_pt[truth_idx], truth_pt_mean, truth_pt_std
        ).float()
        truth_data[:n_truth_particles, 1] = rescale(
            truth_eta[truth_idx], truth_eta_mean, truth_eta_std
        ).float()
        truth_data[:n_truth_particles, 2] = rescale(
            truth_phi[truth_idx], truth_phi_mean, truth_phi_std
        ).float()
        tmp = truth_class[truth_idx]
        tmp[tmp <= 2] = 1
        tmp[tmp > 2] = 0
        truth_data[:n_truth_particles, 3] = tmp
        # truth_data[:n_truth_particles, 4] = 1

        pflow_data[:n_pflow_particles, 0] = rescale(
            pflow_pt[pflow_idx], truth_pt_mean, truth_pt_std
        ).float()
        pflow_data[:n_pflow_particles, 1] = rescale(
            pflow_eta[pflow_idx], truth_eta_mean, truth_eta_std
        ).float()
        pflow_data[:n_pflow_particles, 2] = rescale(
            pflow_phi[pflow_idx], truth_phi_mean, truth_phi_std
        ).float()

        if self.train_class:
            tmp = pflow_class[pflow_idx]
            tmp[tmp <= 2] = 1
            tmp[tmp > 2] = 0
            pflow_data[:n_pflow_particles, 3] = tmp.float() * 2 - 1

        mask = torch.zeros(self.max_particles, 2)
        mask[:n_truth_particles, 0] = 1
        mask[:n_pflow_particles, 1] = 1
        mask = mask.bool()

        truth_data = truth_data.to(torch.float32)
        pflow_data = pflow_data.to(torch.float32)

        scale_data = torch.tensor(
            [
                truth_pt_mean,
                truth_pt_std,
                truth_eta_mean,
                truth_eta_std,
                truth_phi_mean,
                truth_phi_std,
            ]
        ).to(torch.float32)

        if n_truth_particles == 0:
            truth_data = torch.zeros(self.max_particles, 4).to(torch.float32)
            pflow_data = torch.zeros(self.max_particles, 3).to(torch.float32)
            mask = torch.zeros(self.max_particles, 2, dtype=bool)
            scale_data = torch.zeros(6).to(torch.float32)

        return truth_data, pflow_data, mask, scale_data

    def extract_filter_data(
        self, pt, eta, phi, pt_mean, eta_mean, phi_mean, pt_std, eta_std, phi_std
    ):
        data = torch.zeros(self.max_particles // 2, 3)
        mask = torch.zeros(self.max_particles // 2)
        n_particles = len(pt)
        if n_particles == 0:
            return data, mask.bool()
        idx = torch.argsort(pt, descending=True)
        data[:n_particles, 0] = rescale(pt[idx], pt_mean, pt_std).float()
        data[:n_particles, 1] = rescale(eta[idx], eta_mean, eta_std).float()
        data[:n_particles, 2] = rescale(phi[idx], phi_mean, phi_std).float()
        mask[:n_particles] = 1
        return data, mask.bool()

    def __len__(self):
        return len(self.n_truth_particles)

    def __getitem__(self, idx):
        return self.get_single_item(idx)

    def init_variables_list(self):
        self.truth_variables = [
            "truth_pt",
            "truth_eta",
            "truth_phi",
            "truth_class",
        ]

        self.pflow_variables = [
            "pflow_pt",
            "pflow_class",
            "pflow_eta",
            "pflow_phi",
        ]
