import argparse

import os
import json
import torch
import numpy as np
from tqdm import tqdm
import os
import uproot
import awkward as ak
from torch.utils.data import DataLoader
import dgl

from lightning import FastSimLightning
from datasetloader import FastSimDataset, collate_graphs, FastsimSampler

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", type=str, required=True)
parser.add_argument("-p", "--checkpoint", type=str, required=True)
parser.add_argument("-g", "--gpu", type=int, default=0)
parser.add_argument("-e", "--eval_dir", type=str, default="evals")
parser.add_argument("-ne", "--num_events", type=int, default=100_00)
parser.add_argument("-bs", "--batch_size", type=int, default=50)
parser.add_argument("--test_path", type=str, default=None)
parser.add_argument("--prefix", type=str, default="")
args = parser.parse_args()


def reshape_phi(phi):
    for i in range(len(phi)):
        while phi[i] > np.pi:
            phi[i] -= 2 * np.pi
        while phi[i] < -np.pi:
            phi[i] += 2 * np.pi
    return phi


os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)


with open(args.config, "r") as fp:
    config = json.load(fp)


eval_dir = config["eval_dir"]

pT_mean, pT_std = (
    config["var_transform"]["particle_pt"]["mean"],
    config["var_transform"]["particle_pt"]["std"],
)
eta_mean, eta_std = (
    config["var_transform"]["particle_eta"]["mean"],
    config["var_transform"]["particle_eta"]["std"],
)
phi_mean, phi_std = (
    config["var_transform"]["particle_phi"]["mean"],
    config["var_transform"]["particle_phi"]["std"],
)

if os.path.exists(f"{args.eval_dir}") is False:
    os.makedirs(f"{args.eval_dir}")
eval_path = f"{args.eval_dir}/{args.prefix}{config['name']}.root"

net = FastSimLightning(config)

checkpoint = torch.load(args.checkpoint, map_location=torch.device("cpu"))
net.load_state_dict(checkpoint["state_dict"])

torch.set_grad_enabled(False)
net.eval()

test_path = args.test_path if args.test_path is not None else config["path_test"]

dataset = FastSimDataset(
    test_path,
    config,
    entry_start=2_000_000 if "1rep" in args.prefix else 0,
    reduce_ds=args.num_events,
)
batch_sampler = FastsimSampler(
    len(dataset), batch_size=args.batch_size, shuffle=False, n_replica=100
)

loader = DataLoader(
    dataset,
    num_workers=10,
    sampler=batch_sampler,
    collate_fn=collate_graphs,
    pin_memory=False,
)

net.net.cuda()
net.cuda()
device = torch.device("cuda")

l_pt_tr, l_pt_pf, l_pt_fs = [], [], []
l_eta_tr, l_eta_pf, l_eta_fs = [], [], []
l_phi_tr, l_phi_pf, l_phi_fs = [], [], []
l_ind_tr, l_ind_pf, l_ind_fs = [], [], []
l_class_tr, l_class_pf, l_class_fs = [], [], []

for g in tqdm(loader):
    g = g.to(device)
    g = net.net.infer(g)
    g_list = dgl.unbatch(g)

    for i in range(len(g_list)):
        pT_mean = g_list[i].nodes["global_node"].data["pt_mean"].cpu().numpy()
        pT_std = g_list[i].nodes["global_node"].data["pt_std"].cpu().numpy()
        eta_mean = g_list[i].nodes["global_node"].data["eta_mean"].cpu().numpy()
        eta_std = g_list[i].nodes["global_node"].data["eta_std"].cpu().numpy()
        phi_mean = g_list[i].nodes["global_node"].data["phi_mean"].cpu().numpy()
        phi_std = g_list[i].nodes["global_node"].data["phi_std"].cpu().numpy()

        pt_fs = np.exp(
            g_list[i]
            .nodes["fastsim_particles"]
            .data["pt_eta_phi_pred"][:, 0]
            .cpu()
            .numpy()
            * pT_std
            + pT_mean
        )
        eta_fs = (
            g_list[i]
            .nodes["fastsim_particles"]
            .data["pt_eta_phi_pred"][:, 1]
            .cpu()
            .numpy()
            * eta_std
            + eta_mean
        )
        phi_fs = (
            g_list[i]
            .nodes["fastsim_particles"]
            .data["pt_eta_phi_pred"][:, 2]
            .cpu()
            .numpy()
            * phi_std
            + phi_mean
        )

        class_fs = (
            g_list[i]
            .nodes["fastsim_particles"]
            .data["class_pred"]
            .cpu()
            .numpy()
            .squeeze(1)
        )
        # class_fs = torch.multinomial(torch.nn.Softmax()(class_fs),1,replacement=True).squeeze(1).detach().cpu().numpy()

        pt_pf = np.exp(
            g_list[i].nodes["pflow_particles"].data["pt"].cpu().numpy() * pT_std
            + pT_mean
        )
        eta_pf = (
            g_list[i].nodes["pflow_particles"].data["eta"].cpu().numpy() * eta_std
            + eta_mean
        )
        phi_pf = (
            g_list[i].nodes["pflow_particles"].data["phi"].cpu().numpy() * phi_std
            + phi_mean
        )

        class_pf = g_list[i].nodes["pflow_particles"].data["class"].cpu().numpy()

        pt_tr = np.exp(
            g_list[i].nodes["truth_particles"].data["features_0"][:, 0].cpu().numpy()
            * pT_std
            + pT_mean
        )
        eta_tr = (
            g_list[i].nodes["truth_particles"].data["features_0"][:, 1].cpu().numpy()
            * eta_std
            + eta_mean
        )
        phi_tr = (
            g_list[i].nodes["truth_particles"].data["features_0"][:, 2].cpu().numpy()
            * phi_std
            + phi_mean
        )

        class_tr = g_list[i].nodes["truth_particles"].data["class"].cpu().numpy()

        l_pt_fs.extend([pt_fs.tolist()])
        l_pt_pf.extend([pt_pf.tolist()])
        l_pt_tr.extend([pt_tr.tolist()])

        l_eta_fs.extend([eta_fs.tolist()])
        l_eta_pf.extend([eta_pf.tolist()])
        l_eta_tr.extend([eta_tr.tolist()])

        l_phi_fs.extend([phi_fs.tolist()])
        l_phi_pf.extend([phi_pf.tolist()])
        l_phi_tr.extend([phi_tr.tolist()])

        l_class_fs.extend([class_fs.tolist()])
        l_class_pf.extend([class_pf.tolist()])
        l_class_tr.extend([class_tr.tolist()])


with uproot.recreate(eval_path) as file:
    file["truth_tree"] = {
        "tr_pt": ak.Array(l_pt_tr),
        "tr_eta": ak.Array(l_eta_tr),
        "tr_phi": ak.Array(l_phi_tr),
        "tr_class": ak.Array(l_class_tr),
    }

    file["pflow_tree"] = {
        "pf_pt": ak.Array(l_pt_pf),
        "pf_eta": ak.Array(l_eta_pf),
        "pf_phi": ak.Array(l_phi_pf),
        "pf_class": ak.Array(l_class_pf),
    }

    file["fastsim_tree"] = {
        "fs_pt": ak.Array(l_pt_fs),
        "fs_eta": ak.Array(l_eta_fs),
        "fs_phi": ak.Array(l_phi_fs),
        "fs_class": ak.Array(l_class_fs),
    }
