import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from scipy.optimize import linear_sum_assignment


class Set2SetLoss_single(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config
        self.var_transform = self.config["var_transform"]

        self.class_loss = nn.BCELoss(reduction="none")
        self.train_class = self.config.get("learn_class", False)
        print("train_class ", self.train_class)

        self.regression_loss = nn.MSELoss(reduction="none")
        self.num_loss = nn.CrossEntropyLoss(reduction="mean")

    def forward(self, g, scatter=True):
        ### new code !!!
        n_pflow = g.batch_num_nodes("pflow_particles")
        pflow_data = g.nodes["pflow_particles"].data
        pflow_pt = torch.split(pflow_data["pt"].cpu(), n_pflow.tolist())
        pflow_eta = torch.split(pflow_data["eta"].cpu(), n_pflow.tolist())
        pflow_phi = torch.split(pflow_data["phi"].cpu(), n_pflow.tolist())
        if self.train_class:
            pflow_class = torch.split(pflow_data["class"], n_pflow.tolist())

        fastsim_data = g.nodes["fastsim_particles"].data
        fastsim_pt_eta_phi = torch.split(
            fastsim_data["pt_eta_phi_pred"].cpu(), n_pflow.tolist()
        )
        if self.train_class:
            fastsim_class = torch.split(
                fastsim_data["class_pred"].cpu(), n_pflow.tolist()
            )
        max_len = n_pflow.max()

        input = torch.zeros((g.batch_size, max_len, 3))  # , device=pflow_pt[0].device)
        target = torch.zeros((g.batch_size, max_len, 3))  # , device=pflow_pt[0].device)

        if self.train_class:
            input_class = torch.zeros(
                (g.batch_size, max_len)
            )  # , device=pflow_pt[0].device)
            target_class = torch.zeros(
                (g.batch_size, max_len)
            )  # , device=pflow_pt[0].device)
        mask = torch.zeros((g.batch_size, max_len), dtype=torch.bool)

        bs = g.batch_size

        for i in range(bs):
            target[i, : n_pflow[i], 0] = pflow_pt[i]
            target[i, : n_pflow[i], 1] = pflow_eta[i]
            target[i, : n_pflow[i], 2] = pflow_phi[i]

            mask[i, : n_pflow[i]] = True

            input[i, : n_pflow[i], :] = fastsim_pt_eta_phi[i]
            if self.train_class:
                input_class[i, : n_pflow[i]] = fastsim_class[i][:, 0]
                target_class[i, : n_pflow[i]] = pflow_class[i]

        new_input = input.unsqueeze(1).expand(-1, target.size(1), -1, -1)
        new_target = target.unsqueeze(2).expand(-1, -1, input.size(1), -1)

        mask_input = mask.unsqueeze(1).expand(-1, target.size(1), -1)
        mask_target = mask.unsqueeze(2).expand(-1, -1, input.size(1))

        if self.train_class:
            new_input_class = input_class.unsqueeze(1).expand(-1, target.size(1), -1)
            new_target_class = target_class.unsqueeze(2).expand(-1, -1, input.size(1))
            pdist_class = self.class_loss(new_input_class, new_target_class)
        else:
            pdist_class = 0
        pdist_pt = F.mse_loss(new_input, new_target, reduction="none")
        pdist_eta = F.mse_loss(new_input, new_target, reduction="none")

        pdist_phi = (
            2
            * (
                1
                - torch.cos(
                    (new_input - new_target)
                    * g.nodes["global_node"].data["std_phi"][0].cpu()
                )
            )
            * 10
        )

        pt_mask = [x for x in range(target.size(-1)) if (x + 3) % 3 == 0]
        eta_mask = [x for x in range(target.size(-1)) if (x + 2) % 3 == 0]
        phi_mask = [x for x in range(target.size(-1)) if (x + 1) % 3 == 0]

        pdist_ptetaphi = torch.cat(
            [
                pdist_pt[:, :, :, pt_mask],
                pdist_eta[:, :, :, eta_mask],
                pdist_phi[:, :, :, phi_mask],
            ],
            dim=-1,
        )

        pdist_ptetaphi = pdist_ptetaphi.mean(3)
        pdist = pdist_ptetaphi + pdist_class

        # set to 0 if both are fake
        mask_both0 = torch.logical_not(torch.logical_or(mask_input, mask_target))
        pdist[mask_both0] = 0

        # set to high value if only one is fake
        mask_one0 = torch.logical_xor(mask_input, mask_target)
        pdist[mask_one0] = 1e4

        pdist_ = pdist.detach().cpu().numpy()
        indices = np.array([linear_sum_assignment(p) for p in pdist_])

        indices = indices.shape[2] * indices[:, 0] + indices[:, 1]

        losses = (
            torch.gather(
                pdist.flatten(1, 2),
                1,
                torch.from_numpy(indices).to(device=pdist.device),
            )
            .mean(1)
            .to(device=g.device)
        )

        num_loss = (
            self.num_loss(
                g.nodes["global_node"].data["set_size_pred"],
                g.batch_num_nodes("pflow_particles"),
            ).mean()
            / 1.5
        )

        total_loss = losses.mean() + num_loss

        return {
            "total_loss": total_loss.to(device=g.device),
            "num_loss": num_loss.item(),
        }
