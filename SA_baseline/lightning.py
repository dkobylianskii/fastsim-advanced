import comet_ml
from pytorch_lightning.core.module import LightningModule
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import sys

sys.path.append("./models/")

from models.fastsim_model import FastSimModel
from loss_set2set import Set2SetLoss_single as Set2SetLoss

from datasetloader import FastsimSampler, FastSimDataset, collate_graphs
import numpy as np

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt

plt.style.use("seaborn-whitegrid")
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import gaussian_kde


class FastSimLightning(LightningModule):
    def __init__(self, config, comet_exp=None):
        super().__init__()
        torch.manual_seed(1)
        self.config = config
        self.net = FastSimModel(self.config)

        if config["model_type"] == "tspn":
            self.loss = Set2SetLoss(config)

        self.comet_exp = comet_exp

    def set_comet_exp(self, comet_exp):
        self.comet_exp = comet_exp

    def forward(self, g):
        return self.net(g)

    def training_step(self, batch, batch_idx):
        g = batch
        self(g)
        losses = self.loss(g)

        # self.log('val/loss', losses['loss'])
        for loss_type in self.config["loss_types"]:
            self.log("train/" + loss_type, losses[loss_type])

        return losses["total_loss"]

    def validation_step(self, batch, batch_idx):
        g = batch
        self(g)
        losses = self.loss(g)

        return_dict = {
            "val_loss": losses["total_loss"],
            "unmatched": (  # truth, pflow, fastsim
                torch.cat(
                    [
                        g.nodes["truth_particles"].data["pt"].unsqueeze(1),
                        g.nodes["truth_particles"].data["eta"].unsqueeze(1),
                        g.nodes["truth_particles"].data["phi"].unsqueeze(1),
                    ],
                    dim=1,
                )
                .cpu()
                .numpy(),
                torch.cat(
                    [
                        g.nodes["pflow_particles"].data["pt"].unsqueeze(1),
                        g.nodes["pflow_particles"].data["eta"].unsqueeze(1),
                        g.nodes["pflow_particles"].data["phi"].unsqueeze(1),
                    ],
                    dim=1,
                )
                .cpu()
                .numpy(),
                g.nodes["fastsim_particles"].data["pt_eta_phi_pred"].cpu().numpy(),
            ),
            "n particles": (
                torch.cumsum(
                    torch.cat(
                        [
                            torch.tensor([0], device=g.device),
                            g.batch_num_nodes("truth_particles"),
                        ],
                        dim=0,
                    ),
                    dim=0,
                ),
                torch.cumsum(
                    torch.cat(
                        [
                            torch.tensor([0], device=g.device),
                            g.batch_num_nodes("pflow_particles"),
                        ],
                        dim=0,
                    ),
                    dim=0,
                ),
                torch.cumsum(
                    torch.cat(
                        [
                            torch.tensor([0], device=g.device),
                            g.batch_num_nodes("fastsim_particles"),
                        ],
                        dim=0,
                    ),
                    dim=0,
                ),
            ),
        }

        for loss_type in self.config["loss_types"]:
            self.log(
                "val/" + loss_type,
                losses[loss_type],
                batch_size=self.config["batchsize"],
            )
            return_dict[loss_type] = losses[loss_type]

        return return_dict

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config["learningrate"])
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[100], gamma=0.5
        )

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def train_dataloader(self):
        if "reduce_ds_train" in self.config:
            reduce_ds = self.config["reduce_ds_train"]
        else:
            reduce_ds = 1
        dataset = FastSimDataset(
            self.config["path_train"], self.config, reduce_ds=reduce_ds
        )
        batch_sampler = FastsimSampler(
            len(dataset),
            batch_size=self.config["batchsize"],
            shuffle=True,
            n_replica=self.config["n_replica"],
        )
        loader = DataLoader(
            dataset,
            num_workers=self.config["num_workers"],
            sampler=batch_sampler,
            collate_fn=collate_graphs,
            pin_memory=False,
        )
        return loader

    def val_dataloader(self):
        if "reduce_ds_val" in self.config:
            reduce_ds = self.config["reduce_ds_val"]
        else:
            reduce_ds = 1

        dataset = FastSimDataset(
            self.config["path_valid"],
            self.config,
            reduce_ds=reduce_ds,
            entry_start=self.config["reduce_ds_train"],
        )
        batch_sampler = FastsimSampler(
            len(dataset),
            batch_size=self.config["batchsize"],
            shuffle=False,
            n_replica=self.config["n_replica"],
        )
        loader = DataLoader(
            dataset,
            num_workers=self.config["num_workers"],
            sampler=batch_sampler,
            collate_fn=collate_graphs,
            pin_memory=False,
        )
        return loader

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        self.log("val/avg-loss epoch", avg_loss)

        if plt.get_fignums():
            plt.clf()
            fig = plt.gcf()
        else:
            fig = plt.figure(figsize=(10, 15), dpi=100, tight_layout=True)
        canvas = FigureCanvas(fig)

        axs = [None] * 12

        for i, out in enumerate(outputs):
            if i < 4:
                n_truth, n_pflow, n_fastsim = out["n particles"]

                truth_pt = out["unmatched"][0][n_truth[i] : n_truth[i + 1], 0]
                pflow_pt = out["unmatched"][1][n_pflow[i] : n_pflow[i + 1], 0]
                fastsim_pt = out["unmatched"][2][n_fastsim[i] : n_fastsim[i + 1], 0]

                truth_eta = out["unmatched"][0][n_truth[i] : n_truth[i + 1], 1]
                pflow_eta = out["unmatched"][1][n_pflow[i] : n_pflow[i + 1], 1]
                fastsim_eta = out["unmatched"][2][n_fastsim[i] : n_fastsim[i + 1], 1]

                truth_phi = out["unmatched"][0][n_truth[i] : n_truth[i + 1], 2]
                pflow_phi = out["unmatched"][1][n_pflow[i] : n_pflow[i + 1], 2]
                fastsim_phi = out["unmatched"][2][n_fastsim[i] : n_fastsim[i + 1], 2]

                axs[(i * 3)] = fig.add_subplot(4, 3, (i * 3) + 1)
                axs[(i * 3)].scatter(pflow_pt, pflow_eta, label="pflow")
                axs[(i * 3)].scatter(fastsim_pt, fastsim_eta, label="fastsim")
                axs[(i * 3)].scatter(truth_pt, truth_eta, c="k", alpha=0.7, marker="^")
                axs[(i * 3)].set_xlabel(r"$p_T$")
                axs[(i * 3)].set_ylabel(r"$\eta$")
                axs[(i * 3)].legend()

                # scatter plot phi vs eta

                axs[(i * 3) + 1] = fig.add_subplot(4, 3, (i * 3) + 2)
                axs[(i * 3) + 1].scatter(pflow_eta, pflow_phi, label="pflow")
                axs[(i * 3) + 1].scatter(fastsim_eta, fastsim_phi, label="fastsim")
                axs[(i * 3) + 1].scatter(
                    truth_eta, truth_phi, c="k", alpha=0.7, marker="^"
                )
                axs[(i * 3) + 1].set_xlabel(r"scaled $\eta$")
                axs[(i * 3) + 1].set_ylabel(r"scaled $\phi$")
                axs[(i * 3) + 1].legend()

        canvas.draw()
        w, h = fig.get_size_inches() * fig.get_dpi()
        image = np.fromstring(canvas.tostring_rgb(), dtype="uint8").reshape(
            int(h), int(w), 3
        )

        if self.comet_exp is not None:
            self.comet_exp.log_image(
                image_data=image,
                name="truth vs reco scatter",
                overwrite=False,
                image_format="png",
            )
        else:
            plt.savefig("scatter.png")

    def fancy_scatter(self, fig, ax, x, y):
        xy = np.vstack([x, y])
        try:
            z = gaussian_kde(xy)(xy)
            idx = z.argsort()
            x, y, z = x[idx], y[idx], z[idx]
            im = ax.scatter(x, y, s=3, c=z, cmap="cool")
        except:
            im = ax.scatter(x, y, s=3)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax, orientation="vertical")

        if len(x) > 0 and len(y) > 0:
            max_val = max(np.max(x), np.max(y))
            min_val = min(np.min(x), np.min(y))
            ax.set_xlim(min_val, max_val)
            ax.set_ylim(min_val, max_val)
