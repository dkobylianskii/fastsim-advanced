import torch
import torch.nn as nn
from modules import DenseNetwork, SineCosineEncoding, TimestepEmbedder
from transformer import TransformerEncoder, TransformerCrossAttentionLayer


class FlowNetSimple(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        hidden_dim = config["hidden_dim"]
        num_heads = config["num_heads"]
        num_ca_layers = config["num_ca_layers"]

        act = config["act"]
        self.hidden_dim = hidden_dim
        self.act = act

        self.use_pos_embd = config["use_pos_embd"]
        self.use_global = config.get("use_global", False)
        self.train_npf = config.get("train_npf", False)
        self.use_prev = config.get("use_prev", False)
        self.train_class = config.get("train_class", False)

        self.time_embedding = TimestepEmbedder(hidden_dim)

        if self.use_pos_embd:
            self.pos_embedding = SineCosineEncoding(
                hidden_dim // 4, min_value=0, max_value=256
            )

        self.truth_init = DenseNetwork(
            inpt_dim=4 + (hidden_dim // 4 if self.use_pos_embd else 0),
            outp_dim=hidden_dim,
            hddn_dim=[hidden_dim, hidden_dim],
            act_h=act,
        )
        fs_data_dim = 3 + self.train_class
        self.fs_init = DenseNetwork(
            inpt_dim=fs_data_dim
            + (hidden_dim // 4 if self.use_pos_embd else 0)
            + fs_data_dim * self.use_prev,
            outp_dim=hidden_dim,
            hddn_dim=[hidden_dim, hidden_dim],
            act_h=act,
            ctxt_dim=2 * hidden_dim,
        )
        if self.use_global:
            self.global_embedding = DenseNetwork(
                inpt_dim=6,
                outp_dim=hidden_dim,
                hddn_dim=[hidden_dim, hidden_dim],
                act_h=act,
            )

        self.truth_embedding = TransformerEncoder(
            model_dim=hidden_dim,
            num_layers=2,
            mha_config={
                "num_heads": num_heads,
                "new_mask": True,
            },
            dense_config={
                "act_h": act,
                "hddn_dim": 2 * hidden_dim,
            },
        )

        self.ca_blocks = nn.ModuleList(
            [
                TransformerCrossAttentionLayer(
                    model_dim=hidden_dim,
                    mha_config={
                        "num_heads": num_heads,
                        "new_mask": True,
                    },
                    dense_config={
                        "act_h": act,
                        "hddn_dim": 2 * hidden_dim,
                    },
                    ctxt_dim=2 * hidden_dim,
                )
                for _ in range(num_ca_layers)
            ]
        )
        self.ca_blocks2 = nn.ModuleList(
            [
                TransformerCrossAttentionLayer(
                    model_dim=hidden_dim,
                    mha_config={
                        "num_heads": num_heads,
                        "new_mask": True,
                    },
                    dense_config={
                        "act_h": act,
                        "hddn_dim": 2 * hidden_dim,
                    },
                    ctxt_dim=2 * hidden_dim,
                )
                for _ in range(num_ca_layers)
            ]
        )

        self.final_layer = DenseNetwork(
            inpt_dim=hidden_dim,
            outp_dim=fs_data_dim,
            hddn_dim=[hidden_dim, hidden_dim, hidden_dim],
            act_h=act,
            ctxt_dim=2 * hidden_dim,
        )
        idx = torch.arange(config["max_particles"]).unsqueeze(0)
        self.register_buffer("idx", idx)

        if self.train_npf:
            self.npf_model = DenseNetwork(
                inpt_dim=hidden_dim + 1,
                outp_dim=config["max_particles"],
                hddn_dim=[hidden_dim // 2, hidden_dim // 2],
                act_h=act,
            )
            self.npf_loss = nn.CrossEntropyLoss(reduction="none")

    def forward(
        self,
        fs_data,
        truth_data,
        mask,
        timestep,
        scale=None,
        fs_data_prev=None,
        return_npf=False,
    ):
        truth_mask = mask[..., 0]
        pf_mask = mask[..., 1]

        if self.use_pos_embd:
            idx = self.idx.expand(fs_data.size(0), -1)
            pos_embd = self.pos_embedding(idx)[:, : fs_data.size(1)]
            fs_data_ = torch.cat([fs_data, pos_embd], dim=-1)
            truth_data_ = torch.cat([truth_data, pos_embd], dim=-1)
        else:
            fs_data = fs_data_
            truth_data_ = truth_data
        truth_embd = self.truth_init(truth_data_)
        truth_embd = self.truth_embedding(truth_embd, mask=truth_mask)

        truth_ctxt = torch.sum(
            truth_embd * truth_mask.unsqueeze(-1), dim=1
        ) / torch.sum(truth_mask, dim=1, keepdim=True)
        time_embd = self.time_embedding(timestep)

        ctxt = torch.cat(
            [
                truth_ctxt + self.global_embedding(scale)
                if self.use_global
                else time_embd,
                time_embd,
            ],
            -1,
        )
        if fs_data_prev is not None and self.use_prev:
            fs_data_ = torch.cat([fs_data_, fs_data_prev], -1)
        fs_embd = self.fs_init(fs_data_, ctxt)
        for block, block2 in zip(self.ca_blocks, self.ca_blocks2):
            fs_embd = block(
                q_seq=fs_embd,
                kv_seq=truth_embd,
                ctxt=ctxt,
                q_mask=pf_mask,
                kv_mask=truth_mask,
            )
            truth_embd = block2(
                q_seq=truth_embd,
                kv_seq=fs_embd,
                ctxt=ctxt,
                q_mask=truth_mask,
                kv_mask=pf_mask,
            )
        fs_out = self.final_layer(fs_embd, ctxt)
        if self.train_npf and return_npf:
            num_pf = mask[..., 1].sum(-1).long()
            num_tr = mask[..., 0].sum(-1).float().view(-1, 1)
            npf_logits = self.npf_model(torch.cat([truth_ctxt, num_tr], dim=-1))
            npf_loss = self.npf_loss(npf_logits, num_pf)
            return fs_out, npf_loss
        return fs_out

    def get_embd(self, truth_data, truth_mask):
        if self.use_pos_embd:
            idx = self.idx.expand(truth_data.size(0), -1)
            pos_embd = self.pos_embedding(idx)[:, : truth_data.size(1)]
            truth_data_ = torch.cat([truth_data, pos_embd], dim=-1)
        else:
            truth_data_ = truth_data
        truth_embd = self.truth_init(truth_data_)
        truth_embd = self.truth_embedding(truth_embd, mask=truth_mask)
        truth_ctxt = torch.sum(
            truth_embd * truth_mask.unsqueeze(-1), dim=1
        ) / torch.sum(truth_mask, dim=1, keepdim=True)
        return truth_ctxt

    @torch.no_grad()
    def sample_npf(self, truth_data, mask):
        truth_mask = mask[..., 0]
        truth_ctxt = self.get_embd(truth_data, truth_mask)
        num_tr = mask[..., 0].sum(-1).float().view(-1, 1)
        npf_logits = self.npf_model(torch.cat([truth_ctxt, num_tr], dim=-1))
        pred_num_pf = torch.multinomial(
            npf_logits.softmax(-1), 1, replacement=True
        ).squeeze(1)
        return pred_num_pf.cpu()


class EDMNetSimple(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        hidden_dim = config["hidden_dim"]
        num_heads = config["num_heads"]
        num_ca_layers = config["num_ca_layers"]

        act = config["act"]
        self.hidden_dim = hidden_dim
        self.act = act

        self.use_pos_embd = config["use_pos_embd"]
        self.use_global = config.get("use_global", False)
        self.train_npf = config.get("train_npf", False)
        self.use_prev = config.get("use_prev", False)
        self.train_class = config.get("train_class", False)

        self.time_embedding = TimestepEmbedder(hidden_dim)

        if self.use_pos_embd:
            self.pos_embedding = SineCosineEncoding(
                hidden_dim // 4, min_value=0, max_value=256
            )

        self.truth_init = DenseNetwork(
            inpt_dim=4 + (hidden_dim // 4 if self.use_pos_embd else 0),
            outp_dim=hidden_dim,
            hddn_dim=[hidden_dim, hidden_dim],
            act_h=act,
        )
        fs_data_dim = 3 + self.train_class
        self.fs_init = DenseNetwork(
            inpt_dim=fs_data_dim
            + (hidden_dim // 4 if self.use_pos_embd else 0)
            + fs_data_dim * self.use_prev,
            outp_dim=hidden_dim,
            hddn_dim=[hidden_dim, hidden_dim],
            act_h=act,
            ctxt_dim=2 * hidden_dim,
        )
        if self.use_global:
            self.global_embedding = DenseNetwork(
                inpt_dim=6,
                outp_dim=hidden_dim,
                hddn_dim=[hidden_dim, hidden_dim],
                act_h=act,
            )

        self.truth_embedding = TransformerEncoder(
            model_dim=hidden_dim,
            num_layers=2,
            mha_config={
                "num_heads": num_heads,
                "new_mask": True,
            },
            dense_config={
                "act_h": act,
                "hddn_dim": 2 * hidden_dim,
            },
        )

        self.ca_blocks = nn.ModuleList(
            [
                TransformerCrossAttentionLayer(
                    model_dim=hidden_dim,
                    mha_config={
                        "num_heads": num_heads,
                        "new_mask": True,
                    },
                    dense_config={
                        "act_h": act,
                        "hddn_dim": 2 * hidden_dim,
                    },
                    ctxt_dim=2 * hidden_dim,
                )
                for _ in range(num_ca_layers)
            ]
        )

        self.ca_blocks2 = nn.ModuleList(
            [
                TransformerCrossAttentionLayer(
                    model_dim=hidden_dim,
                    mha_config={
                        "num_heads": num_heads,
                        "new_mask": True,
                    },
                    dense_config={
                        "act_h": act,
                        "hddn_dim": 2 * hidden_dim,
                    },
                    ctxt_dim=2 * hidden_dim,
                )
                for _ in range(num_ca_layers)
            ]
        )

        self.final_layer = DenseNetwork(
            inpt_dim=hidden_dim,
            outp_dim=fs_data_dim,
            hddn_dim=[hidden_dim, hidden_dim, hidden_dim],
            act_h=act,
            ctxt_dim=2 * hidden_dim,
        )
        idx = torch.arange(config["max_particles"]).unsqueeze(0)
        self.register_buffer("idx", idx)

        if self.train_npf:
            self.npf_model = DenseNetwork(
                inpt_dim=hidden_dim + 1,
                outp_dim=config["max_particles"],
                hddn_dim=[hidden_dim // 2, hidden_dim // 2],
                act_h=act,
            )
            self.npf_loss = nn.CrossEntropyLoss(reduction="none")

        self.sigma_min = 0
        self.sigma_max = float("inf")
        self.sigma_data = float(config.get("sigma_data", 1.1))

    def forward(
        self,
        fs_data,
        truth_data,
        mask,
        timestep,
        scale=None,
        fs_data_prev=None,
        return_npf=False,
    ):
        truth_mask = mask[..., 0]
        pf_mask = mask[..., 1]

        c_skip = self.sigma_data**2 / (timestep**2 + self.sigma_data**2)
        c_out = timestep * self.sigma_data / (timestep**2 + self.sigma_data**2).sqrt()
        c_in = 1 / (self.sigma_data**2 + timestep**2).sqrt()
        c_noise = timestep.log() / 4.0

        c_in = c_in.reshape(-1, 1, 1)
        c_out = c_out.reshape(-1, 1, 1)
        c_skip = c_skip.reshape(-1, 1, 1)

        if self.use_pos_embd:
            idx = self.idx.expand(fs_data.size(0), -1)
            pos_embd = self.pos_embedding(idx)[:, : fs_data.size(1)]
            fs_data_ = torch.cat([c_in * fs_data, pos_embd], dim=-1)
            truth_data_ = torch.cat([truth_data, pos_embd], dim=-1)
        else:
            fs_data_ = fs_data
            truth_data_ = truth_data
        truth_embd = self.truth_init(truth_data_)
        truth_embd = self.truth_embedding(truth_embd, mask=truth_mask)

        truth_ctxt = torch.sum(
            truth_embd * truth_mask.unsqueeze(-1), dim=1
        ) / torch.sum(truth_mask, dim=1, keepdim=True)

        time_embd = self.time_embedding(c_noise)

        ctxt = torch.cat(
            [
                truth_ctxt + self.global_embedding(scale)
                if self.use_global
                else time_embd,
                time_embd,
            ],
            -1,
        )
        if fs_data_prev is not None and self.use_prev:
            fs_data_ = torch.cat([fs_data_, fs_data_prev], -1)
        fs_embd = self.fs_init(fs_data_, ctxt)

        for block, block2 in zip(self.ca_blocks, self.ca_blocks2):
            fs_embd = block(
                q_seq=fs_embd,
                kv_seq=truth_embd,
                ctxt=ctxt,
                q_mask=pf_mask,
                kv_mask=truth_mask,
            )
            truth_embd = block2(
                q_seq=truth_embd,
                kv_seq=fs_embd,
                ctxt=ctxt,
                q_mask=truth_mask,
                kv_mask=pf_mask,
            )
        fs_out = self.final_layer(fs_embd, ctxt) * c_out + c_skip * fs_data
        if self.train_npf and return_npf:
            num_pf = mask[..., 1].sum(-1).long()
            num_tr = mask[..., 0].sum(-1).float().view(-1, 1)
            npf_logits = self.npf_model(torch.cat([truth_ctxt, num_tr], dim=-1))
            npf_loss = self.npf_loss(npf_logits, num_pf)
            return fs_out, npf_loss
        return fs_out

    def get_embd(self, truth_data, truth_mask):
        if self.use_pos_embd:
            idx = self.idx.expand(truth_data.size(0), -1)
            pos_embd = self.pos_embedding(idx)[:, : truth_data.size(1)]
            truth_data_ = torch.cat([truth_data, pos_embd], dim=-1)
        else:
            truth_data_ = truth_data
        truth_embd = self.truth_init(truth_data_)
        truth_embd = self.truth_embedding(truth_embd, mask=truth_mask)
        truth_ctxt = torch.sum(
            truth_embd * truth_mask.unsqueeze(-1), dim=1
        ) / torch.sum(truth_mask, dim=1, keepdim=True)

        return truth_ctxt

    @torch.no_grad()
    def sample_npf(self, truth_data, mask):
        truth_mask = mask[..., 0]
        truth_ctxt = self.get_embd(truth_data, truth_mask)
        num_tr = mask[..., 0].sum(-1).float().view(-1, 1)
        npf_logits = self.npf_model(torch.cat([truth_ctxt, num_tr], dim=-1))
        pred_num_pf = torch.multinomial(
            npf_logits.softmax(-1), 1, replacement=True
        ).squeeze(1)
        return pred_num_pf.cpu()
