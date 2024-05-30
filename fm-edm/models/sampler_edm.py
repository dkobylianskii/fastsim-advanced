import torch
from tqdm import tqdm


@torch.no_grad()
def transfer(x_curr, d_curr, t_curr, t_next):
    x_next = x_curr + d_curr * (t_next - t_curr)
    return x_next


@torch.no_grad()
def runge_kutta(model, truth, fastsim, mask, scale, t_list, ets, deriv_prev=None):
    x_1 = model.forward(
        fastsim,
        truth,
        mask,
        timestep=t_list[0].expand(fastsim.shape[0]),
        scale=scale,
        fs_data_prev=deriv_prev,
    )
    d_1 = (fastsim - x_1) / t_list[0]
    ets.append(d_1)
    x_2 = transfer(fastsim, d_1, t_list[0], t_list[1])

    x_2 = model.forward(
        x_2,
        truth,
        mask,
        timestep=t_list[1].expand(fastsim.shape[0]),
        scale=scale,
        fs_data_prev=x_1,
    )
    d_2 = (fastsim - x_2) / t_list[1]
    x_3 = transfer(fastsim, d_2, t_list[0], t_list[1])

    x_3 = model.forward(
        x_3,
        truth,
        mask,
        timestep=t_list[1].expand(fastsim.shape[0]),
        scale=scale,
        fs_data_prev=x_2,
    )
    d_3 = (fastsim - x_3) / t_list[1]
    x_4 = transfer(fastsim, d_3, t_list[0], t_list[2])

    x_4 = model.forward(
        x_4,
        truth,
        mask,
        timestep=t_list[2].expand(fastsim.shape[0]),
        scale=scale,
        fs_data_prev=x_3,
    )
    d_4 = (fastsim - x_4) / t_list[2]
    d_t = (1 / 6) * (d_1 + 2 * d_2 + 2 * d_3 + d_4)

    return d_t, x_1


@torch.no_grad()
def gen_order_4(model, truth, fastsim, mask, scale, t, t_next, ets, deriv_prev=None):
    t_list = [t, (t + t_next) / 2, t_next]
    if len(ets) > 2:
        deriv_ = model.forward(
            fastsim,
            truth,
            mask,
            timestep=t.expand(fastsim.shape[0]),
            scale=scale,
            fs_data_prev=deriv_prev,
        )
        deriv_prev = deriv_
        ets.append((fastsim - deriv_) / t)
        deriv = (1 / 24) * (55 * ets[-1] - 59 * ets[-2] + 37 * ets[-3] - 9 * ets[-4])
    else:
        deriv, x_1 = runge_kutta(
            model,
            truth,
            fastsim,
            mask,
            scale,
            t_list,
            ets,
            deriv_prev=deriv_prev,
        )
        deriv_prev = x_1

    fastsim_next = transfer(fastsim, deriv, t, t_next)
    return fastsim_next, deriv_prev


@torch.no_grad()
def pndm_sampler(model, truth, pflow, mask, scale, n_steps, save_seq=False):
    sigma_min = 0.002
    sigma_max = 80
    rho = 7

    seq = []
    device = truth.device
    fastsim = torch.randn_like(pflow, device=device)
    fastsim[~mask[..., 1]] = 0

    step_indices = torch.arange(n_steps, dtype=torch.float64, device=device)
    t_steps = (
        sigma_max ** (1 / rho)
        + step_indices
        / (n_steps - 1)
        * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ) ** rho
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])]).float()

    fastsim = fastsim * t_steps[0]

    ets = []
    deriv_prev = torch.zeros_like(pflow, device=device)

    for i, (t_cur, t_next) in tqdm(
        enumerate(zip(t_steps[:-1], t_steps[1:])), total=n_steps
    ):
        fastsim, deriv_prev = gen_order_4(
            model,
            truth,
            fastsim,
            mask,
            scale,
            t_cur,
            t_next,
            ets,
            deriv_prev=deriv_prev,
        )
        if save_seq:
            seq.append(fastsim.cpu())
    if save_seq:
        seq = torch.stack(seq)
    return fastsim.cpu(), seq


@torch.no_grad()
def edm_sampler(model, truth, pflow, mask, scale, n_steps, save_seq=False):
    def round_sigma(sigma):
        return torch.as_tensor(sigma)

    sigma_min = 0.002
    sigma_max = 80
    rho = 7

    seq = []
    latents = torch.randn((pflow.shape[0], pflow.shape[1], 4), device=pflow.device)

    step_indices = torch.arange(n_steps, dtype=torch.float64, device=latents.device)
    t_steps = (
        sigma_max ** (1 / rho)
        + step_indices
        / (n_steps - 1)
        * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ) ** rho
    t_steps = torch.cat(
        [round_sigma(t_steps), torch.zeros_like(t_steps[:1])]
    ).float()  # t_N = 0

    x_next = latents * t_steps[0]
    x_next[~mask[..., 1]] = 0

    x_prev = torch.zeros_like(x_next, device=x_next.device)
    # self.convert_to_f64(gp)
    for i, (t_cur, t_next) in tqdm(
        enumerate(zip(t_steps[:-1], t_steps[1:])), total=n_steps
    ):  # 0, ..., N-1
        x_cur = x_next

        t_hat = round_sigma(t_cur)
        x_hat = x_cur + (t_hat**2 - t_cur**2).sqrt()

        x_pred = model.forward(
            x_next,
            truth,
            mask,
            timestep=t_hat.expand(x_next.shape[0]),
            scale=scale,
            fs_data_prev=x_prev,
        )

        denoised = x_pred
        x_prev = x_pred
        d_cur = (x_hat - denoised) / t_hat
        t_next_prime = t_next
        x_next = x_hat + (t_next_prime - t_hat) * d_cur

        # Apply 2nd order correction.
        if i < n_steps - 1:
            x_pred = model.forward(
                x_next,
                truth,
                mask,
                timestep=t_next.expand(x_next.shape[0]),
                scale=scale,
                fs_data_prev=x_prev,
            )

            denoised = x_pred
            t_next_prime = t_next
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next_prime - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
    return x_next.cpu(), seq
