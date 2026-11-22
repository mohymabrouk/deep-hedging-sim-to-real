"""Training market (d): a small DDPM (Ho et al. 2020) on windows of daily BTC log returns.

* data: windows of PATH_LEN daily log returns, standardised by the training std. Daily returns are taken at all
  24 hourly anchors (08:00->08:00, 09:00->09:00, ...): more overlapping windows, no new information.
* model: 1D U-Net (3 levels, GroupNorm, SiLU, self-attention in the middle), epsilon prediction, cosine noise
  schedule with DIFF_STEPS levels, EMA weights; DDIM (eta = 0) sampling with DIFF_SAMPLE_STEPS steps.
* sampling clips the predicted clean window to +-DIFF_X0_CLIP standard deviations at every step (the usual
  "clip_denoised"; the largest 2018-2021 daily move is 8.2 sd). Without it the first DDIM step divides by
  sqrt(alpha_bar) ~ 5e-5 and every sample blew up (median |r| of 8.5 in log terms, seen in validation).
"""

import logging
import math
import time

import numpy as np
import torch
from torch import nn

from dh import config

log = logging.getLogger(__name__)


def sinusoidal(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    f = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float32) / half)
    a = t.float()[:, None] * f[None, :]
    return torch.cat([torch.sin(a), torch.cos(a)], 1)


class ResBlock(nn.Module):
    def __init__(self, cin: int, cout: int, tdim: int):
        super().__init__()
        self.n1, self.c1 = nn.GroupNorm(8, cin), nn.Conv1d(cin, cout, 3, padding=1)
        self.t = nn.Linear(tdim, cout)
        self.n2, self.c2 = nn.GroupNorm(8, cout), nn.Conv1d(cout, cout, 3, padding=1)
        self.skip = nn.Conv1d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, te):
        h = self.c1(nn.functional.silu(self.n1(x))) + self.t(te)[:, :, None]
        h = self.c2(nn.functional.silu(self.n2(h)))
        return h + self.skip(x)


class Attention(nn.Module):
    def __init__(self, c: int, heads: int = 4):
        super().__init__()
        self.n = nn.GroupNorm(8, c)
        self.a = nn.MultiheadAttention(c, heads, batch_first=True)

    def forward(self, x):
        h = self.n(x).transpose(1, 2)
        h, _ = self.a(h, h, h, need_weights=False)
        return x + h.transpose(1, 2)


class UNet1D(nn.Module):
    def __init__(self, ch: int = config.DIFF_CHANNELS, mults=(1, 2, 4)):
        super().__init__()
        tdim = 4 * ch
        self.ch = ch
        self.temb = nn.Sequential(nn.Linear(ch, tdim), nn.SiLU(), nn.Linear(tdim, tdim))
        self.inp = nn.Conv1d(1, ch, 3, padding=1)
        chans = [ch * m for m in mults]
        self.down, self.downsample = nn.ModuleList(), nn.ModuleList()
        c = ch
        for i, co in enumerate(chans):
            self.down.append(nn.ModuleList([ResBlock(c, co, tdim), ResBlock(co, co, tdim)]))
            c = co
            self.downsample.append(nn.Conv1d(c, c, 3, stride=2, padding=1) if i < len(chans) - 1 else nn.Identity())
        self.mid1, self.mid_attn, self.mid2 = ResBlock(c, c, tdim), Attention(c), ResBlock(c, c, tdim)
        self.up, self.upsample = nn.ModuleList(), nn.ModuleList()
        for i, co in reversed(list(enumerate(chans))):
            self.up.append(nn.ModuleList([ResBlock(c + co, co, tdim), ResBlock(co, co, tdim)]))
            c = co
            self.upsample.append(nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv1d(c, chans[i - 1], 3, padding=1)) if i > 0 else nn.Identity())
            if i > 0:
                c = chans[i - 1]
        self.out = nn.Sequential(nn.GroupNorm(8, c), nn.SiLU(), nn.Conv1d(c, 1, 3, padding=1))
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    def forward(self, x, t):
        te = self.temb(sinusoidal(t, self.ch))
        h = self.inp(x)
        skips = []
        for blocks, ds in zip(self.down, self.downsample):
            for b in blocks:
                h = b(h, te)
            skips.append(h)
            h = ds(h)
        h = self.mid2(self.mid_attn(self.mid1(h, te)), te)
        for blocks, us in zip(self.up, self.upsample):
            h = torch.cat([h, skips.pop()], 1)
            for b in blocks:
                h = b(h, te)
            h = us(h)
        return self.out(h)


def cosine_alpha_bar(T: int, s: float = 0.008) -> torch.Tensor:
    t = torch.arange(T + 1, dtype=torch.float64) / T
    f = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    ab = f / f[0]
    beta = torch.clamp(1 - ab[1:] / ab[:-1], max=0.999)
    return torch.cumprod(1 - beta, 0).float()  # alpha_bar for t = 0..T-1


def make_windows(series_list: list[np.ndarray], length: int = config.PATH_LEN) -> np.ndarray:
    out = []
    for s in series_list:
        s = np.asarray(s, float)
        if len(s) >= length:
            out.append(np.lib.stride_tricks.sliding_window_view(s, length))
    return np.concatenate(out, 0)


class Diffusion:
    name = "diffusion"

    def __init__(self, iters: int = config.DIFF_TRAIN_ITERS, seed: int = config.SEED):
        self.iters, self.seed = iters, seed

    def fit(self, r=None, windows: np.ndarray = None, **_):
        """windows: [n, PATH_LEN] daily log returns (all anchors)."""
        torch.manual_seed(self.seed)
        g = torch.Generator().manual_seed(self.seed)
        self.scale = float(windows.std())
        X = torch.as_tensor(windows / self.scale, dtype=torch.float32)[:, None, :]
        self.model = UNet1D()
        self.ema = UNet1D()
        self.ema.load_state_dict(self.model.state_dict())
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.ab = cosine_alpha_bar(config.DIFF_STEPS)
        opt = torch.optim.AdamW(self.model.parameters(), lr=config.DIFF_LR, weight_decay=0.0)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(1.0, (i + 1) / 500) * (0.5 * (1 + math.cos(math.pi * i / self.iters)) * 0.9 + 0.1))
        t0, self.losses = time.time(), []
        self.n_params = sum(p.numel() for p in self.model.parameters())
        log.info("diffusion: %d windows, %d parameters, %d iterations", len(X), self.n_params, self.iters)
        for it in range(self.iters):
            idx = torch.randint(0, len(X), (config.DIFF_BATCH,), generator=g)
            x0 = X[idx]
            t = torch.randint(0, config.DIFF_STEPS, (len(idx),), generator=g)
            eps = torch.randn(x0.shape, generator=g)
            a = self.ab[t][:, None, None]
            xt = a.sqrt() * x0 + (1 - a).sqrt() * eps
            loss = nn.functional.mse_loss(self.model(xt, t), eps)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            opt.step()
            sched.step()
            with torch.no_grad():
                for pe, pm in zip(self.ema.parameters(), self.model.parameters()):
                    pe.mul_(0.999).add_(pm, alpha=0.001)
            self.losses.append(float(loss.detach()))
            if it % 500 == 0 or it == self.iters - 1:
                log.info("  diffusion it %5d  loss %.4f (avg500 %.4f)  %.0fs", it, self.losses[-1], np.mean(self.losses[-500:]), time.time() - t0)
        self.train_seconds = time.time() - t0
        self.ema.eval()
        return self

    def params(self) -> dict:
        if hasattr(self, "_params"):
            return {**self._params, "x0_clip_sd": config.DIFF_X0_CLIP}
        return {"n_params": self.n_params, "iters": self.iters, "scale_daily_std": self.scale, "train_seconds": round(self.train_seconds),
                "final_loss_avg500": float(np.mean(self.losses[-500:])), "sample_steps": config.DIFF_SAMPLE_STEPS}

    @torch.no_grad()
    def sample(self, n: int, rng: np.random.Generator, length: int = config.PATH_LEN, batch: int = 2000) -> np.ndarray:
        g = torch.Generator().manual_seed(int(rng.integers(2**31)))
        steps = torch.linspace(config.DIFF_STEPS - 1, 0, config.DIFF_SAMPLE_STEPS).round().long()
        out = []
        for s in range(0, n, batch):
            m = min(batch, n - s)
            x = torch.randn((m, 1, length), generator=g)
            for i, t in enumerate(steps):
                a = self.ab[t]
                a_prev = self.ab[steps[i + 1]] if i + 1 < len(steps) else torch.tensor(1.0)
                eps = self.ema(x, torch.full((m,), int(t)))
                x0 = ((x - (1 - a).sqrt() * eps) / a.sqrt()).clamp(-config.DIFF_X0_CLIP, config.DIFF_X0_CLIP)
                eps = (x - a.sqrt() * x0) / (1 - a).sqrt()
                x = a_prev.sqrt() * x0 + (1 - a_prev).sqrt() * eps
            out.append(x[:, 0, :].numpy() * self.scale)
        return np.concatenate(out, 0).astype(float)

    def state(self) -> dict:
        return {"ema": self.ema.state_dict(), "scale": self.scale, "params": self.params()}

    @classmethod
    def load(cls, path) -> "Diffusion":
        st = torch.load(path)
        d = cls()
        d.ema = UNet1D()
        d.ema.load_state_dict(st["ema"])
        d.ema.eval()
        d.scale = st["scale"]
        d.ab = cosine_alpha_bar(config.DIFF_STEPS)
        d._params = st["params"]
        return d
