import os
import json
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, AutoImageProcessor


# ---- config ----
ALPHAS = [0.60, 0.65, 0.40]
BLEND_W = [0.10, 0.40, 0.40]
SIGMA_SCALE = 1.0


class _AttentionMIL(nn.Module):
    """Gated attention MIL on frozen DinoV2 tokens."""
    def __init__(self, dino, dim=512, n_out=3, drop=0.3):
        super().__init__()
        self.dino = dino
        for p in self.dino.parameters():
            p.requires_grad = False
        self.proj = nn.Sequential(nn.Linear(768, dim), nn.GELU(), nn.Dropout(drop))
        self.gv = nn.Sequential(nn.Linear(dim, 128), nn.Tanh())
        self.gu = nn.Sequential(nn.Linear(dim, 128), nn.Sigmoid())
        self.gw = nn.Linear(128, 1)
        self.mu_head = nn.Sequential(
            nn.Linear(dim, 128), nn.GELU(), nn.Dropout(drop), nn.Linear(128, n_out))
        self.logvar_head = nn.Sequential(
            nn.Linear(dim, 128), nn.GELU(), nn.Dropout(drop), nn.Linear(128, n_out))

    def forward(self, x):
        with torch.no_grad():
            tok = self.dino(x)[0][:, 1:]
        e = self.proj(tok.mean(dim=1))
        h = self.gv(e) * self.gu(e)
        w = torch.softmax(self.gw(h), dim=0)
        z = (w * e).sum(dim=0)
        return self.mu_head(z), torch.clamp(self.logvar_head(z), -10.0, 4.0)


class _FusionMIL(nn.Module):
    """DinoV2 + species/domain embedding fusion."""
    def __init__(self, n_sp=300, n_dom=30, img_dim=256, sp_dim=64, dom_dim=16,
                 n_out=3, drop=0.3):
        super().__init__()
        self.img_proj = nn.Sequential(nn.Linear(768, img_dim), nn.GELU(), nn.Dropout(drop))
        self.sp_embed = nn.Embedding(n_sp, sp_dim, padding_idx=0)
        self.dom_embed = nn.Embedding(n_dom, dom_dim, padding_idx=0)
        fdim = img_dim + sp_dim + dom_dim
        self.gv = nn.Sequential(nn.Linear(fdim, 64), nn.Tanh())
        self.gu = nn.Sequential(nn.Linear(fdim, 64), nn.Sigmoid())
        self.gw = nn.Linear(64, 1)
        self.mu_head = nn.Sequential(
            nn.Linear(fdim, 128), nn.GELU(), nn.Dropout(drop), nn.Linear(128, n_out))
        self.logvar_head = nn.Sequential(
            nn.Linear(fdim, 128), nn.GELU(), nn.Dropout(drop), nn.Linear(128, n_out))

    def forward(self, img_feats, sp_idx, dom_idx):
        N = img_feats.shape[0]
        img = self.img_proj(img_feats)
        sp = self.sp_embed(sp_idx)
        dm = self.dom_embed(dom_idx.unsqueeze(0)).expand(N, -1)
        x = torch.cat([img, sp, dm], dim=1)
        h = self.gv(x) * self.gu(x)
        w = torch.softmax(self.gw(h), dim=0)
        z = (w * x).sum(dim=0)
        return self.mu_head(z), torch.clamp(self.logvar_head(z), -10.0, 4.0)


class _PriorResidual(nn.Module):
    """Species prior + image-based residual."""
    def __init__(self, img_dim=128, n_out=3, drop=0.3):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(768, img_dim), nn.GELU(), nn.Dropout(drop))
        self.head = nn.Sequential(
            nn.Linear(img_dim, 64), nn.GELU(), nn.Dropout(drop), nn.Linear(64, n_out))

    def forward_bag(self, feats):
        return self.head(self.proj(feats).mean(dim=0))


class Model:
    def __init__(self):
        self.models = {}
        self.dino = None
        self.proc = None
        self.lookup = None
        self.svr_lu = None
        self._sp_vocab = {}
        self._dom_vocab = {}
        self._sp_priors = {}
        self._global_mu = [0.0, 0.0, 0.0]

    def load(self):
        here = os.path.dirname(__file__)
        self.dino = AutoModel.from_pretrained("facebook/dinov2-base").cuda().eval()
        self.proc = AutoImageProcessor.from_pretrained("facebook/dinov2-base")

        # ---- lookup tables ----
        lp = os.path.join(here, "lookup.json")
        if os.path.exists(lp):
            with open(lp) as f:
                self.lookup = json.load(f)
        sp = os.path.join(here, "bioclip_svr_predictions.json")
        if os.path.exists(sp):
            with open(sp) as f:
                self.svr_lu = json.load(f)

        # ---- v2: attention MIL ----
        fp = os.path.join(here, "v2_model.pth")
        if os.path.exists(fp):
            m = _AttentionMIL(self.dino).cuda(); m.eval()
            st = torch.load(fp, map_location="cuda")
            st = {k: v for k, v in st.items() if not k.startswith("dino.")}
            m.load_state_dict(st, strict=False)
            self.models["v2"] = m

        # ---- v5: fusion MIL ----
        fp = os.path.join(here, "v5_model.pth")
        if os.path.exists(fp):
            ckpt = torch.load(fp, map_location="cuda")
            self._sp_vocab = ckpt["species_vocab"]
            self._dom_vocab = ckpt["domain_vocab"]
            n_sp = ckpt["model_state"]["sp_embed.weight"].shape[0]
            n_dom = ckpt["model_state"]["dom_embed.weight"].shape[0]
            m = _FusionMIL(n_sp=n_sp, n_dom=n_dom).cuda(); m.eval()
            m.load_state_dict(ckpt["model_state"])
            self.models["v5"] = m

        # ---- v6: prior + residual ----
        fp = os.path.join(here, "v6_model.pth")
        if os.path.exists(fp):
            ckpt = torch.load(fp, map_location="cuda")
            self._sp_priors = ckpt["species_priors"]
            self._global_mu = ckpt["global_mean"]
            m = _PriorResidual().cuda(); m.eval()
            m.load_state_dict(ckpt["model_state"])
            self.models["v6"] = m

    # ---- hierarchical lookup: pair > species > domain > global ----

    def _lu_query(self, lu, dom, sp_names):
        if lu is None:
            return None
        vals = []
        for s in sp_names:
            pk = f"{dom}|{s}"
            if pk in lu["pair"]:
                vals.append(lu["pair"][pk])
            elif s in lu["species"]:
                vals.append(lu["species"][s])
        if vals:
            return np.mean(vals, axis=0)
        dk = str(dom)
        if dk in lu["domain"]:
            return np.array(lu["domain"][dk])
        return np.array(lu["global"])

    # ---- predict ----

    def predict(self, datapoints):
        imgs = [e["relative_img"] for e in datapoints]
        t = torch.stack(
            [self.proc(im, return_tensors="pt")["pixel_values"][0] for im in imgs]
        ).cuda()

        with torch.no_grad():
            tok = self.dino(t)[0][:, 1:]
            feats = tok.mean(dim=1)

        # ---- run ensemble members ----
        mus, vrs = [], []
        if "v2" in self.models:
            with torch.no_grad():
                mu, lv = self.models["v2"](t)
            mus.append(mu.cpu()); vrs.append(torch.exp(lv).cpu())

        if "v5" in self.models:
            si = torch.tensor(
                [self._sp_vocab.get(e["scientificName"], 0) for e in datapoints]).cuda()
            di = torch.tensor(
                self._dom_vocab.get(datapoints[0].get("domainID", -1), 0)).cuda()
            with torch.no_grad():
                mu, lv = self.models["v5"](feats, si, di)
            mus.append(mu.cpu()); vrs.append(torch.exp(lv).cpu())

        if "v6" in self.models:
            pr = [self._sp_priors.get(e.get("scientificName", ""), self._global_mu)
                  for e in datapoints]
            bp = torch.tensor(pr).mean(dim=0)
            with torch.no_grad():
                res = self.models["v6"].forward_bag(feats).cpu()
            mus.append(bp + res); vrs.append(torch.ones(3) * 0.8)

        mu_s = torch.stack(mus)
        mu_f = mu_s.mean(dim=0)

        # ---- sigma from aleatoric + epistemic uncertainty ----
        mv = torch.stack(vrs).mean(dim=0)
        amb = ((mu_s - mu_f) ** 2).mean(dim=0)
        sig = torch.sqrt(mv + amb) * SIGMA_SCALE

        # ---- blend with lookup tables ----
        dom = datapoints[0].get("domainID", -1)
        sp_names = sorted(set(e.get("scientificName", "") for e in datapoints))

        lu_mu = self._lu_query(self.lookup, dom, sp_names)
        if lu_mu is not None:
            m = mu_f.numpy()
            m = np.array([ALPHAS[i] * m[i] + (1 - ALPHAS[i]) * lu_mu[i] for i in range(3)])
            mu_f = torch.tensor(m, dtype=torch.float32)

        svr_mu = self._lu_query(self.svr_lu, dom, sp_names)
        if svr_mu is not None:
            m = mu_f.numpy()
            for i in range(3):
                m[i] = (1 - BLEND_W[i]) * m[i] + BLEND_W[i] * svr_mu[i]
            mu_f = torch.tensor(m, dtype=torch.float32)

        return {
            "SPEI_30d": {"mu": mu_f[0].item(), "sigma": sig[0].item()},
            "SPEI_1y":  {"mu": mu_f[1].item(), "sigma": sig[1].item()},
            "SPEI_2y":  {"mu": mu_f[2].item(), "sigma": sig[2].item()},
        }
