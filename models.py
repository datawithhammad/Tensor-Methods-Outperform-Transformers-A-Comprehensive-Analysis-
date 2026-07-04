"""The seven link-prediction models evaluated in the paper.

Classical embeddings : TransE, DistMult, ComplEx      (negative sampling)
Tensor factorisation : Tensor-CP, Tensor-Tucker       (1-N scoring, reciprocal relations)
Graph neural network : R-GCN encoder + DistMult decoder
Transformer          : LP-BERT-style encoder trained from scratch
                       (BERT architecture over [CLS, head, relation] token
                       sequences; no language-model pretraining)

Every model exposes:
    score_tails(h, r) -> [B, n_entities]   scores for (h, r, ?)
    score_heads(t, r) -> [B, n_entities]   scores for (?, r, t)
    family in {"ns", "1n", "gnn"}          which training loop applies
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Classical embedding models (trained with uniform negative sampling)
# --------------------------------------------------------------------------- #
class TransE(nn.Module):
    family = "ns"

    def __init__(self, n_ent, n_rel, dim=100, margin=2.0):
        super().__init__()
        self.ent = nn.Embedding(n_ent, dim)
        self.rel = nn.Embedding(n_rel, dim)
        nn.init.xavier_uniform_(self.ent.weight)
        nn.init.xavier_uniform_(self.rel.weight)
        self.margin = margin

    def normalize(self):
        with torch.no_grad():
            self.ent.weight.data = F.normalize(self.ent.weight.data, dim=1)

    def score(self, h, r, t):  # higher is better
        return -torch.norm(self.ent(h) + self.rel(r) - self.ent(t), p=2, dim=1)

    def loss(self, pos, neg):
        h, r, t = pos
        nh, nr, nt = neg
        pos_s = self.score(h, r, t)
        neg_s = self.score(nh, nr, nt)
        k = neg_s.shape[0] // pos_s.shape[0]
        return F.relu(self.margin + neg_s - pos_s.repeat_interleave(k)).mean()

    def score_tails(self, h, r):
        q = self.ent(h) + self.rel(r)
        return -torch.cdist(q, self.ent.weight)

    def score_heads(self, t, r):
        q = self.ent(t) - self.rel(r)
        return -torch.cdist(q, self.ent.weight)


class DistMult(nn.Module):
    family = "ns"

    def __init__(self, n_ent, n_rel, dim=100):
        super().__init__()
        self.ent = nn.Embedding(n_ent, dim)
        self.rel = nn.Embedding(n_rel, dim)
        nn.init.xavier_uniform_(self.ent.weight)
        nn.init.xavier_uniform_(self.rel.weight)

    def normalize(self):
        pass

    def score(self, h, r, t):
        return (self.ent(h) * self.rel(r) * self.ent(t)).sum(dim=1)

    def loss(self, pos, neg):
        pos_s = self.score(*pos)
        neg_s = self.score(*neg)
        return (
            F.binary_cross_entropy_with_logits(pos_s, torch.ones_like(pos_s))
            + F.binary_cross_entropy_with_logits(neg_s, torch.zeros_like(neg_s))
        ) / 2

    def score_tails(self, h, r):
        return (self.ent(h) * self.rel(r)) @ self.ent.weight.T

    def score_heads(self, t, r):
        return (self.ent(t) * self.rel(r)) @ self.ent.weight.T


class ComplEx(nn.Module):
    family = "ns"

    def __init__(self, n_ent, n_rel, dim=100):
        super().__init__()
        self.ent_re = nn.Embedding(n_ent, dim)
        self.ent_im = nn.Embedding(n_ent, dim)
        self.rel_re = nn.Embedding(n_rel, dim)
        self.rel_im = nn.Embedding(n_rel, dim)
        for emb in (self.ent_re, self.ent_im, self.rel_re, self.rel_im):
            nn.init.xavier_uniform_(emb.weight)

    def normalize(self):
        pass

    def score(self, h, r, t):
        hr, hi = self.ent_re(h), self.ent_im(h)
        rr, ri = self.rel_re(r), self.rel_im(r)
        tr, ti = self.ent_re(t), self.ent_im(t)
        return (hr * rr * tr + hi * rr * ti + hr * ri * ti - hi * ri * tr).sum(dim=1)

    loss = DistMult.loss

    def score_tails(self, h, r):
        hr, hi = self.ent_re(h), self.ent_im(h)
        rr, ri = self.rel_re(r), self.rel_im(r)
        c_re = hr * rr - hi * ri
        c_im = hr * ri + hi * rr
        return c_re @ self.ent_re.weight.T + c_im @ self.ent_im.weight.T

    def score_heads(self, t, r):
        tr, ti = self.ent_re(t), self.ent_im(t)
        rr, ri = self.rel_re(r), self.rel_im(r)
        d_re = rr * tr + ri * ti
        d_im = rr * ti - ri * tr
        return d_re @ self.ent_re.weight.T + d_im @ self.ent_im.weight.T


# --------------------------------------------------------------------------- #
# Tensor factorisation models (1-N scoring with reciprocal relations:
# relation ids [n_rel, 2*n_rel) are the inverses, so head prediction for
# (?, r, t) is tail prediction for (t, r_inv, ?)).
# --------------------------------------------------------------------------- #
class TensorCP(nn.Module):
    family = "1n"

    def __init__(self, n_ent, n_rel, dim=100, dropout=0.2):
        super().__init__()
        self.n_rel = n_rel
        self.subj = nn.Embedding(n_ent, dim)
        self.obj = nn.Embedding(n_ent, dim)
        self.rel = nn.Embedding(2 * n_rel, dim)
        for emb in (self.subj, self.obj, self.rel):
            nn.init.xavier_uniform_(emb.weight)
        self.drop = nn.Dropout(dropout)

    def forward_1n(self, h, r):
        q = self.drop(self.subj(h) * self.rel(r))
        return q @ self.obj.weight.T

    def score_tails(self, h, r):
        return self.forward_1n(h, r)

    def score_heads(self, t, r):
        return self.forward_1n(t, r + self.n_rel)


class TensorTucker(nn.Module):
    """TuckER (Balazevic et al., 2019): shared core tensor W [dr, de, de]."""

    family = "1n"

    def __init__(self, n_ent, n_rel, de=100, dr=50, dropouts=(0.2, 0.2, 0.3)):
        super().__init__()
        self.n_rel = n_rel
        self.ent = nn.Embedding(n_ent, de)
        self.rel = nn.Embedding(2 * n_rel, dr)
        nn.init.xavier_uniform_(self.ent.weight)
        nn.init.xavier_uniform_(self.rel.weight)
        self.W = nn.Parameter(torch.empty(dr, de, de).uniform_(-0.1, 0.1))
        self.bn0 = nn.BatchNorm1d(de)
        self.bn1 = nn.BatchNorm1d(de)
        self.d0 = nn.Dropout(dropouts[0])
        self.d1 = nn.Dropout(dropouts[1])
        self.d2 = nn.Dropout(dropouts[2])

    def forward_1n(self, h, r):
        de = self.ent.weight.shape[1]
        e = self.d0(self.bn0(self.ent(h)))                       # [B, de]
        w = (self.rel(r) @ self.W.view(self.W.shape[0], -1)).view(-1, de, de)
        x = self.d1(torch.bmm(e.unsqueeze(1), w).squeeze(1))     # [B, de]
        x = self.d2(self.bn1(x))
        return x @ self.ent.weight.T

    def score_tails(self, h, r):
        return self.forward_1n(h, r)

    def score_heads(self, t, r):
        return self.forward_1n(t, r + self.n_rel)


# --------------------------------------------------------------------------- #
# Graph neural network: R-GCN encoder (basis decomposition, sparse message
# passing over train edges + inverses) with a DistMult decoder.
# --------------------------------------------------------------------------- #
class RGCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, n_rel_mp, n_bases=8, dropout=0.2):
        super().__init__()
        self.bases = nn.Parameter(torch.empty(n_bases, in_dim, out_dim))
        nn.init.xavier_uniform_(self.bases)
        self.coef = nn.Parameter(torch.empty(n_rel_mp, n_bases))
        nn.init.xavier_uniform_(self.coef)
        self.self_loop = nn.Linear(in_dim, out_dim, bias=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_rel, edge_norm, n_ent):
        # out[dst] += coef[rel, b] * norm * (x @ bases[b])[src], per basis b
        src, dst = edge_index
        out = self.self_loop(x)
        idx = torch.stack([dst, src])
        for b in range(self.bases.shape[0]):
            xb = x @ self.bases[b]                                # [N, out]
            vals = self.coef[edge_rel, b] * edge_norm             # [E]
            adj = torch.sparse_coo_tensor(idx, vals, (n_ent, n_ent))
            out = out + torch.sparse.mm(adj, xb)
        return self.drop(F.relu(out))


class GNNLinkPredictor(nn.Module):
    family = "gnn"

    def __init__(self, n_ent, n_rel, train_edges, dim=64, n_bases=8, dropout=0.2):
        super().__init__()
        self.n_ent = n_ent
        self.feat = nn.Embedding(n_ent, dim)
        nn.init.xavier_uniform_(self.feat.weight)

        # message-passing graph: train edges plus inverse edges
        h, r, t = train_edges  # LongTensors
        src = torch.cat([h, t])
        dst = torch.cat([t, h])
        rel = torch.cat([r, r + n_rel])
        self.register_buffer("edge_index", torch.stack([src, dst]))
        self.register_buffer("edge_rel", rel)
        deg = torch.zeros(n_ent).index_add_(0, dst, torch.ones_like(dst, dtype=torch.float))
        self.register_buffer("edge_norm", 1.0 / deg.clamp(min=1.0)[dst])

        self.layer1 = RGCNLayer(dim, dim, 2 * n_rel, n_bases, dropout)
        self.layer2 = RGCNLayer(dim, dim, 2 * n_rel, n_bases, dropout)
        self.rel_dec = nn.Embedding(n_rel, dim)
        nn.init.xavier_uniform_(self.rel_dec.weight)
        self._cached = None  # encoded entity matrix for evaluation

    def encode(self):
        x = self.feat.weight
        x = self.layer1(x, self.edge_index, self.edge_rel, self.edge_norm, self.n_ent)
        x = self.layer2(x, self.edge_index, self.edge_rel, self.edge_norm, self.n_ent)
        return x

    def cache_encoding(self):
        with torch.no_grad():
            self._cached = self.encode()

    def loss(self, pos, neg):
        z = self.encode()
        h, r, t = pos
        nh, nr, nt = neg
        pos_s = (z[h] * self.rel_dec(r) * z[t]).sum(dim=1)
        neg_s = (z[nh] * self.rel_dec(nr) * z[nt]).sum(dim=1)
        return (
            F.binary_cross_entropy_with_logits(pos_s, torch.ones_like(pos_s))
            + F.binary_cross_entropy_with_logits(neg_s, torch.zeros_like(neg_s))
        ) / 2

    def score_tails(self, h, r):
        z = self._cached
        return (z[h] * self.rel_dec(r)) @ z.T

    def score_heads(self, t, r):
        z = self._cached
        return (z[t] * self.rel_dec(r)) @ z.T


# --------------------------------------------------------------------------- #
# LP-BERT-style transformer: BERT-architecture encoder over the token
# sequence [CLS, head, relation]; the CLS output scores all entities
# (weights tied to the entity token embeddings). Trained from scratch with
# 1-N scoring and reciprocal relations, like the tensor models.
# --------------------------------------------------------------------------- #
class LPBert(nn.Module):
    family = "1n"

    def __init__(self, n_ent, n_rel, dim=128, n_layers=2, n_heads=4, ff=256, dropout=0.1):
        super().__init__()
        self.n_ent = n_ent
        self.n_rel = n_rel
        vocab = n_ent + 2 * n_rel + 1  # entities, relations + inverses, CLS
        self.cls_id = vocab - 1
        self.tok = nn.Embedding(vocab, dim)
        self.pos = nn.Embedding(3, dim)
        nn.init.normal_(self.tok.weight, std=0.02)
        nn.init.normal_(self.pos.weight, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=n_heads, dim_feedforward=ff,
            dropout=dropout, activation="gelu", batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.head = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.LayerNorm(dim))
        self.out_bias = nn.Parameter(torch.zeros(n_ent))

    def forward_1n(self, h, r):
        b = h.shape[0]
        cls = torch.full((b,), self.cls_id, dtype=torch.long, device=h.device)
        seq = torch.stack([cls, h, self.n_ent + r], dim=1)        # [B, 3]
        x = self.tok(seq) + self.pos.weight.unsqueeze(0)
        x = self.encoder(x)[:, 0]                                  # CLS output
        x = self.head(x)
        return x @ self.tok.weight[: self.n_ent].T + self.out_bias

    def score_tails(self, h, r):
        return self.forward_1n(h, r)

    def score_heads(self, t, r):
        return self.forward_1n(t, r + self.n_rel)


MODEL_REGISTRY = {
    "TransE": ("Classical", TransE),
    "DistMult": ("Classical", DistMult),
    "ComplEx": ("Classical", ComplEx),
    "Tensor-CP": ("Tensor", TensorCP),
    "Tensor-Tucker": ("Tensor", TensorTucker),
    "GNN": ("GNN", GNNLinkPredictor),
    "LP-BERT": ("Transformer", LPBert),
}
