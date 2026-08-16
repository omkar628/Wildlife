"""
Distance-Aware GNN V3.1 — optimized multi-forest training.

Designed for the user's 20-forest synthetic wildlife dataset.

Key fixes over the previous V3:
1. Each training example is encoded EXACTLY ONCE. The previous V3 encoded
   all 178k rows once in MultiForestDataset.__init__, then encoded them again
   to construct train/val/test, causing a large CPU preprocessing delay.
2. Preprocessing uses camera lookup arrays + itertuples instead of repeated
   pandas .loc calls.
3. All encoded examples are converted to compact tensors before training.
4. Train/val/test are split from those tensors without re-encoding.
5. GCN is intentionally recomputed per training batch so gradients update
   the GCN parameters correctly. It is NOT incorrectly cached across optimizer
   steps.
6. Validation/test use no_grad + inference_mode.
7. CUDA AMP is used when available; the candidate mask uses a dtype-safe
   value, so FP16 does not overflow.
8. Camera feature normalization statistics are fitted on TRAIN forests only
   to avoid even feature-normalization leakage from held-out forests.
9. A block-diagonal 1000-camera graph prevents cross-forest message passing.
10. Prints preprocessing progress so a long silent startup cannot be mistaken
    for GPU training.

Expected files:
  cameras.parquet
  camera_edges.parquet
  training_examples.parquet

Optional output:
  gnn_model_v3_optimized_best.pt
  gnn_model_v3_optimized_results.json

Example Colab:
  !python /content/gnn_model_v3_optimized.py \
      --datadir /content/drive/MyDrive/WildlifeIntelligence/data_20_forests \
      --epochs 60 --batch 1024
"""

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Reproducibility
# ============================================================
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Geography
# ============================================================
def haversine_km(lat1, lon1, lat2, lon2):
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    )
    return 6371.0 * 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def zscore_fit_apply(values, train_mask):
    """Fit normalization on TRAIN cameras only; apply to every camera."""
    values = np.asarray(values, dtype=np.float32)
    train_values = values[train_mask]
    train_values = train_values[np.isfinite(train_values)]

    if len(train_values) == 0:
        mean, std = 0.0, 1.0
    else:
        mean = float(np.mean(train_values))
        std = float(np.std(train_values))
        if not np.isfinite(std) or std < 1e-8:
            std = 1.0

    out = np.nan_to_num(values, nan=mean, posinf=mean, neginf=mean)
    return ((out - mean) / std).astype(np.float32), mean, std


# ============================================================
# GCN
# ============================================================
class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.10):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj_norm):
        x = torch.matmul(adj_norm, x)
        x = self.linear(x)
        x = F.relu(x)
        return self.dropout(x)


class CameraGCN(nn.Module):
    def __init__(self, in_dim, hidden=96, out_dim=96):
        super().__init__()
        self.gcn1 = GCNLayer(in_dim, hidden)
        self.gcn2 = GCNLayer(hidden, out_dim)

    def forward(self, x, adj_norm):
        x = self.gcn1(x, adj_norm)
        x = self.gcn2(x, adj_norm)
        return x


# ============================================================
# Full model
# ============================================================
class DistanceAwareGNN(nn.Module):
    def __init__(self, camera_dim, candidate_extra_dim=9,
                 hidden=96, gru_hidden=128):
        super().__init__()

        self.camera_gnn = CameraGCN(
            in_dim=camera_dim,
            hidden=hidden,
            out_dim=hidden,
        )

        self.history_gru = nn.GRU(
            input_size=hidden + 1,
            hidden_size=gru_hidden,
            num_layers=1,
            batch_first=True,
        )

        score_in = gru_hidden + hidden + candidate_extra_dim

        self.scorer = nn.Sequential(
            nn.Linear(score_in, 128),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        camera_x,
        adj_norm,
        history_idx,
        history_dt,
        candidate_idx,
        candidate_extra,
        candidate_mask,
    ):
        camera_emb = self.camera_gnn(camera_x, adj_norm)

        hist_emb = camera_emb[history_idx]
        hist_in = torch.cat([hist_emb, history_dt], dim=-1)

        _, h = self.history_gru(hist_in)
        tiger_state = h[-1]

        cand_emb = camera_emb[candidate_idx]
        B, C, _ = cand_emb.shape

        tiger_expand = tiger_state.unsqueeze(1).expand(-1, C, -1)

        score_input = torch.cat(
            [tiger_expand, cand_emb, candidate_extra],
            dim=-1,
        )

        logits = self.scorer(score_input).squeeze(-1)

        # FP16-safe masking.
        logits = logits.masked_fill(
            ~candidate_mask,
            torch.finfo(logits.dtype).min,
        )

        return logits


# ============================================================
# Camera graph/features
# ============================================================
class CameraGraph:
    def __init__(self, cameras, edges, train_forests):
        self.cameras = cameras.reset_index(drop=True)
        self.edges = edges.reset_index(drop=True)

        self.camera_ids = self.cameras["camera_id"].tolist()
        self.camera_idx = {
            cid: i for i, cid in enumerate(self.camera_ids)
        }

        if len(self.camera_idx) != len(self.camera_ids):
            raise ValueError("camera_id values must be globally unique.")

        self.n = len(self.camera_ids)

        self.forest = self.cameras["forest_id"].to_numpy()
        self.lat = self.cameras["latitude"].astype(float).to_numpy()
        self.lon = self.cameras["longitude"].astype(float).to_numpy()

        self.elevation = self._column("elevation", 0.0)
        self.water = self._column("water_distance", 1.0)
        self.road = self._column("road_distance", 1.0)
        self.human = self._column("human_disturbance_base", 0.5)
        self.prey = self._column("prey_density_base", 0.5)
        self.sensitivity = self._column("sensitivity", 0.9)

        habitat = (
            self.cameras.get(
                "habitat_type",
                pd.Series(["unknown"] * len(self.cameras)),
            )
            .fillna("unknown")
            .astype(str)
            .to_numpy()
        )
        self.habitat = habitat
        self.habitats = sorted(np.unique(habitat).tolist())
        self.habitat_to_idx = {h: i for i, h in enumerate(self.habitats)}
        self.habitat_code = np.asarray(
            [self.habitat_to_idx[h] for h in habitat],
            dtype=np.int64,
        )

        train_mask = np.isin(self.forest, np.asarray(train_forests))

        self.camera_features_np = self._build_features(train_mask)
        self.adj_norm = self._build_adjacency()

        # Edge corridor quality for candidate-specific feature.
        self.edge_quality = {}
        for e in self.edges.itertuples(index=False):
            a = getattr(e, "camera_a")
            b = getattr(e, "camera_b")
            q = (
                float(getattr(e, "corridor_quality"))
                if hasattr(e, "corridor_quality")
                else 0.5
            )
            self.edge_quality[(a, b)] = q
            self.edge_quality[(b, a)] = q

    def _column(self, name, default):
        if name not in self.cameras.columns:
            return np.full(self.n, default, dtype=np.float32)
        return (
            pd.to_numeric(self.cameras[name], errors="coerce")
            .fillna(default)
            .to_numpy(dtype=np.float32)
        )

    def _build_features(self, train_mask):
        numeric = [
            self.lat,
            self.lon,
            self.elevation,
            self.water,
            self.road,
            self.human,
            self.prey,
            self.sensitivity,
        ]

        feats = []
        for values in numeric:
            norm, _, _ = zscore_fit_apply(values, train_mask)
            feats.append(norm)

        for i in range(len(self.habitats)):
            feats.append(
                (self.habitat_code == i).astype(np.float32)
            )

        return np.stack(feats, axis=1).astype(np.float32)

    def _build_adjacency(self):
        A = np.zeros((self.n, self.n), dtype=np.float32)

        for e in self.edges.itertuples(index=False):
            a = getattr(e, "camera_a")
            b = getattr(e, "camera_b")

            if a not in self.camera_idx or b not in self.camera_idx:
                continue

            ia = self.camera_idx[a]
            ib = self.camera_idx[b]

            # Hard protection against cross-forest edges.
            if self.forest[ia] != self.forest[ib]:
                continue

            d = (
                float(getattr(e, "distance_km"))
                if hasattr(e, "distance_km")
                else 1.0
            )
            w = math.exp(-max(d, 0.0) / 3.0)

            A[ia, ib] = max(A[ia, ib], w)
            A[ib, ia] = max(A[ib, ia], w)

        A += np.eye(self.n, dtype=np.float32)

        degree = A.sum(axis=1)
        inv_sqrt = 1.0 / np.sqrt(np.maximum(degree, 1e-8))
        A_norm = inv_sqrt[:, None] * A * inv_sqrt[None, :]

        return torch.from_numpy(A_norm.astype(np.float32))

    def candidate_extra(self, last_idx, prev_idx, cand_indices):
        cand_indices = np.asarray(cand_indices, dtype=np.int64)

        d_last = haversine_km(
            self.lat[last_idx],
            self.lon[last_idx],
            self.lat[cand_indices],
            self.lon[cand_indices],
        )

        d_prev = haversine_km(
            self.lat[prev_idx],
            self.lon[prev_idx],
            self.lat[cand_indices],
            self.lon[cand_indices],
        )

        # Majority habitat from the five observed history cameras is handled
        # outside this method; caller passes the matching boolean separately.
        return d_last, d_prev


# ============================================================
# Encode examples ONCE
# ============================================================
def encode_examples(examples, graph, max_candidates=20, history_len=5):
    """
    Returns compact numpy arrays grouped by split.

    This is deliberately the only expensive example-encoding pass.
    """
    H = history_len
    C = max_candidates
    E = 9

    arrays = {
        "train": {
            "history_idx": [],
            "history_dt": [],
            "candidate_idx": [],
            "candidate_extra": [],
            "candidate_mask": [],
            "target": [],
        },
        "val": {
            "history_idx": [],
            "history_dt": [],
            "candidate_idx": [],
            "candidate_extra": [],
            "candidate_mask": [],
            "target": [],
        },
        "test": {
            "history_idx": [],
            "history_dt": [],
            "candidate_idx": [],
            "candidate_extra": [],
            "candidate_mask": [],
            "target": [],
        },
    }

    total = len(examples)
    skipped = 0
    start_time = time.time()

    for row_num, row in enumerate(examples.itertuples(index=False), start=1):
        split = str(getattr(row, "split"))
        if split not in arrays:
            skipped += 1
            continue

        history = getattr(row, "history_camera_ids")
        timestamps = getattr(row, "history_timestamps_h")
        candidates = getattr(row, "candidate_camera_ids")
        target = getattr(row, "target_camera_id")

        if history is None or candidates is None:
            skipped += 1
            continue

        history = list(history)
        candidates = list(candidates)

        if len(history) < H:
            skipped += 1
            continue

        history = history[-H:]

        if target not in candidates:
            candidates.append(target)

        # De-duplicate while preserving candidate order.
        candidates = list(dict.fromkeys(candidates))

        # Camera IDs must exist.
        try:
            hist_idx = [graph.camera_idx[c] for c in history]
            cand_idx_raw = [graph.camera_idx[c] for c in candidates]
        except KeyError:
            skipped += 1
            continue

        # All cameras must belong to this example's forest.
        fid = int(getattr(row, "forest_id"))
        if any(graph.forest[i] != fid for i in hist_idx + cand_idx_raw):
            skipped += 1
            continue

        # If candidate set is too large, retain nearest candidates but ALWAYS
        # retain the true target.
        if len(candidates) > C:
            last_idx = hist_idx[-1]

            candidate_arr = np.asarray(cand_idx_raw, dtype=np.int64)
            distances = haversine_km(
                graph.lat[last_idx],
                graph.lon[last_idx],
                graph.lat[candidate_arr],
                graph.lon[candidate_arr],
            )
            order = np.argsort(distances)

            target_idx_global = graph.camera_idx[target]
            target_position_raw = cand_idx_raw.index(target_idx_global)

            keep = list(order[:C])
            if target_position_raw not in keep:
                keep = keep[: C - 1] + [target_position_raw]

            keep = list(dict.fromkeys(keep))
            candidates = [candidates[i] for i in keep]
            cand_idx_raw = [cand_idx_raw[i] for i in keep]

        target_global = graph.camera_idx[target]
        if target_global not in cand_idx_raw:
            skipped += 1
            continue

        target_position = cand_idx_raw.index(target_global)
        if target_position >= C:
            skipped += 1
            continue

        # Time gaps.
        try:
            ts = np.asarray(list(timestamps)[-H:], dtype=np.float32)
        except Exception:
            skipped += 1
            continue

        if len(ts) != H:
            skipped += 1
            continue

        dts = np.zeros(H, dtype=np.float32)
        if H > 1:
            dts[1:] = np.maximum(0.0, np.diff(ts))
        dts = np.log1p(np.clip(dts, 0.0, 72.0)) / np.log1p(72.0)
        dts = dts.reshape(H, 1)

        # Candidate ecological/spatial features.
        last_idx = hist_idx[-1]
        prev_idx = hist_idx[-2] if H >= 2 else last_idx

        cand_arr = np.asarray(cand_idx_raw, dtype=np.int64)

        d_last = haversine_km(
            graph.lat[last_idx],
            graph.lon[last_idx],
            graph.lat[cand_arr],
            graph.lon[cand_arr],
        )
        d_prev = haversine_km(
            graph.lat[prev_idx],
            graph.lon[prev_idx],
            graph.lat[cand_arr],
            graph.lon[cand_arr],
        )

        hist_habitats = graph.habitat_code[np.asarray(hist_idx)]
        values, counts = np.unique(hist_habitats, return_counts=True)
        preferred_habitat = values[np.argmax(counts)]
        habitat_match = (
            graph.habitat_code[cand_arr] == preferred_habitat
        ).astype(np.float32)

        prey = graph.prey[cand_arr]
        human = graph.human[cand_arr]

        water_proximity = np.exp(-np.maximum(graph.water[cand_arr], 0.0) / 2.0)
        road_distance = graph.road[cand_arr]
        sensitivity = graph.sensitivity[cand_arr]

        corridor = np.asarray(
            [
                graph.edge_quality.get(
                    (graph.camera_ids[last_idx], c),
                    0.15,
                )
                for c in candidates
            ],
            dtype=np.float32,
        )

        extra = np.stack(
            [
                np.log1p(np.maximum(d_last, 0.0)) / np.log1p(20.0),
                np.log1p(np.maximum(d_prev, 0.0)) / np.log1p(20.0),
                habitat_match,
                prey,
                human,
                water_proximity,
                np.log1p(np.maximum(road_distance, 0.0)) / np.log1p(10.0),
                sensitivity,
                corridor,
            ],
            axis=1,
        ).astype(np.float32)

        extra = np.nan_to_num(
            extra, nan=0.0, posinf=1.0, neginf=0.0
        )
        extra = np.clip(extra, -5.0, 5.0)

        # Padding.
        hist_out = np.asarray(hist_idx, dtype=np.int64)
        cand_out = np.zeros(C, dtype=np.int64)
        extra_out = np.zeros((C, E), dtype=np.float32)
        mask_out = np.zeros(C, dtype=np.bool_)

        n = min(len(cand_idx_raw), C)
        cand_out[:n] = np.asarray(cand_idx_raw[:n], dtype=np.int64)
        extra_out[:n] = extra[:n]
        mask_out[:n] = True

        arrays[split]["history_idx"].append(hist_out)
        arrays[split]["history_dt"].append(dts)
        arrays[split]["candidate_idx"].append(cand_out)
        arrays[split]["candidate_extra"].append(extra_out)
        arrays[split]["candidate_mask"].append(mask_out)
        arrays[split]["target"].append(target_position)

        if row_num % 20000 == 0 or row_num == total:
            elapsed = time.time() - start_time
            rate = row_num / max(elapsed, 1e-6)
            print(
                f"  Encoding {row_num:,}/{total:,} "
                f"({100.0 * row_num / total:.1f}%) | "
                f"{rate:,.0f} rows/s | skipped {skipped:,}"
            )

    result = {}
    for split, d in arrays.items():
        if not d["target"]:
            raise RuntimeError(f"No usable examples found for split={split}")

        result[split] = {
            "history_idx": np.stack(d["history_idx"]).astype(np.int64),
            "history_dt": np.stack(d["history_dt"]).astype(np.float32),
            "candidate_idx": np.stack(d["candidate_idx"]).astype(np.int64),
            "candidate_extra": np.stack(d["candidate_extra"]).astype(np.float32),
            "candidate_mask": np.stack(d["candidate_mask"]).astype(np.bool_),
            "target": np.asarray(d["target"], dtype=np.int64),
        }

    print(f"  Encoding complete. Skipped: {skipped:,}")
    return result


# ============================================================
# Tensor storage / batching
# ============================================================
def make_tensors(arrays):
    out = {}
    for split, d in arrays.items():
        out[split] = {
            k: torch.from_numpy(v)
            for k, v in d.items()
        }

        # Pinned CPU memory speeds host->CUDA transfers.
        if torch.cuda.is_available():
            for k in out[split]:
                out[split][k] = out[split][k].pin_memory()

    return out


def get_batch(data, indices, device):
    non_blocking = device.type == "cuda"
    return {
        "history_idx": data["history_idx"][indices].to(
            device, non_blocking=non_blocking
        ),
        "history_dt": data["history_dt"][indices].to(
            device, non_blocking=non_blocking
        ),
        "candidate_idx": data["candidate_idx"][indices].to(
            device, non_blocking=non_blocking
        ),
        "candidate_extra": data["candidate_extra"][indices].to(
            device, non_blocking=non_blocking
        ),
        "candidate_mask": data["candidate_mask"][indices].to(
            device, non_blocking=non_blocking
        ),
        "target": data["target"][indices].to(
            device, non_blocking=non_blocking
        ),
    }


# ============================================================
# Metrics
# ============================================================
def topk_metrics(logits, targets):
    max_k = min(5, logits.shape[1])
    _, pred = torch.topk(logits, k=max_k, dim=1)

    result = {}
    for k in (1, 3, 5):
        kk = min(k, logits.shape[1])
        result[f"top{k}"] = (
            (pred[:, :kk] == targets.unsqueeze(1))
            .any(dim=1)
            .float()
            .mean()
            .item()
        )

    order = torch.argsort(logits, dim=1, descending=True)
    matches = order == targets.unsqueeze(1)
    ranks = matches.float().argmax(dim=1) + 1
    result["mrr"] = (1.0 / ranks.float()).mean().item()

    return result


# ============================================================
# Training / evaluation
# ============================================================
def run_epoch(
    model,
    data,
    optimizer,
    camera_x,
    adj_norm,
    device,
    batch_size,
    train,
    scaler,
    seed,
):
    if train:
        model.train()
        generator = torch.Generator()
        generator.manual_seed(seed)
        order = torch.randperm(
            data["target"].shape[0],
            generator=generator,
        )
    else:
        model.eval()
        order = torch.arange(data["target"].shape[0])

    total_loss = 0.0
    total_n = 0
    sums = {"top1": 0.0, "top3": 0.0, "top5": 0.0, "mrr": 0.0}

    use_amp = device.type == "cuda"

    for start in range(0, len(order), batch_size):
        idx = order[start : start + batch_size]
        batch = get_batch(data, idx, device)

        if train:
            optimizer.zero_grad(set_to_none=True)

        context = (
            torch.enable_grad()
            if train
            else torch.inference_mode()
        )

        with context:
            if use_amp:
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                ):
                    logits = model(
                        camera_x,
                        adj_norm,
                        batch["history_idx"],
                        batch["history_dt"],
                        batch["candidate_idx"],
                        batch["candidate_extra"],
                        batch["candidate_mask"],
                    )
                    loss = F.cross_entropy(
                        logits,
                        batch["target"],
                    )
            else:
                logits = model(
                    camera_x,
                    adj_norm,
                    batch["history_idx"],
                    batch["history_dt"],
                    batch["candidate_idx"],
                    batch["candidate_extra"],
                    batch["candidate_mask"],
                )
                loss = F.cross_entropy(
                    logits,
                    batch["target"],
                )

            if train:
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), 2.0
                    )
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), 2.0
                    )
                    optimizer.step()

        n = batch["target"].shape[0]
        total_loss += float(loss.detach().item()) * n
        total_n += n

        m = topk_metrics(
            logits.detach().float(),
            batch["target"],
        )

        for k in sums:
            sums[k] += m[k] * n

    if total_n == 0:
        raise RuntimeError("run_epoch received zero examples.")

    return {
        "loss": total_loss / total_n,
        **{k: v / total_n for k, v in sums.items()},
    }


# ============================================================
# Baselines
# ============================================================
def evaluate_nearest(examples, graph):
    correct = 0
    total = 0

    for row in examples.itertuples(index=False):
        history = list(getattr(row, "history_camera_ids"))
        candidates = list(getattr(row, "candidate_camera_ids"))
        target = getattr(row, "target_camera_id")

        if not history or not candidates:
            continue

        last = history[-1]
        if last not in graph.camera_idx:
            continue

        last_idx = graph.camera_idx[last]
        cand_idx = [
            graph.camera_idx[c]
            for c in candidates
            if c in graph.camera_idx
        ]

        if not cand_idx:
            continue

        d = haversine_km(
            graph.lat[last_idx],
            graph.lon[last_idx],
            graph.lat[np.asarray(cand_idx)],
            graph.lon[np.asarray(cand_idx)],
        )

        pred_idx = cand_idx[int(np.argmin(d))]
        pred_camera = graph.camera_ids[pred_idx]

        correct += int(pred_camera == target)
        total += 1

    return correct / max(total, 1)


def evaluate_random(examples):
    vals = []
    for row in examples.itertuples(index=False):
        candidates = list(getattr(row, "candidate_camera_ids"))
        if candidates:
            vals.append(1.0 / len(candidates))
    return float(np.mean(vals)) if vals else 0.0


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--datadir",
        type=str,
        default="/content/data_20_forests",
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument("--gru-hidden", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-candidates", type=int, default=20)

    args = parser.parse_args()
    seed_everything(args.seed)

    data_dir = Path(args.datadir)

    if not data_dir.exists():
        raise FileNotFoundError(
            f"Dataset directory does not exist: {data_dir}"
        )

    required = [
        "cameras.parquet",
        "camera_edges.parquet",
        "training_examples.parquet",
    ]

    for name in required:
        path = data_dir / name
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 72)
    print("DISTANCE-AWARE GNN V3.1 — OPTIMIZED MULTI-FOREST TRAINING")
    print("=" * 72)
    print("Device:", device)

    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        print("GPU:", torch.cuda.get_device_name(0))
        print(
            "GPU memory:",
            round(props.total_memory / 1024**3, 2),
            "GB",
        )

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------
    cameras = pd.read_parquet(data_dir / "cameras.parquet")
    edges = pd.read_parquet(data_dir / "camera_edges.parquet")
    examples = pd.read_parquet(
        data_dir / "training_examples.parquet"
    )

    print("\nLoaded:")
    print("  Cameras:", len(cameras))
    print("  Edges:", len(edges))
    print("  Examples:", len(examples))
    print(
        "  Forests:",
        sorted(examples.forest_id.unique().tolist()),
    )
    print(
        "  Split sizes:",
        examples["split"].value_counts().to_dict(),
    )

    # --------------------------------------------------------
    # Strict forest split checks
    # --------------------------------------------------------
    train_forests = set(
        examples.loc[
            examples["split"] == "train", "forest_id"
        ].unique()
    )
    val_forests = set(
        examples.loc[
            examples["split"] == "val", "forest_id"
        ].unique()
    )
    test_forests = set(
        examples.loc[
            examples["split"] == "test", "forest_id"
        ].unique()
    )

    if train_forests & val_forests:
        raise RuntimeError(
            f"Train/val forest overlap detected: "
            f"{train_forests & val_forests}"
        )

    if train_forests & test_forests:
        raise RuntimeError(
            f"Train/test forest overlap detected: "
            f"{train_forests & test_forests}"
        )

    if val_forests & test_forests:
        raise RuntimeError(
            f"Val/test forest overlap detected: "
            f"{val_forests & test_forests}"
        )

    print("\nForest split:")
    print("  Train:", sorted(train_forests))
    print("  Val:", sorted(val_forests))
    print("  Test:", sorted(test_forests))

    # --------------------------------------------------------
    # Build graph/features
    # --------------------------------------------------------
    print("\nBuilding global block-diagonal graph...")
    graph = CameraGraph(
        cameras,
        edges,
        train_forests,
    )

    print(
        "  Global nodes:",
        graph.n,
        "| feature dim:",
        graph.camera_features_np.shape[1],
    )

    # --------------------------------------------------------
    # Encode exactly once
    # --------------------------------------------------------
    print("\nEncoding training examples ONCE...")
    t0 = time.time()

    encoded = encode_examples(
        examples,
        graph,
        max_candidates=args.max_candidates,
        history_len=5,
    )

    print(
        f"Encoding time: {time.time() - t0:.1f} seconds"
    )

    tensors = make_tensors(encoded)

    print("\nTensor split sizes:")
    for split in ("train", "val", "test"):
        print(
            f"  {split.capitalize():5s}: "
            f"{tensors[split]['target'].shape[0]:,}"
        )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------
    feature_dim = graph.camera_features_np.shape[1]
    candidate_extra_dim = 9

    model = DistanceAwareGNN(
        camera_dim=feature_dim,
        candidate_extra_dim=candidate_extra_dim,
        hidden=args.hidden,
        gru_hidden=args.gru_hidden,
    ).to(device)

    params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print("\nModel parameters:", params)
    print("AMP:", device.type == "cuda")
    print("Batch size:", args.batch)

    # --------------------------------------------------------
    # GPU tensors
    # --------------------------------------------------------
    camera_x = torch.from_numpy(
        graph.camera_features_np
    ).to(device)

    adj_norm = graph.adj_norm.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=4,
    )

    if device.type == "cuda":
        try:
            scaler = torch.amp.GradScaler(
                "cuda",
                enabled=True,
            )
        except TypeError:
            scaler = torch.cuda.amp.GradScaler(enabled=True)
    else:
        scaler = None

    best_val = float("inf")
    best_epoch = -1
    patience_count = 0
    history = []

    checkpoint_path = (
        data_dir / "gnn_model_v3_optimized_best.pt"
    )

    print("\nStarting training...")
    print("If preprocessing is complete, Epoch 001 should appear quickly.")

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        train_result = run_epoch(
            model,
            tensors["train"],
            optimizer,
            camera_x,
            adj_norm,
            device,
            args.batch,
            train=True,
            scaler=scaler,
            seed=args.seed + epoch,
        )

        val_result = run_epoch(
            model,
            tensors["val"],
            optimizer,
            camera_x,
            adj_norm,
            device,
            args.batch,
            train=False,
            scaler=None,
            seed=args.seed,
        )

        scheduler.step(val_result["loss"])

        epoch_time = time.time() - epoch_start

        record = {
            "epoch": epoch,
            "train": train_result,
            "val": val_result,
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": epoch_time,
        }
        history.append(record)

        print(
            f"Epoch {epoch:03d} | "
            f"{epoch_time:.1f}s | "
            f"train loss {train_result['loss']:.4f} "
            f"top1 {train_result['top1']:.3f} | "
            f"val loss {val_result['loss']:.4f} "
            f"top1 {val_result['top1']:.3f} "
            f"top3 {val_result['top3']:.3f} "
            f"top5 {val_result['top5']:.3f} "
            f"MRR {val_result['mrr']:.3f}"
        )

        if val_result["loss"] < best_val - 1e-5:
            best_val = val_result["loss"]
            best_epoch = epoch
            patience_count = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "camera_feature_dim": feature_dim,
                    "candidate_extra_dim": candidate_extra_dim,
                    "hidden": args.hidden,
                    "gru_hidden": args.gru_hidden,
                    "max_candidates": args.max_candidates,
                    "history_len": 5,
                    "seed": args.seed,
                    "version": "v3.1_optimized_multi_forest",
                    "train_forests": sorted(
                        map(int, train_forests)
                    ),
                    "val_forests": sorted(
                        map(int, val_forests)
                    ),
                    "test_forests": sorted(
                        map(int, test_forests)
                    ),
                },
                checkpoint_path,
            )

            print("  -> saved BEST:", checkpoint_path)
        else:
            patience_count += 1

        if patience_count >= args.patience:
            print(
                f"\nEarly stopping at epoch {epoch}. "
                f"Best epoch = {best_epoch}"
            )
            break

    # --------------------------------------------------------
    # Restore best model
    # --------------------------------------------------------
    if not checkpoint_path.exists():
        raise RuntimeError(
            "No best checkpoint was created."
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    # --------------------------------------------------------
    # Final metrics
    # --------------------------------------------------------
    train_final = run_epoch(
        model,
        tensors["train"],
        optimizer,
        camera_x,
        adj_norm,
        device,
        args.batch,
        train=False,
        scaler=None,
        seed=args.seed,
    )

    val_final = run_epoch(
        model,
        tensors["val"],
        optimizer,
        camera_x,
        adj_norm,
        device,
        args.batch,
        train=False,
        scaler=None,
        seed=args.seed,
    )

    test_final = run_epoch(
        model,
        tensors["test"],
        optimizer,
        camera_x,
        adj_norm,
        device,
        args.batch,
        train=False,
        scaler=None,
        seed=args.seed,
    )

    # --------------------------------------------------------
    # Baselines
    # --------------------------------------------------------
    baseline_results = {}

    for split in ("train", "val", "test"):
        sub = examples[
            examples["split"] == split
        ].reset_index(drop=True)

        baseline_results[split] = {
            "nearest_top1": evaluate_nearest(
                sub, graph
            ),
            "random_top1": evaluate_random(sub),
            "n": len(sub),
        }

    # --------------------------------------------------------
    # Save report
    # --------------------------------------------------------
    report = {
        "version": "v3.1_optimized_multi_forest",
        "device": str(device),
        "gpu": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None
        ),
        "epochs_requested": args.epochs,
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "train": train_final,
        "val": val_final,
        "test": test_final,
        "baselines": baseline_results,
        "dataset": {
            "cameras": len(cameras),
            "edges": len(edges),
            "examples": len(examples),
            "train_examples": len(tensors["train"]["target"]),
            "val_examples": len(tensors["val"]["target"]),
            "test_examples": len(tensors["test"]["target"]),
        },
        "forest_split": {
            "train": sorted(map(int, train_forests)),
            "val": sorted(map(int, val_forests)),
            "test": sorted(map(int, test_forests)),
        },
        "architecture": {
            "camera_encoder": "2-layer GCN",
            "history_encoder": "GRU",
            "candidate_encoder": (
                "camera GCN embedding + "
                "9 spatial/ecological features"
            ),
            "camera_id_embedding": False,
            "multi_forest_graph": (
                "single block-diagonal graph; "
                "no cross-forest message passing"
            ),
            "mixed_precision": device.type == "cuda",
            "example_encoding": "single pass",
            "train_feature_normalization": (
                "fit on train forests only"
            ),
        },
        "history": history,
        "checkpoint": str(checkpoint_path),
    }

    result_path = (
        data_dir / "gnn_model_v3_optimized_results.json"
    )

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # --------------------------------------------------------
    # Final print
    # --------------------------------------------------------
    print("\n" + "=" * 72)
    print("FINAL GNN V3.1 RESULTS")
    print("=" * 72)

    for name, result in (
        ("Train", train_final),
        ("Val", val_final),
        ("TEST", test_final),
    ):
        print(
            f"{name:5s} "
            f"Top-1: {result['top1']:.2%} | "
            f"Top-3: {result['top3']:.2%} | "
            f"Top-5: {result['top5']:.2%} | "
            f"MRR: {result['mrr']:.3f}"
        )

    nearest_test = baseline_results["test"]["nearest_top1"]
    random_test = baseline_results["test"]["random_top1"]

    print(
        f"\nNearest-camera TEST: {nearest_test:.2%}"
    )
    print(
        f"Random baseline TEST: {random_test:.2%}"
    )
    print(
        f"GNN - nearest: "
        f"{test_final['top1'] - nearest_test:+.2%}"
    )

    print("\nSaved:")
    print(" ", checkpoint_path)
    print(" ", result_path)

    print("\nTraining integrity:")
    print("  Train forests:", sorted(train_forests))
    print("  Val forests:", sorted(val_forests))
    print("  Test forests:", sorted(test_forests))
    print("  Test forests were NOT used for optimization.")

    if device.type == "cuda":
        print(
            "\nPeak GPU memory:",
            round(
                torch.cuda.max_memory_allocated() / 1024**3,
                2,
            ),
            "GB",
        )


if __name__ == "__main__":
    main()
