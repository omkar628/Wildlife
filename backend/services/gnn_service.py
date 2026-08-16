"""GNN next-camera inference over existing tiger observations.

Loads ``gnn_model_v3_optimized_best.pt`` into the DistanceAwareGNN
architecture. Does not retrain, does not invent identities, and does not
read the synthetic training parquet files.

If history, coordinates, or graph inputs are insufficient, callers receive
``Insufficient data for GNN prediction`` rather than a fabricated ranking.
"""

from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from backend.config import Settings
from backend.database.connection import Database
from backend.database.repositories import CameraRepository, TigerRepository
from backend.graph.builder import GraphService

logger = logging.getLogger(__name__)

INSUFFICIENT = "Insufficient data for GNN prediction"
HISTORY_LEN = 5
MAX_CANDIDATES = 20
CANDIDATE_EXTRA_DIM = 9
CAMERA_FEATURE_DIM = 12

# Training used 8 numeric columns + 4 habitat one-hots. The habitat vocabulary
# and z-score statistics were not stored in the checkpoint. Production uses
# these documented slots and live-camera z-scores.
HABITAT_SLOTS = ("dry_deciduous", "moist_deciduous", "evergreen", "other")

ECOLOGICAL_DEFAULTS = {
    "water_distance": 1.0,
    "road_distance": 1.0,
    "human_disturbance_base": 0.5,
    "prey_density_base": 0.5,
    "sensitivity": 0.9,
}
ELEVATION_DEFAULT = 0.0
CORRIDOR_DEFAULT = 0.15

METADATA_ALIASES = {
    "water_distance": ("water_distance", "water", "dist_water"),
    "road_distance": ("road_distance", "road", "dist_road"),
    "human_disturbance_base": (
        "human_disturbance_base",
        "human_disturbance",
        "human",
    ),
    "prey_density_base": ("prey_density_base", "prey_density", "prey"),
    "sensitivity": ("sensitivity",),
    "elevation": ("elevation",),
    "habitat": ("habitat_type", "habitat"),
    "corridor_quality": ("corridor_quality",),
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _insufficient(detail: str, **extra: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "available": False,
        "reason": INSUFFICIENT,
        "detail": detail,
    }
    payload.update(extra)
    return payload


def _parse_metadata(raw: Any) -> dict[str, Any]:
    if raw is None or raw == "":
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            loaded = json.loads(raw)
        except (TypeError, ValueError):
            return {}
        return loaded if isinstance(loaded, dict) else {}
    return {}


def _metadata_number(meta: dict[str, Any], key: str) -> float | None:
    for alias in METADATA_ALIASES.get(key, (key,)):
        if alias not in meta or meta[alias] is None or meta[alias] == "":
            continue
        try:
            value = float(meta[alias])
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            return value
    return None


def _metadata_text(meta: dict[str, Any], key: str) -> str | None:
    for alias in METADATA_ALIASES.get(key, (key,)):
        value = meta.get(alias)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def map_habitat(raw: str | None) -> str:
    """Map free-text habitat onto the 4 production one-hot slots."""
    if not raw:
        return "other"
    text = raw.strip().lower().replace("-", "_").replace(" ", "_")
    if text in HABITAT_SLOTS:
        return text
    if "evergreen" in text or "rain" in text:
        return "evergreen"
    if "moist" in text or "damp" in text or "wet" in text:
        return "moist_deciduous"
    if "dry" in text or "teak" in text or "deciduous" in text:
        return "dry_deciduous"
    if "grass" in text or "open" in text:
        return "other"
    return "other"


def parse_timestamp(value: Any) -> datetime | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def timestamp_hours(value: Any) -> float | None:
    parsed = parse_timestamp(value)
    if parsed is None:
        return None
    return parsed.timestamp() / 3600.0


def zscore_column(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        mean, std = 0.0, 1.0
    else:
        mean = float(np.mean(finite))
        std = float(np.std(finite))
        if not np.isfinite(std) or std < 1e-8:
            std = 1.0
    cleaned = np.nan_to_num(values, nan=mean, posinf=mean, neginf=mean)
    return ((cleaned - mean) / std).astype(np.float32)


class GNNService:
    def __init__(self, settings: Settings, db: Database) -> None:
        self.settings = settings
        self.db = db
        self.cameras = CameraRepository(db)
        self.tigers = TigerRepository(db)
        self.graph = GraphService(db)
        self.device_name = "cpu"
        self._model: Any = None
        self._torch: Any = None
        self._checkpoint: dict[str, Any] = {}
        self._load_error: str | None = None
        try:
            self._load()
        except Exception as exc:
            self._model = None
            self._load_error = f"GNN service failed to start: {exc}"
            logger.exception(self._load_error)

    @property
    def weights_path(self) -> Path:
        path = self.settings.gnn_weights_path
        if path is None:
            return Path("gnn_model_v3_optimized_best.pt")
        return Path(path)

    def available(self) -> bool:
        return self._model is not None

    def status(self) -> dict[str, Any]:
        return {
            "loaded": self._model is not None,
            "device": self.device_name,
            "path": str(self.weights_path),
            "version": self._checkpoint.get("version"),
            "camera_feature_dim": self._checkpoint.get("camera_feature_dim"),
            "history_len": self._checkpoint.get("history_len", HISTORY_LEN),
            "max_candidates": self._checkpoint.get("max_candidates", MAX_CANDIDATES),
            "reason": self._load_error,
        }

    def graph_payload(self) -> dict[str, Any]:
        payload = self.status()
        payload["implemented"] = True
        payload["history_len_required"] = HISTORY_LEN
        predictions: list[dict[str, Any]] = []
        if self._model is not None:
            for tiger in self.tigers.list_all():
                tiger_id = str(tiger["tiger_id"])
                predictions.append(self.predict_for_tiger(tiger_id))
        payload["predictions"] = predictions
        return payload

    def predict_for_tiger(self, tiger_id: str | None) -> dict[str, Any]:
        identity = (tiger_id or "").strip()
        if not identity:
            return _insufficient("A tiger_id query parameter is required.")

        if self._model is None:
            return {
                "available": False,
                "reason": self._load_error or "GNN model is not loaded.",
                "tiger_id": identity,
                "detail": self._load_error or "GNN model is not loaded.",
            }

        tiger = self.tigers.get(identity)
        if tiger is None:
            return _insufficient(
                f"Tiger {identity} is not in the tigers table.",
                tiger_id=identity,
            )

        events = self.graph.get_tiger_history(identity)
        usable: list[dict[str, Any]] = []
        for event in events:
            camera_id = event.camera_id
            if not camera_id:
                continue
            usable.append(
                {
                    "camera_id": str(camera_id),
                    "timestamp": event.timestamp,
                    "observation_id": event.observation_id,
                    "confidence": event.confidence,
                }
            )

        if len(usable) < HISTORY_LEN:
            return _insufficient(
                "Need 5 identified camera observations for GNN prediction.",
                tiger_id=identity,
                observation_count=len(usable),
                history_len_required=HISTORY_LEN,
            )

        history = usable[-HISTORY_LEN:]
        camera_rows = self.cameras.list_all()
        try:
            live = self._build_live_graph(camera_rows)
        except ValueError as exc:
            return _insufficient(str(exc), tiger_id=identity)

        try:
            tensors, extras = self._encode_example(history, live)
        except ValueError as exc:
            return _insufficient(str(exc), tiger_id=identity)

        try:
            ranked = self._run_inference(live, tensors)
        except Exception as exc:
            logger.exception("GNN inference failed for %s", identity)
            return {
                "available": False,
                "reason": f"GNN inference failed: {exc}",
                "tiger_id": identity,
                "detail": str(exc),
            }

        if not ranked:
            return _insufficient(
                "GNN returned no ranked candidate cameras.",
                tiger_id=identity,
            )

        top = ranked[0]
        return {
            "available": True,
            "tiger_id": identity,
            "predicted_camera_id": top["camera_id"],
            "confidence": top["confidence"],
            "ranked_candidates": ranked,
            "history": [
                {
                    "camera_id": item["camera_id"],
                    "timestamp": item["timestamp"],
                    "observation_id": item["observation_id"],
                }
                for item in history
            ],
            "prediction_timestamp": _utc_now(),
            "feature_degraded": extras["feature_degraded"],
            "feature_notes": extras["feature_notes"],
            "model": {
                "version": self._checkpoint.get("version"),
                "device": self.device_name,
                "path": str(self.weights_path),
            },
        }

    def _load(self) -> None:
        path = self.weights_path
        if not path.is_file():
            self._load_error = f"GNN weights not found at {path}."
            logger.warning(self._load_error)
            return
        try:
            import torch
        except ImportError as exc:
            self._load_error = f"PyTorch is not installed: {exc}"
            logger.warning(self._load_error)
            return

        try:
            from backend.services.gnn_architecture import DistanceAwareGNN
        except Exception as exc:
            self._load_error = f"Failed to import DistanceAwareGNN architecture: {exc}"
            logger.exception(self._load_error)
            return

        preferred = (self.settings.gnn_device or "auto").strip().lower()
        if preferred in {"", "auto"}:
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        elif preferred.startswith("cuda"):
            if not torch.cuda.is_available():
                logger.warning("CUDA requested for GNN but unavailable; using CPU.")
                device_name = "cpu"
            else:
                device_name = preferred
        else:
            device_name = "cpu"

        device = torch.device(device_name)
        try:
            checkpoint = torch.load(path, map_location=device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(path, map_location=device)
        except Exception as exc:
            self._load_error = f"Failed to load GNN weights: {exc}"
            logger.exception(self._load_error)
            return

        if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
            self._load_error = "GNN checkpoint is missing model_state_dict."
            return

        camera_dim = int(checkpoint.get("camera_feature_dim") or CAMERA_FEATURE_DIM)
        extra_dim = int(checkpoint.get("candidate_extra_dim") or CANDIDATE_EXTRA_DIM)
        hidden = int(checkpoint.get("hidden") or 96)
        gru_hidden = int(checkpoint.get("gru_hidden") or 128)

        try:
            model = DistanceAwareGNN(
                camera_dim=camera_dim,
                candidate_extra_dim=extra_dim,
                hidden=hidden,
                gru_hidden=gru_hidden,
            )
            model.load_state_dict(checkpoint["model_state_dict"])
            model.to(device)
            model.eval()
        except Exception as exc:
            self._load_error = f"Failed to initialize DistanceAwareGNN: {exc}"
            logger.exception(self._load_error)
            return

        self._torch = torch
        self._model = model
        self._checkpoint = checkpoint
        self.device_name = device_name
        self._load_error = None
        logger.info("Loaded GNN from %s on %s", path, device_name)

    def _camera_fields(self, row: dict[str, Any]) -> dict[str, Any]:
        meta = _parse_metadata(row.get("metadata"))
        notes: list[str] = []
        degraded = False

        lat = row.get("latitude")
        lon = row.get("longitude")
        try:
            lat_f = float(lat) if lat is not None else None
            lon_f = float(lon) if lon is not None else None
        except (TypeError, ValueError):
            lat_f, lon_f = None, None
        if lat_f is None or lon_f is None or not (math.isfinite(lat_f) and math.isfinite(lon_f)):
            raise ValueError(
                f"Camera {row.get('camera_id')} is missing latitude/longitude."
            )

        elevation = row.get("elevation")
        try:
            elev_f = float(elevation) if elevation is not None else None
        except (TypeError, ValueError):
            elev_f = None
        if elev_f is None:
            elev_f = _metadata_number(meta, "elevation")
        if elev_f is None:
            elev_f = ELEVATION_DEFAULT
            degraded = True
            notes.append(
                f"{row.get('camera_id')}: elevation defaulted to {ELEVATION_DEFAULT}."
            )

        ecological: dict[str, float] = {}
        for name, default in ECOLOGICAL_DEFAULTS.items():
            found = _metadata_number(meta, name)
            if found is None:
                ecological[name] = default
                degraded = True
                notes.append(f"{row.get('camera_id')}: {name} defaulted to {default}.")
            else:
                ecological[name] = found

        habitat_raw = row.get("habitat") or _metadata_text(meta, "habitat")
        if not habitat_raw:
            degraded = True
            notes.append(f"{row.get('camera_id')}: habitat defaulted to 'other'.")
        habitat = map_habitat(habitat_raw if isinstance(habitat_raw, str) else None)
        if habitat_raw and habitat == "other" and map_habitat(str(habitat_raw)) == "other":
            # Free-text that we could not map still occupies the production 'other' slot.
            notes.append(
                f"{row.get('camera_id')}: habitat '{habitat_raw}' mapped to production slot 'other'."
            )
            degraded = True

        corridor = _metadata_number(meta, "corridor_quality")

        return {
            "camera_id": str(row["camera_id"]),
            "latitude": lat_f,
            "longitude": lon_f,
            "elevation": float(elev_f),
            "habitat": habitat,
            "habitat_raw": habitat_raw,
            "corridor_quality": corridor,
            "feature_degraded": degraded,
            "feature_notes": notes,
            **ecological,
        }

    def _build_live_graph(self, camera_rows: list[dict[str, Any]]) -> dict[str, Any]:
        from backend.services.gnn_architecture import haversine_km

        usable: list[dict[str, Any]] = []
        skipped_no_coords = 0
        all_notes: list[str] = []
        degraded = False

        for row in camera_rows:
            if not row.get("camera_id"):
                continue
            try:
                fields = self._camera_fields(row)
            except ValueError:
                skipped_no_coords += 1
                continue
            usable.append(fields)
            if fields["feature_degraded"]:
                degraded = True
            all_notes.extend(fields["feature_notes"])

        if len(usable) < 2:
            raise ValueError(
                "Need at least two cameras with latitude/longitude for GNN prediction."
            )

        # Habitat one-hots are a production encoding (training vocab was not saved).
        degraded = True
        all_notes.append(
            "Habitat one-hots use the production 4-slot mapping; "
            "training habitat vocabulary was not stored in the checkpoint."
        )
        all_notes.append(
            "Camera numeric features are z-scored on live cameras; "
            "training normalization statistics were not stored in the checkpoint."
        )

        n = len(usable)
        ids = [item["camera_id"] for item in usable]
        index = {cid: i for i, cid in enumerate(ids)}

        lat = np.asarray([item["latitude"] for item in usable], dtype=np.float32)
        lon = np.asarray([item["longitude"] for item in usable], dtype=np.float32)
        elevation = np.asarray([item["elevation"] for item in usable], dtype=np.float32)
        water = np.asarray([item["water_distance"] for item in usable], dtype=np.float32)
        road = np.asarray([item["road_distance"] for item in usable], dtype=np.float32)
        human = np.asarray([item["human_disturbance_base"] for item in usable], dtype=np.float32)
        prey = np.asarray([item["prey_density_base"] for item in usable], dtype=np.float32)
        sensitivity = np.asarray([item["sensitivity"] for item in usable], dtype=np.float32)
        habitat_code = np.asarray(
            [HABITAT_SLOTS.index(item["habitat"]) for item in usable],
            dtype=np.int64,
        )

        numeric = [lat, lon, elevation, water, road, human, prey, sensitivity]
        feats = [zscore_column(col) for col in numeric]
        for slot in range(len(HABITAT_SLOTS)):
            feats.append((habitat_code == slot).astype(np.float32))
        camera_x = np.stack(feats, axis=1).astype(np.float32)
        if camera_x.shape[1] != CAMERA_FEATURE_DIM:
            raise ValueError(
                f"Invalid graph: camera feature dim is {camera_x.shape[1]}, expected {CAMERA_FEATURE_DIM}."
            )

        adj = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            for j in range(i + 1, n):
                distance = float(
                    haversine_km(lat[i], lon[i], lat[j], lon[j])
                )
                weight = math.exp(-max(distance, 0.0) / 3.0)
                adj[i, j] = weight
                adj[j, i] = weight

        transitions = self.graph.get_camera_connections()
        for edge in transitions:
            if edge.source not in index or edge.target not in index:
                continue
            ia, ib = index[edge.source], index[edge.target]
            if ia == ib:
                continue
            weight = math.exp(-max(float(haversine_km(lat[ia], lon[ia], lat[ib], lon[ib])), 0.0) / 3.0)
            weight = max(weight, min(1.0, 0.25 * max(edge.weight, 1)))
            adj[ia, ib] = max(adj[ia, ib], weight)
            adj[ib, ia] = max(adj[ib, ia], weight)

        adj += np.eye(n, dtype=np.float32)
        degree = adj.sum(axis=1)
        if not np.all(np.isfinite(degree)) or np.any(degree <= 0):
            raise ValueError("Invalid graph: adjacency degree is not finite.")
        inv_sqrt = 1.0 / np.sqrt(np.maximum(degree, 1e-8))
        adj_norm = inv_sqrt[:, None] * adj * inv_sqrt[None, :]

        edge_quality: dict[tuple[str, str], float] = {}
        for item in usable:
            if item["corridor_quality"] is not None:
                # Per-camera metadata applies as a last-hop prior, not a pair.
                continue
        for edge in transitions:
            quality = min(1.0, 0.2 * max(edge.weight, 1))
            edge_quality[(edge.source, edge.target)] = quality
            edge_quality[(edge.target, edge.source)] = quality

        return {
            "ids": ids,
            "index": index,
            "lat": lat,
            "lon": lon,
            "water": water,
            "road": road,
            "human": human,
            "prey": prey,
            "sensitivity": sensitivity,
            "habitat_code": habitat_code,
            "camera_x": camera_x,
            "adj_norm": adj_norm.astype(np.float32),
            "edge_quality": edge_quality,
            "feature_degraded": degraded,
            "feature_notes": all_notes,
            "skipped_no_coords": skipped_no_coords,
        }

    def _encode_example(
        self,
        history: list[dict[str, Any]],
        live: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        from backend.services.gnn_architecture import haversine_km

        index: dict[str, int] = live["index"]
        hist_ids = [item["camera_id"] for item in history]
        missing = [cid for cid in hist_ids if cid not in index]
        if missing:
            raise ValueError(
                "History cameras are missing latitude/longitude: " + ", ".join(missing)
            )

        hist_idx = [index[cid] for cid in hist_ids]
        last_idx = hist_idx[-1]
        prev_idx = hist_idx[-2] if len(hist_idx) >= 2 else last_idx

        candidate_ids = [cid for cid in live["ids"] if cid != hist_ids[-1]]
        if not candidate_ids:
            raise ValueError("No candidate cameras besides the last observed camera.")

        last_lat = live["lat"][last_idx]
        last_lon = live["lon"][last_idx]
        candidate_ids.sort(
            key=lambda cid: float(
                haversine_km(
                    last_lat,
                    last_lon,
                    live["lat"][index[cid]],
                    live["lon"][index[cid]],
                )
            )
        )
        candidate_ids = candidate_ids[:MAX_CANDIDATES]
        cand_idx = [index[cid] for cid in candidate_ids]
        cand_arr = np.asarray(cand_idx, dtype=np.int64)

        hours: list[float] = []
        missing_time = False
        for item in history:
            value = timestamp_hours(item.get("timestamp"))
            if value is None:
                missing_time = True
                hours.append(float(len(hours)))
            else:
                hours.append(value)
        if missing_time:
            live = {
                **live,
                "feature_degraded": True,
                "feature_notes": list(live["feature_notes"])
                + ["One or more history timestamps were missing; time gaps were filled."],
            }

        ts = np.asarray(hours, dtype=np.float32)
        dts = np.zeros(HISTORY_LEN, dtype=np.float32)
        if HISTORY_LEN > 1:
            dts[1:] = np.maximum(0.0, np.diff(ts))
        dts = np.log1p(np.clip(dts, 0.0, 72.0)) / np.log1p(72.0)
        dts = dts.reshape(HISTORY_LEN, 1)

        d_last = haversine_km(
            live["lat"][last_idx],
            live["lon"][last_idx],
            live["lat"][cand_arr],
            live["lon"][cand_arr],
        )
        d_prev = haversine_km(
            live["lat"][prev_idx],
            live["lon"][prev_idx],
            live["lat"][cand_arr],
            live["lon"][cand_arr],
        )

        hist_habitats = live["habitat_code"][np.asarray(hist_idx)]
        values, counts = np.unique(hist_habitats, return_counts=True)
        preferred_habitat = values[int(np.argmax(counts))]
        habitat_match = (live["habitat_code"][cand_arr] == preferred_habitat).astype(
            np.float32
        )

        corridor = np.asarray(
            [
                live["edge_quality"].get(
                    (hist_ids[-1], cid),
                    CORRIDOR_DEFAULT,
                )
                for cid in candidate_ids
            ],
            dtype=np.float32,
        )
        if any(
            (hist_ids[-1], cid) not in live["edge_quality"] for cid in candidate_ids
        ):
            live = {
                **live,
                "feature_degraded": True,
                "feature_notes": list(live["feature_notes"])
                + [f"Missing corridor_quality defaulted to {CORRIDOR_DEFAULT}."],
            }

        extra = np.stack(
            [
                np.log1p(np.maximum(d_last, 0.0)) / np.log1p(20.0),
                np.log1p(np.maximum(d_prev, 0.0)) / np.log1p(20.0),
                habitat_match,
                live["prey"][cand_arr],
                live["human"][cand_arr],
                np.exp(-np.maximum(live["water"][cand_arr], 0.0) / 2.0),
                np.log1p(np.maximum(live["road"][cand_arr], 0.0)) / np.log1p(10.0),
                live["sensitivity"][cand_arr],
                corridor,
            ],
            axis=1,
        ).astype(np.float32)
        extra = np.nan_to_num(extra, nan=0.0, posinf=1.0, neginf=0.0)
        extra = np.clip(extra, -5.0, 5.0)

        cand_out = np.zeros(MAX_CANDIDATES, dtype=np.int64)
        extra_out = np.zeros((MAX_CANDIDATES, CANDIDATE_EXTRA_DIM), dtype=np.float32)
        mask_out = np.zeros(MAX_CANDIDATES, dtype=np.bool_)
        count = min(len(cand_idx), MAX_CANDIDATES)
        cand_out[:count] = np.asarray(cand_idx[:count], dtype=np.int64)
        extra_out[:count] = extra[:count]
        mask_out[:count] = True

        tensors = {
            "history_idx": np.asarray(hist_idx, dtype=np.int64),
            "history_dt": dts.astype(np.float32),
            "candidate_idx": cand_out,
            "candidate_extra": extra_out,
            "candidate_mask": mask_out,
            "candidate_ids": candidate_ids,
        }
        extras = {
            "feature_degraded": bool(live["feature_degraded"]),
            "feature_notes": list(dict.fromkeys(live["feature_notes"])),
        }
        return tensors, extras

    def _run_inference(
        self,
        live: dict[str, Any],
        tensors: dict[str, Any],
    ) -> list[dict[str, Any]]:
        torch = self._torch
        device = torch.device(self.device_name)
        camera_x = torch.from_numpy(live["camera_x"]).to(device)
        adj_norm = torch.from_numpy(live["adj_norm"]).to(device)
        history_idx = torch.from_numpy(tensors["history_idx"]).unsqueeze(0).to(device)
        history_dt = torch.from_numpy(tensors["history_dt"]).unsqueeze(0).to(device)
        candidate_idx = torch.from_numpy(tensors["candidate_idx"]).unsqueeze(0).to(device)
        candidate_extra = torch.from_numpy(tensors["candidate_extra"]).unsqueeze(0).to(
            device
        )
        candidate_mask = torch.from_numpy(tensors["candidate_mask"]).unsqueeze(0).to(
            device
        )

        self._model.eval()
        with torch.inference_mode():
            logits = self._model(
                camera_x,
                adj_norm,
                history_idx,
                history_dt,
                candidate_idx,
                candidate_extra,
                candidate_mask,
            )
            masked = logits[0].clone()
            valid = candidate_mask[0]
            if not bool(valid.any()):
                return []
            masked = masked.masked_fill(~valid, torch.finfo(masked.dtype).min)
            probabilities = torch.softmax(masked[valid].float(), dim=0)

        candidate_ids: list[str] = tensors["candidate_ids"]
        scores = probabilities.detach().cpu().tolist()
        ranked = []
        order = np.argsort([-score for score in scores])
        for rank, position in enumerate(order, start=1):
            ranked.append(
                {
                    "rank": rank,
                    "camera_id": candidate_ids[int(position)],
                    "confidence": float(scores[int(position)]),
                }
            )
        return ranked
