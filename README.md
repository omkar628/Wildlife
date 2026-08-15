# Wildlife Intelligence

Offline desktop workflow for camera-trap analysis.

This milestone takes a folder of images, runs the local YOLO11n detector
(`best.pt`), stores results in SQLite, and shows them in a React dashboard.
Tiger Re-ID and the GNN are wired as future plug-in points. They are **not**
implemented yet.

```
SD card / camera folder
        ↓
image ingestion + SHA-256
        ↓
YOLO (tiger / prey / rival / human)
        ↓
confidence filter
        ↓
SQLite + tiger crops
        ↓
dashboard / human review
```

## A. What was already in this folder

The repository started with **only** four trained assets. There was no Python
package, no notebook, no inference script, and no Git history.

| File | Size | What it is |
| --- | --- | --- |
| `best.pt` | ~5.5 MB | Trained YOLO11n detector. Classes: `0 tiger`, `1 prey`, `2 rival`, `3 human`. |
| `tiger_reid_arcface.pth` | ~190 MB | PyTorch Re-ID state dict (ConvNeXt-like backbone + 512-d bottleneck + ArcFace). |
| `tiger_vector_index.faiss` | ~3.7 MB | FAISS `IndexFlatIP`, header reports **512-d** and **1887** vectors. |
| `tiger_metadata.pkl` | ~160 KB | List of 1887 `{tiger_id, image_path}` records (107 IDs from the Amur Tiger dataset). |

Those four files were **not moved, renamed, copied, or modified**.

The application looks for them first under `models/detector` and
`models/reid`, then falls back to the project root (their current location).

## B. Files that must stay untouched

- `best.pt`
- `tiger_reid_arcface.pth`
- `tiger_vector_index.faiss`
- `tiger_metadata.pkl`

Do not delete, overwrite, convert, or recommit these.

## C. Tiger Re-ID status

The weights, FAISS index, and metadata are present. **The inference code is
not.** This repo does not contain the model class, preprocessing, embedding
normalization, or match threshold used when those files were trained.

Observed from the files (not sufficient to run identity):

- Checkpoint is a raw `state_dict`, not a full training script.
- Final layers: `bottleneck` 768→512, `bn` 512, `arcface.weight` shape `(107, 512)`.
- FAISS magic `IxFI` = inner-product index, 512-d, 1887 rows (matches the pickle).
- Gallery IDs are numeric strings such as `"250"`, not field IDs like `T017`.
- Pickle image paths point at a Kaggle dataset that is not on this machine.

Missing before Re-ID can be turned on:

1. The Python class whose `forward` returns the 512-d embedding.
2. Exact crop size, resize, color order, and normalization.
3. Whether embeddings are L2-normalized before FAISS.
4. The score threshold / top-k rule used to assign an ID.
5. Mapping from gallery IDs to field IDs (`T017`).

`backend/reid/` is the adapter. It reports this status and returns
`tiger_id = None` until the original inference code is provided.

## Prerequisites

- Windows 10/11
- **Python 3.10+** (this machine was checked with 3.10.8)
- **Node.js 20+** and npm (checked with Node 24 / npm 11)
- Git
- The four model files above, left in the project root
- Optional NVIDIA GPU + CUDA PyTorch. CPU works; it is slower.

No cloud account is required. Inference is local.

## 1. Open a terminal in the project folder

```bat
cd /d F:\WildlifeIntelligence
```

## 2. Python virtual environment

```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If `torch` is not already installed:

```bat
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

If you already have PyTorch in the same Python and want the venv to see it:

```bat
python -m venv .venv --system-site-packages
```

## 3. Frontend packages

```bat
cd frontend
npm install
cd ..
```

## 4. Start the backend (Terminal 1)

Always start from the project root so `backend.main:app` can be imported.

```bat
cd /d F:\WildlifeIntelligence
.venv\Scripts\activate
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

Or:

```bat
python main.py
```

API docs: http://127.0.0.1:8000/docs
Health: http://127.0.0.1:8000/api/health

## 5. Start the frontend (Terminal 2)

```bat
cd /d F:\WildlifeIntelligence\frontend
npm run dev
```

Open Chrome at **http://localhost:5173**.

Vite proxies `/api` to the FastAPI server, so leave both terminals running.
Frontend edits hot-reload. Backend edits reload via `--reload`.

## 6. Import a camera folder

1. Open **Import folder**.
2. Paste a path such as `D:\CameraTrap\C01`.
3. Set **Camera ID** to `C01` (or any id you use).
4. Optionally add latitude, longitude, elevation, habitat.
5. Adjust the auto-accept threshold if needed (default **0.60**).
6. Click **Start processing**.

The UI stays responsive. Progress is on the **Processing** page.

Rules:

- Recursive scan of `.jpg` / `.jpeg` / `.png`
- Original images are never deleted or modified
- Duplicate files are skipped by SHA-256 of file bytes
- A stopped job can be started again; completed hashes are not reprocessed
- Corrupt images are logged and skipped

## Where things live

| Item | Location |
| --- | --- |
| YOLO weights | `best.pt` (or `models/detector/best.pt`) |
| Re-ID assets | project root (or `models/reid/`) |
| Future GNN | `models/gnn/` (empty on purpose) |
| SQLite database | `database/wildlife.db` |
| Tiger crops | `data/crops/image_<id>/detection_<id>.jpg` |
| Logs | `logs/app.log` |
| Config | `config/settings.yaml` and optional `.env` |

Copy `.env.example` to `.env` to override paths or thresholds without editing code.

Thresholds are **not** hard-coded as `0.60` in the pipeline. Change them in:

- `config/settings.yaml` → `confidence.auto_accept`
- environment `WI_CONFIDENCE_AUTO_ACCEPT`
- the Import page / `PUT /api/settings`

Detections below `detect_min` (default 0.15) are discarded.
Detections at or above `auto_accept` are stored as accepted.
Everything in between goes to the review queue.

## Development workflow

Keep it to two terminals and Chrome. Do not package an `.exe` yet.

```
Terminal 1   uvicorn backend.main:app --reload
Terminal 2   cd frontend && npm run dev
Chrome       http://localhost:5173
```

## Tests

Tests use tiny generated JPEGs. They do **not** need your camera-trap dataset
and they do **not** load `best.pt`.

```bat
cd /d F:\WildlifeIntelligence
.venv\Scripts\activate
pytest
```

Covered:

- database init and foreign keys
- recursive ingestion
- SHA-256 hashing
- YOLO result parsing
- configurable confidence filter
- detection insert + review queue
- duplicate skip
- resume of pending/failed images
- API health, settings, import, review, graph

## Project layout

```
WildlifeIntelligence/
├── best.pt                         (untouched, discovered automatically)
├── tiger_reid_arcface.pth
├── tiger_vector_index.faiss
├── tiger_metadata.pkl
├── backend/
│   ├── main.py                     FastAPI app
│   ├── config.py
│   ├── api/routes/                 HTTP endpoints
│   ├── database/                   SQLite schema + repositories
│   ├── detector/                   YOLO service + parser
│   ├── ingestion/                  scan, hash, EXIF
│   ├── review/
│   ├── reid/                       adapter only
│   ├── graph/                      camera / observation graphs, no GNN
│   └── services/                   pipeline, crops, confidence
├── frontend/                       React + Vite
├── config/settings.yaml
├── database/wildlife.db            created on first run
├── models/detector|reid|gnn        placeholders; originals stay at root
├── tests/
├── requirements.txt
└── main.py
```

## Future plug-ins (not in this milestone)

- Re-ID: implement `TigerReIDBackend` in `backend/reid/` using the original training code.
- GNN: add `models/gnn/` and consume `GET /api/graph`. Do not change YOLO, SQLite, or ingestion.
- Frontend already treats graph data as events (`tiger_id`, `camera_id`, `timestamp`, `confidence`) so animation can be added later without touching inference.

## Troubleshooting

**`No module named backend`**
Run uvicorn from `F:\WildlifeIntelligence`, not from `backend\`.

**Detector not found**
Confirm `best.pt` is still in the project root. Check `/api/health`.

**Frontend cannot reach the API**
Start the backend first. Vite proxies `/api` to `127.0.0.1:8000`.

**`ultralytics` or OpenCV install fails**
Install Visual C++ redistributable, then `pip install -r requirements.txt` again.

**Processing is slow**
That is expected on CPU. Lower `detector.batch_size` in `config/settings.yaml` if RAM is tight. Install a CUDA build of PyTorch if you have a GPU.

**Same photos imported twice**
If the pixels are identical, SHA-256 will mark them as duplicates even when the filename or folder changed.

**Re-ID does nothing**
Expected. See “Tiger Re-ID status” above.

**Port 8000 or 5173 already in use**
Stop the other process, or change `WI_PORT` / the Vite `server.port`.

## Git

`.gitignore` excludes virtualenvs, `node_modules`, model weights (`*.pt`,
`*.pth`, `*.faiss`, `*.pkl`), SQLite files, and generated `data/` / `logs/`.
Do not commit the trained assets to a normal Git remote.
