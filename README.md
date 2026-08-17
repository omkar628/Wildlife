# Wildlife Intelligence

Offline desktop workflow for camera-trap analysis.

This app takes a folder of images, runs the local YOLO11n detector
(`best.pt`), stores results in SQLite, and shows them in a React dashboard.
Local tiger identity uses MegaDescriptor-S-224 (T001+ IDs). Next-station
ranking uses the shipped GNN V3.1 weights. The ATRW ArcFace/FAISS gallery
is inspected only and is never assigned to field tigers.

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
MegaDescriptor local Re-ID (T001+)
        ↓
dashboard / human review / GNN next-station
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

**Production identity is local MegaDescriptor + T001+ IDs.** First IDs are
created by a reviewer. Later high-confidence matches (default cosine ≥ 0.90)
can auto-assign only after the YOLO class is accepted or confirmed as tiger.
ATRW numeric gallery IDs such as `"250"` are rejected.

The shipped ATRW files (`tiger_reid_arcface.pth`, `tiger_vector_index.faiss`,
`tiger_metadata.pkl`) remain in the repo for inspection. They are **not**
used to assign field IDs. MegaDescriptor is loaded from
`BVRA/MegaDescriptor-S-224` (Hugging Face cache after the first download).

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

## 3. Node packages

Install once from the **project root** (Electron) and once in `frontend` (Vite/React):

```bat
cd /d F:\WildlifeIntelligence
npm install
cd frontend
npm install
cd ..
```

## 4. Start the desktop app (one command)

```bat
cd /d F:\WildlifeIntelligence
npm run dev
```

This starts:

1. Python/FastAPI on http://127.0.0.1:8000
2. Vite on http://127.0.0.1:5173 (hot reload)
3. An Electron desktop window that loads the Vite UI

You should **not** need to type a folder path. Use **Select Camera Folder**.

Optional: keep a browser preview at http://localhost:5173 with `npm run dev:web`,
but the native folder picker only works inside the Electron window.

API docs: http://127.0.0.1:8000/docs
Health: http://127.0.0.1:8000/api/health

To start only the backend by hand:

```bat
.venv\Scripts\activate
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

## 5. Import a camera folder

1. In the desktop window open **Import folder**.
2. Click **Select Camera Folder**.
3. Choose an SD card or local folder in the Windows dialog (for example `D:\CameraTrap\C01`).
4. Confirm or edit the Camera ID (filled from the folder name when possible).
5. Optionally add latitude, longitude, elevation, habitat.
6. Adjust the auto-accept threshold if needed (default **0.60**).
7. Click **Start processing**.

The UI stays responsive. Progress is on the **Processing** page.

Rules:

- Recursive scan of `.jpg` / `.jpeg` / `.png` / `.webp`
- Original images are never deleted or modified
- Duplicate files are skipped by SHA-256 of file bytes
- A stopped job can be started again; completed hashes are not reprocessed
- Corrupt images are logged and skipped

## Where things live

| Item | Location |
| --- | --- |
| YOLO weights | `best.pt` (or `models/detector/best.pt`) |
| Re-ID assets | project root (or `models/reid/`) |
| GNN V3.1 weights | `gnn_model_v3_optimized_best.pt` (or `models/gnn/`) |
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

One terminal. Do not package an `.exe` yet.

```
Project folder:
    npm run dev

Electron window opens.
Vite hot-reloads UI changes.
FastAPI --reload picks up backend changes.
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
│   ├── reid/                       MegaDescriptor + local T001+ identity
│   ├── graph/                      camera / observation graphs
│   └── services/                   pipeline, crops, confidence, GNN V3.1
├── electron/                       Desktop shell + native folder picker
├── frontend/                       React + Vite
├── config/settings.yaml
├── database/wildlife.db            created on first run
├── models/detector|reid|gnn        placeholders; originals stay at root
├── tests/
├── requirements.txt
└── main.py
```

## Identity and GNN

- Re-ID: MegaDescriptor-S-224 embeds tiger crops into a local gallery. Humans
  create T001, T002, … IDs. Auto-assign only happens above
  `reid.match_threshold` and only for accepted tiger detections.
- GNN: `GET /api/graph` and `GET /api/graph/tigers/{id}/route` rank the next
  camera from identified history. Need 5 identified observations and cameras
  with latitude/longitude. Do not change YOLO, SQLite, or GNN architecture.

## Troubleshooting

**`No module named backend`**
Run uvicorn from `F:\WildlifeIntelligence`, not from `backend\`.

**Detector not found**
Confirm `best.pt` is still in the project root. Check `/api/health`.

**Frontend cannot reach the API**
Use `npm run dev` from the project root so backend, Vite, and Electron start together.

**Select Camera Folder does nothing / says desktop app required**
You are in a browser tab. Close Chrome and run `npm run dev` so the Electron window opens.

**Folder picker cannot see the SD card**
Wait until Windows assigns a drive letter, then click Select Camera Folder again.

**`ultralytics` or OpenCV install fails**
Install Visual C++ redistributable, then `pip install -r requirements.txt` again.

**Processing is slow**
That is expected on CPU. Lower `detector.batch_size` in `config/settings.yaml` if RAM is tight. Install a CUDA build of PyTorch if you have a GPU.

**Same photos imported twice**
If the pixels are identical, SHA-256 will mark them as duplicates even when the filename or folder changed.

**Re-ID stays on human confirm**
If MegaDescriptor is disabled (`WI_REID_ENABLED=false`) or failed to load,
identity is human-only. Check `/api/health` → `reid.loaded`.

**Port 8000 or 5173 already in use**
Stop the other process, or change `WI_PORT` / the Vite `server.port`.

## Git

`.gitignore` excludes virtualenvs, `node_modules`, model weights (`*.pt`,
`*.pth`, `*.faiss`, `*.pkl`), SQLite files, and generated `data/` / `logs/`.
Do not commit the trained assets to a normal Git remote.
