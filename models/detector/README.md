# Detector weights

Expected file:

- `best.pt` — trained YOLO11n (classes: tiger, prey, rival, human)

Your trained file currently lives at the **project root**:

    WildlifeIntelligence/best.pt

The application looks for weights in this order:

1. `models/detector/best.pt`
2. `./best.pt` (current location)

Do **not** delete, rename, or overwrite the original `best.pt`.
If you later want this folder to hold the file, copy it here yourself.
The app will keep working with the root-level file until then.
