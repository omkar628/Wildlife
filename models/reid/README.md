# Tiger Re-ID assets

Expected files:

- `tiger_reid_arcface.pth`
- `tiger_vector_index.faiss`
- `tiger_metadata.pkl`

These currently live at the **project root** and must not be moved or modified
by the application.

The app discovers them automatically. Re-ID inference is **not implemented**
because the original model class and preprocessing code are not in this repo.
See the README section “Tiger Re-ID status”.
