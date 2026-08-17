# Tiger Re-ID assets

Expected files:

- `tiger_reid_arcface.pth`
- `tiger_vector_index.faiss`
- `tiger_metadata.pkl`

These currently live at the **project root** and must not be moved or modified
by the application.

The app discovers them automatically. They are inspected only.

Production identity uses **MegaDescriptor-S-224** and local T001+ IDs.
The ATRW ArcFace/FAISS gallery is never assigned to field tigers.
