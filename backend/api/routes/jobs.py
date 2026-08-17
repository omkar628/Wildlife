from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from backend.api.deps import db_dep, get_pipeline
from backend.api.schemas import BatchImportRequest, ImportPreviewRequest, ImportRequest
from backend.database.connection import Database
from backend.database.repositories import CameraRepository, ErrorRepository, JobRepository
from backend.ingestion.scanner import discover_camera_folders, match_folder_to_camera
from backend.services.pipeline import PipelineService

router = APIRouter(tags=["jobs"])


@router.post("/import/preview")
def preview_import(
    payload: ImportPreviewRequest,
    db: Database = Depends(db_dep),
) -> dict:
    try:
        preview = discover_camera_folders(payload.folder_path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    known = {row["camera_id"] for row in CameraRepository(db).list_all()}
    folders = []
    for item in preview["camera_folders"]:
        match = match_folder_to_camera(item["folder_name"], known)
        folders.append(
            {
                **item,
                "suggested_camera_id": match["suggested_camera_id"],
                "camera_exists": not match["unknown_camera_folder"],
                "match_status": match["match_status"],
                "unknown_camera_folder": match["unknown_camera_folder"],
            }
        )
    preview["camera_folders"] = folders
    preview["known_cameras"] = sorted(known)
    return preview


@router.post("/import/batch")
def start_batch_import(
    payload: BatchImportRequest,
    pipeline: PipelineService = Depends(get_pipeline),
) -> dict:
    jobs = []
    for item in payload.cameras:
        try:
            jobs.append(
                pipeline.start_import(
                    folder_path=item.folder_path,
                    camera_id=item.camera_id,
                    latitude=item.latitude,
                    longitude=item.longitude,
                    elevation=item.elevation,
                    habitat=item.habitat,
                    create_if_missing=item.create_if_missing,
                    name=item.name,
                )
            )
        except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"jobs": jobs}


@router.post("/import")
def start_import(
    payload: ImportRequest,
    pipeline: PipelineService = Depends(get_pipeline),
) -> dict:
    try:
        return pipeline.start_import(
            folder_path=payload.folder_path,
            camera_id=payload.camera_id,
            latitude=payload.latitude,
            longitude=payload.longitude,
            elevation=payload.elevation,
            habitat=payload.habitat,
            create_if_missing=payload.create_if_missing,
            name=payload.name,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except NotADirectoryError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/jobs")
def list_jobs(db: Database = Depends(db_dep)) -> dict:
    return {"jobs": JobRepository(db).list_recent(50)}


@router.get("/jobs/{job_id}")
def get_job(job_id: int, db: Database = Depends(db_dep)) -> dict:
    job = JobRepository(db).get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    errors = ErrorRepository(db).list_for_job(job_id)
    return {"job": job, "errors": errors, "elapsed_note": job.get("error_message")}


@router.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: int, pipeline: PipelineService = Depends(get_pipeline), db: Database = Depends(db_dep)) -> dict:
    job = JobRepository(db).get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    pipeline.cancel(job_id)
    return {"ok": True, "job_id": job_id}
