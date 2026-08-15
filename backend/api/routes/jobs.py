from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from backend.api.deps import db_dep, get_pipeline
from backend.api.schemas import ImportRequest
from backend.database.connection import Database
from backend.database.repositories import ErrorRepository, JobRepository
from backend.services.pipeline import PipelineService

router = APIRouter(tags=["jobs"])


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
    elapsed = None
    if job.get("started_at") and job.get("finished_at"):
        elapsed = None
    return {"job": job, "errors": errors, "elapsed_note": elapsed}


@router.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: int, pipeline: PipelineService = Depends(get_pipeline), db: Database = Depends(db_dep)) -> dict:
    job = JobRepository(db).get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    pipeline.cancel(job_id)
    return {"ok": True, "job_id": job_id}
