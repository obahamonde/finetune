from fastapi import APIRouter, HTTPException, Request

from .schema import (
    FineTuningCheckpointList,
    FineTuningEventList,
    FineTuningJobCreate,
    FineTuningJobList,
    FineTuningJobResponse,
    FineTuningJobUpdate,
)
from .worker import (
    cancel_fine_tuning_job,
    create_fine_tuning_job,
    delete_fine_tuning_job,
    get_fine_tuning_job,
    list_fine_tuning_checkpoints,
    list_fine_tuning_events,
    list_fine_tuning_jobs,
    update_fine_tuning_job,
)

app = APIRouter()


@app.post("/fine_tuning/jobs", response_model=FineTuningJobResponse)
async def create_job(req: Request, job: FineTuningJobCreate):
    try:
        org_id = req.headers.get("X-Org-Id", None)
        if not org_id:
            raise HTTPException(status_code=400, detail="X-Org-Id header is required")
        return await create_fine_tuning_job(org_id=org_id, job=job)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/fine_tuning/jobs/{job_id}", response_model=FineTuningJobResponse)
async def get_job(job_id: str):
    job = await get_fine_tuning_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Fine-tuning job not found")
    return job


@app.post("/fine_tuning/jobs/{job_id}", response_model=FineTuningJobResponse)
async def update_job(job_id: str, job_update: FineTuningJobUpdate):
    updated_job = await update_fine_tuning_job(job_id, job_update)
    if not updated_job:
        raise HTTPException(status_code=404, detail="Fine-tuning job not found")
    return updated_job


@app.delete("/fine_tuning/jobs/{job_id}", response_model=FineTuningJobResponse)
async def delete_job(job_id: str):
    deleted_job = await delete_fine_tuning_job(job_id)
    if not deleted_job:
        raise HTTPException(status_code=404, detail="Fine-tuning job not found")
    return deleted_job


@app.get("/fine_tuning/jobs", response_model=FineTuningJobList)
async def list_jobs(after: str, limit: int = 20):
    return await list_fine_tuning_jobs(after, limit)


@app.get("/fine_tuning/jobs/{job_id}/events", response_model=FineTuningEventList)
async def list_events(job_id: str, after: str, limit: int = 20):
    return await list_fine_tuning_events(job_id, after, limit)


@app.get(
    "/fine_tuning/jobs/{job_id}/checkpoints", response_model=FineTuningCheckpointList
)
async def list_checkpoints(job_id: str, after: str, limit: int = 10):
    return await list_fine_tuning_checkpoints(job_id, after, limit)


@app.post("/fine_tuning/jobs/{job_id}/cancel", response_model=FineTuningJobResponse)
async def cancel_job(job_id: str):
    cancelled_job = await cancel_fine_tuning_job(job_id)
    if not cancelled_job:
        raise HTTPException(status_code=404, detail="Fine-tuning job not found")
    return cancelled_job
