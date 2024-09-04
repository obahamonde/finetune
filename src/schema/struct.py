from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class FineTuningJobCreate(BaseModel):
    model: str
    training_file: str
    validation_file: Optional[str] = None
    hyperparameters: Optional[Dict[str, Any]] = None
    suffix: Optional[str] = None
    integrations: Optional[List[str]] = None
    seed: Optional[int] = None


class FineTuningJobUpdate(BaseModel):
    metadata: Optional[Dict[str, Any]] = None


class FineTuningJobResponse(BaseModel):
    id: str
    object: str = Field(default="fine_tuning.job")
    created_at: int
    model: str
    training_file: str
    validation_file: Optional[str] = None
    hyperparameters: Dict[str, Any]
    status: str
    fine_tuned_model: Optional[str] = None
    organization_id: str
    result_files: List[str]
    trained_tokens: Optional[int] = None
    error: Optional[Dict[str, Any]] = None
    finished_at: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    integrations: Optional[List[str]] = None
    seed: Optional[int] = None
    estimated_finish: Optional[int] = None


class FineTuningJobList(BaseModel):
    object: str = Field(default="list")
    data: List[FineTuningJobResponse]
    has_more: bool


class FineTuningEventResponse(BaseModel):
    object: str = Field(default="fine_tuning.job.event")
    id: str
    created_at: int
    level: str
    message: str
    data: Optional[Dict[str, Any]] = None
    type: str


class FineTuningEventList(BaseModel):
    object: str = Field(default="list")
    data: List[FineTuningEventResponse]
    has_more: bool


class FineTuningCheckpointResponse(BaseModel):
    object: str = Field(default="fine_tuning.job.checkpoint")
    id: str
    created_at: int
    fine_tuned_model_checkpoint: str
    metrics: Dict[str, Any]
    fine_tuning_job_id: str
    step_number: int


class FineTuningCheckpointList(BaseModel):
    object: str = Field(default="list")
    data: List[FineTuningCheckpointResponse]
    has_more: bool
    first_id: Optional[str] = None
    last_id: Optional[str] = None
