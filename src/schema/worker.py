import time
from typing import Optional, List
from uuid import uuid4
from pydantic import BaseModel
from fastapi import HTTPException
from prisma import Prisma, Json

from .struct import (
    FineTuningJobCreate,
    FineTuningJobUpdate,
    FineTuningJobResponse,
    FineTuningJobList,
    FineTuningEventResponse,
    FineTuningEventList,
    FineTuningCheckpointResponse,
    FineTuningCheckpointList,
)

from pyspark.sql import SparkSession
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset
import awswrangler as wr

def gen_id():
    return f"ft-{uuid4().hex[:24]}"


class SparkFineTuningService:
    def __init__(self, spark: SparkSession):
        self.spark = spark
        
    async def setup(self, db: Prisma, spark: SparkSession):
        await db.connect()
        
    async def teardown(self, db: Prisma, spark: SparkSession):
        await db.disconnect()
        spark.stop()
    async def create_fine_tuning_job(
        self, db: Prisma, *, org_id: str, job: FineTuningJobCreate
    ) -> FineTuningJobResponse:
        current_time = int(time.time())
        db_job = await db.finetuningjob.create(
            data={
                "id": gen_id(),
                "model": job.model,
                "training_file": job.training_file,
                "validation_file": job.validation_file,
                "hyperparameters": Json(job.hyperparameters or {}),
                "created_at": current_time,
                "status": "queued",
                "organization_id": org_id,
                "integrations": Json(job.integrations or []),
                "seed": job.seed,
            }
        )
        # Trigger the fine-tuning process asynchronously
        await self.run_fine_tuning_job(id=db_job.id)
        return FineTuningJobResponse(**db_job.model_dump())

    async def get_fine_tuning_job(self, db: Prisma, *, job_id: str) -> FineTuningJobResponse:
        db_job = await db.finetuningjob.find_unique(where={"id": job_id})
        if db_job:
            return FineTuningJobResponse(**db_job.model_dump())
        raise HTTPException(status_code=404, detail="Fine-tuning job not found")

    async def update_fine_tuning_job(
        self, db: Prisma, *, job_id: str, job_update: FineTuningJobUpdate
    ) -> FineTuningJobResponse:
        if job_update.metadata is not None:
            db_job = await db.finetuningjob.update(
                where={"id": job_id}, data={"metadata": Json(job_update.metadata)}
            )
            if db_job:
                return FineTuningJobResponse(**db_job.model_dump())
        raise HTTPException(status_code=404, detail="Fine-tuning job not found")

    async def list_fine_tuning_jobs(
        self, db: Prisma, *, after: Optional[str], limit: int = 20
    ) -> FineTuningJobList:
        jobs = await db.finetuningjob.find_many(
            take=limit,
            cursor={"id": after} if after else None,
            order={"created_at": "desc"},
        )
        return FineTuningJobList(
            data=[FineTuningJobResponse(**job.model_dump()) for job in jobs],
            has_more=len(jobs) == limit,
        )

    async def list_fine_tuning_events(
        self, db: Prisma, *, job_id: str, after: Optional[str], limit: int = 20
    ) -> FineTuningEventList:
        events = await db.finetuningevent.find_many(
            where={"fine_tuning_job_id": job_id},
            take=limit,
            cursor={"id": after} if after else None,
            order={"created_at": "desc"},
        )
        return FineTuningEventList(
            data=[FineTuningEventResponse(**event.model_dump()) for event in events],
            has_more=len(events) == limit,
        )

    async def list_fine_tuning_checkpoints(
        self, db: Prisma, *, job_id: str, after: Optional[str], limit: int = 10
    ) -> FineTuningCheckpointList:
        checkpoints = await db.finetuningcheckpoint.find_many(
            where={"fine_tuning_job_id": job_id},
            take=limit,
            cursor={"id": after} if after else None,
            order={"created_at": "desc"},
        )
        return FineTuningCheckpointList(
            data=[
                FineTuningCheckpointResponse(**checkpoint.model_dump())
                for checkpoint in checkpoints
            ],
            has_more=len(checkpoints) == limit,
            first_id=checkpoints[0].id if checkpoints else None,
            last_id=checkpoints[-1].id if checkpoints else None,
        )

    async def cancel_fine_tuning_job(self, db: Prisma, *, job_id: str) -> FineTuningJobResponse:
        db_job = await db.finetuningjob.update(
            where={"id": job_id}, data={"status": "cancelled"}
        )
        if not db_job:
            raise HTTPException(status_code=404, detail="Fine-tuning job not found")
        return FineTuningJobResponse(**db_job.model_dump())

    async def run_fine_tuning_job(self, db: Prisma, job_id: str):
        # This method should be run in a separate thread or process
        # It handles the actual fine-tuning process using Spark and transformers
        job = await self.get_fine_tuning_job(db=db, job_id=job_id)
        if not job:
            return
        try:
            # Update job status
            await self.update_job_status(db=db, job_id=job_id, status="running")

            # Prepare data using Spark
            train_dataset = self.prepare_dataset(file_path=job.training_file)
            val_dataset = self.prepare_dataset(file_path=job.validation_file) if job.validation_file else None

            # Load model and tokenizer
            model = AutoModelForCausalLM.from_pretrained(job.model)
            tokenizer = AutoTokenizer.from_pretrained(job.model)

            # Set up training arguments
            training_args = TrainingArguments(
                output_dir=f"/tmp/{job_id}",
                num_train_epochs=job.hyperparameters.get("num_epochs", 3),
                per_device_train_batch_size=job.hyperparameters.get("batch_size", 4),
                learning_rate=job.hyperparameters.get("learning_rate", 2e-5),
                fp16=True,
                save_steps=1000,
                logging_steps=100,
                save_total_limit=2,
            )

            # Set up trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
            )

            # Train the model
            trainer.train()

            # Save the model
            output_path = f"s3://indiecloud-data/models/{job_id}"
            trainer.save_model(output_path)

            # Update job status and set fine_tuned_model
            await self.update_job_status(db=db, job_id=job_id, status="succeeded")
            await self.update_fine_tuned_model(db=db, job_id=job_id, model_name=f"{job.model}-ft-{job_id}")

        except Exception as e:
            # Update job status to failed and log error
            await self.update_job_status(db=db, job_id=job_id, status="failed")
            await self.log_error(db=db, job_id=job_id, error_message=str(e))

    def prepare_dataset(self, file_path: str) -> Dataset:
        # Read JSONL file from S3 using awswrangler
        df = wr.s3.read_json(path=file_path)
        
        # Convert pandas DataFrame to Spark DataFrame
        spark_df = self.spark.createDataFrame(df)
        
        # Perform any necessary transformations using Spark
        # ...

        # Convert back to a format compatible with HuggingFace's datasets
        pandas_df = spark_df.toPandas()
        return Dataset.from_pandas(pandas_df)

    async def update_job_status(self, db: Prisma, job_id: str, status: str):
        await db.finetuningjob.update(where={"id": job_id}, data={"status": status})

    async def update_fine_tuned_model(self, db: Prisma, job_id: str, model_name: str):
        # Update fine_tuned_model in the database
        await db.finetuningjob.update(where={"id": job_id}, data={"fine_tuned_model": model_name})

    async def log_error(self, db: Prisma, job_id: str, error_message: str):
        # Log error in the database
        await db.finetuningjob.update(where={"id": job_id}, data={"error": Json({"message": error_message})})
