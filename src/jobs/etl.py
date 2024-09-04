import os
import awswrangler as wr
import threading
import pandas as pd  # type: ignore
import boto3
from botocore.client import Config
from dotenv import load_dotenv
from .library import ETLJob


load_dotenv()

job = ETLJob[pd.DataFrame]()

session = boto3.Session(
    aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    region_name="us-east-1",
    profile_name="default",
)

s3 = session.client(
    "s3",
    config=Config(
        read_timeout=1000, retries={"max_attempts": 10}, signature_version="s3v4"
    ),
)


class PandasJob(ETLJob[pd.DataFrame]):
    pass


job = PandasJob()


@job.extract
def extract(*, file_path: str):
    job.data = {"0": pd.read_json(file_path, lines=True)}
    return job.data


@job.transform
def transform():
    assert job.data, "DataFrame not found"
    df = job.data["0"]
    df = df.dropna()
    df = df.drop_duplicates()
    job.data["0"] = df
    return job.data


@job.load
def load(*, path: str):
    assert job.data, "DataFrame not found"
    value = wr.s3.to_json(
        df=job.data["0"],
        path=path,
        index=False,
        mode="overwrite",
        boto3_session=session,
        dataset=True,
        compression="gzip",
    )
    job.report.logger.info(value)
    return job.data


@job
def pipeline(*, file_path: str, path: str):
    with threading.Lock():
        extract(file_path=file_path)
        transform()
        return load(path=path)


def main():
    path = "./data/dataset.jsonl"
    dest = f"s3://{os.environ['AWS_S3_BUCKET']}/data"
    df = pipeline(file_path=path, path=dest)
    job.report.logger.info(f"DataFrame loaded to {dest}")
    job.report.logger.info(df["0"].head())
