from pathlib import Path

import boto3

from omegaconf import DictConfig


def create_bucket(config: DictConfig) -> None:
    s3 = boto3.client(config.source)
    s3.create_bucket(Bucket=config.s3_bucket_name, CreateBucketConfiguration={"LocationConstraint": config.region})


def upload_to_s3(config: DictConfig, filename: str | Path, key: str) -> None:
    s3 = boto3.client(config.source)
    s3.upload_file(filename, config.s3_bucket_name, key)
