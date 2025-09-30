import shutil
import pathlib

import click

from omegaconf import OmegaConf

from anime_recommender.constants import Filepath
from anime_recommender.scripts.setup import DatasetLoader, DatasetProcessor
from anime_recommender.scripts.factory import context_factory
from anime_recommender.scripts.runtime import ARSTrainer, delete_endpoint, create_endpoint_from_training_job
from anime_recommender.scripts.boto_sdk import upload_to_s3, create_bucket
from anime_recommender.scripts.callbacks import EventsCallback
from anime_recommender.scripts.resolvers import get_latest_job_name, region_name_resolver, execution_role_resolver

callback = EventsCallback()
log = callback.logger
file_ = Filepath.aws_uris_config_path
config = OmegaConf.load(file_=file_)
OmegaConf.register_new_resolver(name="region_resolver", resolver=region_name_resolver)
OmegaConf.register_new_resolver(name="role_resolver", resolver=execution_role_resolver)
OmegaConf.register_new_resolver(name="latest_job_name", resolver=get_latest_job_name)


@click.group()
def data():
    """Data management commands."""
    pass


@click.group()
def s3():
    """S3 Bucket management commands."""
    pass


@click.group()
def job():
    """SageMaker Job management commands."""
    pass


@data.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--confirm", is_flag=True, help="Skip confirmation prompt")
def rmtree(path: str, confirm: bool):
    """Remove data directory in order to free space."""

    if not confirm:
        click.confirm(f"Remove {path} tree?", abort=True)
    shutil.rmtree(pathlib.Path(path), ignore_errors=True)


@data.command()
@click.option("-o", "--output", default="anime-genre.csv")
def load(output: str):
    """Unpacks archive, joins the tables and writes CSV file."""

    archive_path = Filepath.archive_path
    ds_loader = DatasetLoader(log=log, archive_path=archive_path)
    anime_pd, ratings_pd = ds_loader.load_pandas_data_frames()
    ds_processor = DatasetProcessor(log=log, anime_pd=anime_pd, ratings_pd=ratings_pd)
    ds_processor.save_to_csv(filename=output)


@data.command()
@click.option("--seed", type=click.INT, default=42)
@click.option("--ratio", type=click.FloatRange(0.0, 1.0), default=0.7)
def split(ratio: float, seed: int):
    """Split the joined table into train/test, then write to CSV."""

    cxt_factory = context_factory(log=log, ratio=ratio, seed=seed)
    _ = cxt_factory.split_and_write_train_test(write=True)


@data.command()
@click.option("--seed", type=click.INT, default=42)
@click.option("--ratio", type=click.FloatRange(0.0, 1.0), default=0.7)
def recordio_format(ratio: float, seed: int):
    """Write RecordIO-protobuf files for training and testing."""

    cxt_factory = context_factory(log=log, ratio=ratio, seed=seed)
    cxt_factory.make_recordio_files()


@data.command()
@click.option("--seed", type=click.INT, default=42)
@click.option("--ratio", type=click.FloatRange(0.0, 1.0), default=0.7)
def svm_format(ratio: float, seed: int):
    """Write libSVM files for training and testing."""

    cxt_factory = context_factory(log=log, ratio=ratio, seed=seed)
    cxt_factory.make_svmlight_files()


@data.command()
@click.option("--seed", type=click.INT, default=42)
@click.option("--ratio", type=click.FloatRange(0.0, 1.0), default=0.7)
def lookup_files(ratio: float, seed: int):
    """Create two one-hot-encoding lookup files mapping ID --> Index."""

    cxt_factory = context_factory(log=log, ratio=ratio, seed=seed)
    cxt_factory.create_lookup_files()


@s3.command()
def create():
    """Create S3-bucket. Name specified on YAML"""

    create_bucket(config=config)


@s3.command()
@click.option("-f", "--filename", type=click.Path(exists=True), required=True)
@click.option("--key", type=click.STRING, required=True)
def upload(filename: str, key: str):
    """Upload file into desired S3-bucket"""

    filename = pathlib.Path(filename)
    upload_to_s3(config=config, filename=filename, key=key)


@job.command()
@click.option("-c", "--cfg", type=click.Path(exists=True), required=True)
def train(cfg: str):
    """Begins the Training job. All kwargs are in the DictConfig"""

    config = OmegaConf.load(file_=cfg)
    trainer = ARSTrainer(config=config)
    trainer.trainjob()


@job.command()
def deploy():
    """Creates endpoint from the training job and returns the endpoint name."""

    endpoint_name = create_endpoint_from_training_job(config=config)
    click.echo(f"Creating {endpoint_name}")


@job.command()
def cleanup():
    """Full cleanup of: Endpoint, EndpointConfig and Model."""

    delete_endpoint(config=config)
    click.echo("Cleanup completed")
