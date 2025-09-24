import shutil
import pathlib

import click

from anime_recommender.scripts import setup, factory, callbacks

callback = callbacks.EventsCallback()
log = callback.logger


@click.group()
def data():
    """Data management commands."""
    pass


@click.group()
def s3():
    """S3 Bucket management commands."""
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

    archive_path = pathlib.Path("src") / "anime_recommender" / "data" / "archive.zip"
    ds_loader = setup.DatasetLoader(log=log, archive_path=archive_path)
    anime_pd, ratings_pd = ds_loader.load_pandas_data_frames()
    ds_processor = setup.DatasetProcessor(log=log, anime_pd=anime_pd, ratings_pd=ratings_pd)
    ds_processor.save_to_csv(filename=output)


@data.command()
@click.option("--seed", type=click.INT, default=42)
@click.option("--ratio", type=click.FloatRange(0.0, 1.0), default=0.7)
def split(ratio: float, seed: int):
    """Split the joined table into train/test, then write to CSV."""

    cxt_factory = factory.context_factory(log=log, ratio=ratio, seed=seed)
    _ = cxt_factory.split_and_write_train_test(write=True)


@data.command()
@click.option("--seed", type=click.INT, default=42)
@click.option("--ratio", type=click.FloatRange(0.0, 1.0), default=0.7)
def recordio_format(ratio: float, seed: int):
    """Write RecordIO-protobuf files for training and testing."""

    cxt_factory = factory.context_factory(log=log, ratio=ratio, seed=seed)
    cxt_factory.make_recordio_files()


@data.command()
@click.option("--seed", type=click.INT, default=42)
@click.option("--ratio", type=click.FloatRange(0.0, 1.0), default=0.7)
def svm_format(ratio: float, seed: int):
    """Write libSVM files for training and testing."""

    cxt_factory = factory.context_factory(log=log, ratio=ratio, seed=seed)
    cxt_factory.make_svmlight_files()


@data.command()
@click.option("--seed", type=click.INT, default=42)
@click.option("--ratio", type=click.FloatRange(0.0, 1.0), default=0.7)
def lookup_files(ratio: float, seed: int):
    """Create two one-hot-encoding lookup files mapping ID --> Index."""

    cxt_factory = factory.context_factory(log=log, ratio=ratio, seed=seed)
    cxt_factory.create_lookup_files()
