import shutil
import pathlib

import click

from anime_recommender.scripts import setup, callbacks

callback = callbacks.EventCB()
log = callback.logger


@click.group()
def cli():
    pass


@cli.command()
@click.option("--raw", is_flag=True, default=False)
@click.option("-o", "--output", type=str, default="anime-genre.csv")
@click.option("--split", is_flag=True, default=False)
@click.option("--ratio", type=float, default=0.7)
@click.option("--seed", type=int, default=42)
@click.option("--rmtree", is_flag=True, default=False)
def data(raw: bool, output: str, split: bool, ratio: float, seed: int, rmtree: bool):
    archive_path = pathlib.Path("src") / "anime_recommender" / "data" / "archive.zip"
    if raw:
        loader = setup.DatasetLoader(log=log, archive_path=archive_path)
        anime_pd, ratings_pd = loader.load_pandas_data_frames()
        processor = setup.DatasetProcessor(log=log, anime_pd=anime_pd, ratings_pd=ratings_pd)
        processor.save_to_csv(filename=output)
        if split:
            data = processor._merge()
            ds_context = setup.DatasetContext(log=log, train_split_ratio=ratio, data=data, seed=seed)
            _ = ds_context.split_and_write_train_test(write=True)
    elif rmtree:
        datapath = archive_path.parent.joinpath("raw")
        if datapath.exists():
            shutil.rmtree(datapath, ignore_errors=True)
            log.info(f"Remove tree: {datapath.as_posix()} @on_job_end")
        else:
            log.debug("{datapath} doesn't exist")

    else:
        log.info("No action taken")


@cli.command()
@click.option("-o", "--output")
def recordio(output: str):
    pass


@cli.command()
@click.option("-o", "--output")
def svm(output: str):
    pass


@cli.command()
@click.option("-o", "--output")
def lookup(output: str):
    pass
