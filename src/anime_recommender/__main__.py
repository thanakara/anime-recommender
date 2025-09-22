import pathlib

import click

from anime_recommender.scripts import setup, callbacks

callback = callbacks.EventCB()
log = callback.logger


@click.group()
def cli():
    pass


@cli.command()
@click.option("--raw", type=bool, default=False)
@click.option("-o", "--output", type=str, default="anime-genre.csv")
@click.option("--split", type=bool, default=False)
@click.option("--ratio", type=float, default=0.7)
@click.option("--seed", type=int, default=42)
def data(raw: bool, output: str, split: bool, ratio: float, seed: int):
    archive_path = pathlib.Path("src") / "anime_recommender" / "data" / "archive.zip"
    loader = setup.DatasetLoader(log=log, archive_path=archive_path)
    anime_pd, ratings_pd = loader.load_pandas_data_frames()
    processor = setup.DatasetProcessor(log=log, anime_pd=anime_pd, ratings_pd=ratings_pd)
    if raw:
        processor.save_to_csv(filename=output)
    if split:
        data = processor._merge()
        ds_context = setup.DatasetContext(log=log, train_split_ratio=ratio, data=data, seed=seed)
        _ = ds_context.split_and_write_train_test(write=True)


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
