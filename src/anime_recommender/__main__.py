import pathlib

import click

from anime_recommender.scripts import setup, callbacks


@click.group()
def cli():
    pass


@cli.command()
@click.option("--raw", type=bool, default=False)
@click.option("-fn", "--filename", type=str, default="anime-genre.csv")
def data(raw: bool, filename: str):
    callback = callbacks.EventCB()
    log = callback.logger
    if raw:
        archive_path = pathlib.Path("src") / "anime_recommender" / "data" / "archive.zip"
        loader = setup.DatasetLoader(log=log, archive_path=archive_path)
        anime_pd, ratings_pd = loader.load_pandas_data_frames()
        processor = setup.DatasetProcessor(log=log, anime_pd=anime_pd, ratings_pd=ratings_pd)
        processor.save_to_csv(filename=filename)
    else:
        log.info("no Jobs in the queue")
