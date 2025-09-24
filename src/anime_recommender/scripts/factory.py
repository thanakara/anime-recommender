import logging

from anime_recommender.scripts import setup
from anime_recommender.constants import core


def context_factory(log: logging.Logger, ratio: float, seed: int) -> setup.DatasetContext:
    archive_path = core.Filepath.archive_path
    dataraw_path = core.Filepath.data_raw
    assert dataraw_path.exists(), f"{str(dataraw_path)} doesn't exist"

    loader = setup.DatasetLoader(log=log, archive_path=archive_path)
    loader._extracted = True
    anime_pd, ratings_pd = loader.load_pandas_data_frames()

    processor = setup.DatasetProcessor(log=log, anime_pd=anime_pd, ratings_pd=ratings_pd)
    data = processor._merge()

    return setup.DatasetContext(log=log, data=data, train_split_ratio=ratio, seed=seed)
