import logging

from anime_recommender.constants import Filepath
from anime_recommender.scripts.setup import DatasetLoader, DatasetContext, DatasetProcessor


def context_factory(log: logging.Logger, ratio: float, seed: int) -> DatasetContext:
    archive_path = Filepath.archive_path
    dataraw_path = Filepath.data_raw
    assert dataraw_path.exists(), f"{str(dataraw_path)} doesn't exist"

    loader = DatasetLoader(log=log, archive_path=archive_path)
    loader._extracted = True
    anime_pd, ratings_pd = loader.load_pandas_data_frames()

    processor = DatasetProcessor(log=log, anime_pd=anime_pd, ratings_pd=ratings_pd)
    data = processor._merge()

    return DatasetContext(log=log, data=data, train_split_ratio=ratio, seed=seed)
