import shutil
import logging

from pathlib import Path

import numpy as np
import pandas as pd

from alive_progress import alive_bar


class DatasetLoader:
    def __init__(self, log: logging.Logger, archive_path: Path):
        self.log = log
        self.archive_path = archive_path
        self.data_raw = Path("src") / "anime_recommender" / "data" / "raw"
        self._extracted = False

    def _unpack_archive(self) -> None:
        """Unpacks the ZipFile and keeps only neccessary files."""
        if not self._extracted:
            self.log.info("Unpacking Archive @on_job_start")
            shutil.unpack_archive(filename=self.archive_path, extract_dir=self.data_raw, format="zip")
            for file_ in self.data_raw.iterdir():
                if file_.name == "html folder":
                    self.log.debug(f"Remove tree {file_.name}")
                    shutil.rmtree(file_, ignore_errors=True)
                elif file_.name not in ("anime.csv", "rating_complete.csv"):
                    self.log.debug(f"Unlink file {file_.name}")
                    Path.unlink(file_)
            self._extracted = True
        self.log.info("Unpacking Archive @on_job_end")

    def load_pandas_data_frames(self) -> list[pd.DataFrame]:
        """Returns the neccessary DataFrames"""

        if not self._extracted:
            self._unpack_archive()
        return [pd.read_csv(self.data_raw.joinpath(csv.name)) for csv in self.data_raw.iterdir()]


class DatasetProcessor:
    def __init__(self, log: logging.Logger, anime_pd: pd.DataFrame, ratings_pd: pd.DataFrame):
        self.log = log
        self.anime_pd = anime_pd
        self.ratings_pd = ratings_pd

    def _merge(self) -> pd.DataFrame:
        """
        SELECT  rate.rating,
                rate.user_id,
                rate.anime_id,
                anm.Name,
                anm.Genres,
                anm.Score

        FROM ratings_pd rate

        INNER JOIN anime_pd anm
            ON rate.anime_id = anm.MAL_ID
        """
        self.log.info("Join Tables @on_job_start")
        merge = pd.merge(
            left=self.ratings_pd[["rating", "user_id", "anime_id"]],
            right=self.anime_pd[["MAL_ID", "Name", "Genres", "Score"]],
            left_on="anime_id",
            right_on="MAL_ID",
        )
        # Processing the merged table
        merge.rename(
            columns={
                "Name": "anime",
                "Genres": "genres",
                "Score": "score",
            },
            inplace=True,
        )
        merge.drop("MAL_ID", axis=1, inplace=True)
        merge.rating = merge.rating.astype(np.float32)
        records = len(merge)
        self.log.debug(f"Total records: {records:_}")
        self.log.info("Join Tables @on_job_end")
        return merge

    def save_to_csv(self, filename: str | Path) -> None:
        """
        This method saves from the merged table, the columns:
            - anime_id
            - anime
            - genres,
        which will be used in the prediction stage.
        Also saves the anime-dimension in .txt file which will
        be used in the training stage using Sagemaker's Estimator.
        """
        filename = Path(filename)
        assert filename.suffix == ".csv"
        train_and_inference_dir = Path("src") / "anime_recommender" / "data" / "train-and-inference"
        if not train_and_inference_dir.exists():
            train_and_inference_dir.mkdir(parents=True, exist_ok=True)
        fullpath = train_and_inference_dir.joinpath(filename)
        merge_pd = self._merge()
        self.log.info("Save Join Table to CSV @on_job_start")

        # Write only: animeID, anime-name, genres
        with alive_bar(spinner="classic") as bar:
            merge_pd[["anime_id", "anime", "genres"]].to_csv(fullpath)
            bar()

        unique_users = merge_pd.user_id.unique()
        unique_anime = merge_pd.anime_id.unique()
        anime_dimension = len(unique_users) + len(unique_anime)

        with fullpath.with_name("dimension.txt").open("w") as f:
            f.write(str(anime_dimension))
        self.log.info("Save Join Table to CSV @on_job_end")
