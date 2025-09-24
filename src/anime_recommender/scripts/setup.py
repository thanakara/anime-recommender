import shutil
import logging

from pathlib import Path

import numpy as np
import pandas as pd
import sagemaker.amazon.common as smac

from sklearn import datasets, preprocessing
from alive_progress import alive_bar


class DatasetLoader:
    """---------------------------------------------------+
    | Class used to load and unpack the DataFrames needed |
    +---------------------------------------------------"""

    def __init__(self, log: logging.Logger, archive_path: Path) -> None:
        self.log = log
        self.archive_path = archive_path
        self.data_raw = Path("src") / "anime_recommender" / "data" / "raw"
        self._extracted = False

    def _unpack_archive(self) -> None:
        """Unpacks the ZipFile and keeps only neccessary files."""

        if not self._extracted:
            self.log.info("===== Unpack Archive Job =====")
            shutil.unpack_archive(
                filename=self.archive_path,
                extract_dir=self.data_raw,
                format="zip",
            )
            for file_ in self.data_raw.iterdir():
                if file_.name == "html folder":
                    self.log.debug(f"Removing tree {file_.name}")
                    shutil.rmtree(file_, ignore_errors=True)
                elif file_.name not in ("anime.csv", "rating_complete.csv"):
                    self.log.debug(f"Unlinking file {file_.name}")
                    Path.unlink(file_)
            self._extracted = True

    def load_pandas_data_frames(self) -> list[pd.DataFrame]:
        """Returns the neccessary DataFrames"""

        if not self._extracted:
            self._unpack_archive()

        # NOTE: on the cloud the ordering shifts --can't use .iterdir() method
        # return [pd.read_csv(self.data_raw.joinpath(csv.name)) for csv in self.data_raw.iterdir()]
        return [pd.read_csv(self.data_raw.joinpath(csv)) for csv in ("anime.csv", "rating_complete.csv")]


class DatasetProcessor:
    """-------------------------------------------------------------------------------+
    | Class used to join DataFrames and write the CSV file used later for predictions |
    +-------------------------------------------------------------------------------"""

    def __init__(
        self,
        log: logging.Logger,
        anime_pd: pd.DataFrame,
        ratings_pd: pd.DataFrame,
    ) -> None:
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

        type_filter = self.anime_pd.Type == "TV"
        self.anime_pd = self.anime_pd[type_filter]
        self.log.info("===== Join Tables Job =====")
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
        train_and_inference_dir = Path("src") / "anime_recommender" / "data" / "train+inference"
        if not train_and_inference_dir.exists():
            train_and_inference_dir.mkdir(parents=True, exist_ok=True)
        fullpath = train_and_inference_dir.joinpath(filename)
        merge_pd = self._merge()
        self.log.info("===== Save Raw CSV Job =====")

        # Write only: animeID, anime-name, genres
        with alive_bar(spinner="classic") as bar:
            merge_pd[["anime_id", "anime", "genres"]].to_csv(fullpath, index=False)
            bar()

        unique_users = merge_pd.user_id.unique()
        unique_anime = merge_pd.anime_id.unique()
        anime_dimension = len(unique_users) + len(unique_anime)

        with fullpath.with_name("dimension.txt").open("w") as f:
            f.write(str(anime_dimension))


class DatasetContext:
    """------------------------------------------------------------------------------+
    | Class used to create all the necessary files needed for inference and training |
    +------------------------------------------------------------------------------"""

    _DATAPATH = Path("src") / "anime_recommender" / "data" / "train+inference"
    if not _DATAPATH.exists():
        _DATAPATH.mkdir(parents=True, exist_ok=True)

    _cols = ["user_id", "anime_id"]
    _encodings = None
    _encoder = None

    def __init__(self, log: logging.Logger, data: pd.DataFrame, train_split_ratio: float = 0.7, seed: int = 42) -> None:
        self.log = log
        self.data = data
        self.train_split_ratio = train_split_ratio
        self.seed = seed

    def _permute(self) -> pd.DataFrame:
        """Returns the dataset permuted on some seed"""

        np.random.seed(self.seed)
        perm = np.random.permutation(len(self.data))
        df_perm = self.data.iloc[perm]
        return df_perm

    def split_and_write_train_test(self, write: bool = True) -> int:
        """
        Splits the permuted dataset on some ratio
        Writes to CSV optionally
        Updates the Training Size variable
        """
        train_filename = self._DATAPATH.joinpath("user-anime-train.csv")
        test_filename = self._DATAPATH.joinpath("user-anime-test.csv")
        df_perm: pd.DataFrame = self._permute()
        train_size = int(len(df_perm) * self.train_split_ratio)

        if write:
            self.log.debug(f"Training Size: {train_size:_}")
            self.log.info("===== Write train CSV Job =====")

            with alive_bar(spinner="classic") as bar:
                df_perm[self._cols].iloc[:train_size].to_csv(train_filename, index=False)
                bar()
            self.log.info("===== Write test CSV Job =====")

            with alive_bar(spinner="classic") as bar:
                bar()
            df_perm[self._cols].iloc[train_size:].to_csv(test_filename, index=False)

        self._train_size = train_size

    def _one_hot_encode(self):  # TODO: Returning hints
        """
        Returns:
            + one-hot encodings of the permuted dataset
            + target as Float32
            + the encoder itself
        """

        df_perm = self._permute()
        if self._encoder is None:
            self.log.info("Encoding the dataset")
            self._encoder = preprocessing.OneHotEncoder(dtype=np.float32)
            self._encodings = self._encoder.fit_transform(df_perm[self._cols])
        return self._encodings, df_perm.rating.values.astype(np.float32), self._encoder

    @staticmethod
    def _write_sparse_recordio_file(filename: Path, X, y=None):
        with filename.open("wb") as f:
            smac.write_spmatrix_to_sparse_tensor(f, X, y)

    def make_recordio_files(self) -> None:
        X, y, _ = self._one_hot_encode()
        self.split_and_write_train_test(write=False)
        train_size = self._train_size
        train_filename = self._DATAPATH.joinpath("user-anime-train.recordio")
        test_filename = self._DATAPATH.joinpath("user-anime-test.recordio")
        self.log.info("====== Write train RecordIO Job =====")
        self.log.warning("This process may take several minutes")
        with alive_bar() as bar:
            self._write_sparse_recordio_file(filename=train_filename, X=X[:train_size], y=y[:train_size])
            bar()
        self.log.info("====== Write test RecordIO Job ======")
        with alive_bar() as bar:
            self._write_sparse_recordio_file(filename=test_filename, X=X[train_size:], y=y[train_size:])
            bar()

    def make_svmlight_files(self) -> None:
        X, y, _ = self._one_hot_encode()
        self.split_and_write_train_test(write=False)
        train_size = self._train_size
        train_filename = self._DATAPATH.joinpath("user-anime-train.svmlight").as_posix()
        test_filename = self._DATAPATH.joinpath("user-anime-test.svmlight").as_posix()

        self.log.info("====== Write train libSVM Job ======")
        with alive_bar() as bar:
            datasets.dump_svmlight_file(X=X[:train_size], y=y[:train_size], f=train_filename)
            bar()
        self.log.info("====== Write test libSVM Job ======")
        with alive_bar() as bar:
            datasets.dump_svmlight_file(X=X[train_size:], y=y[train_size:], f=test_filename)
            bar()

    def _create_categorical_mappings(self) -> tuple[pd.DataFrame]:
        unique_users = self.data.user_id.unique()
        unique_anime = self.data.anime_id.unique()
        cat_userID_to_userIndex = pd.DataFrame(
            data={"user_id": unique_users, "anime_id": np.ones(shape=[len(unique_users)], dtype=np.int32)}
        )
        cat_animeID_to_animeIndex = pd.DataFrame(
            data={"user_id": np.ones(shape=[len(unique_anime)], dtype=np.int32), "anime_id": unique_anime}
        )
        return cat_userID_to_userIndex, cat_animeID_to_animeIndex

    def create_lookup_files(self) -> None:
        *_, encoder = self._one_hot_encode()
        cat_userID_to_userIndex, cat_animeID_to_animeIndex = self._create_categorical_mappings()
        unique_users = self.data.user_id.unique()
        unique_anime = self.data.anime_id.unique()
        X_user = encoder.transform(cat_userID_to_userIndex[self._cols])
        X_anime = encoder.transform(cat_animeID_to_animeIndex[self._cols])

        self.log.info("===== Crete Lookup files Job ======")
        with alive_bar(spinner="classic") as bar:
            datasets.dump_svmlight_file(
                X=X_user, y=unique_users, f=self._DATAPATH.joinpath("ohe-users.svmlight").as_posix()
            )
            datasets.dump_svmlight_file(
                X=X_anime, y=unique_anime, f=self._DATAPATH.joinpath("ohe-anime.svmlight").as_posix()
            )
            bar()
