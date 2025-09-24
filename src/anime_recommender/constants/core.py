from pathlib import Path
from dataclasses import dataclass


@dataclass(frozen=True)
class Filepath:
    source_dir: Path = Path("src")
    data_dir: Path = source_dir / "anime_recommender" / "data"
    archive_path: Path = data_dir.joinpath("archive.zip")
    data_raw: Path = data_dir.joinpath("raw")
    train_and_inference_dir: Path = data_dir.joinpath("train+inference")
    config_path: Path = source_dir / "config"
    logging_config_path: Path = config_path.joinpath("log-config.yaml")
    aws_uris_config_path: Path = logging_config_path.with_name("aws-uris.yaml")
