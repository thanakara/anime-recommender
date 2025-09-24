import json

from pathlib import Path

from omegaconf import OmegaConf

from anime_recommender.scripts.callbacks import EventsCallback

callback = EventsCallback()
log = callback.logger

CONFIG = Path("src") / "config" / "aws-uris.yaml"
config = OmegaConf.load(CONFIG)
container = OmegaConf.to_container(config, resolve=True)

log.info(f"\n{json.dumps(container, indent=3)}")
