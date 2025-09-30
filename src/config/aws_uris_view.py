import json

from omegaconf import OmegaConf

from anime_recommender.scripts import resolvers
from anime_recommender.constants import core
from anime_recommender.scripts.callbacks import EventsCallback

callback = EventsCallback()
log = callback.logger

file_ = core.Filepath.aws_uris_config_path
config = OmegaConf.load(file_=file_)
OmegaConf.register_new_resolver(name="region_resolver", resolver=resolvers.region_name_resolver)
OmegaConf.register_new_resolver(name="role_resolver", resolver=resolvers.execution_role_resolver)
OmegaConf.register_new_resolver(name="latest_job_name", resolver=resolvers.get_latest_job_name)
container = OmegaConf.to_container(config, resolve=True)

log.info(f"\n{json.dumps(container, indent=2)}")
