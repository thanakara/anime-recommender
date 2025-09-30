import boto3
import sagemaker

from omegaconf import OmegaConf, DictConfig

from anime_recommender.constants import Filepath


class ARSTrainer:
    def __init__(self, config: DictConfig):
        self.config = config
        self._sess = sagemaker.Session()
        self._image_uri = sagemaker.image_uris.retrieve(self.config.framework, self.config.region)
        self.estimator = None
        self.predictor = None

    def _build(self) -> sagemaker.estimator.Estimator:
        """
        Configure the training job.
        SDK 2.x version doesn't require train prefix for instance count and type.
        """
        if self.estimator is None:
            estimator = sagemaker.estimator.Estimator(
                image_uri=self._image_uri,
                role=self.config.role,
                instance_count=self.config.instace_count,
                instance_type=self.config.instance_type,
                output_path=self.config.s3_model_output,
                sagemaker_session=self._sess,
                base_job_name=self.self.config.job_name,
                use_spot_instances=self.config.use_spot_instances,
                max_run=self.config.max_run,
                max_wait=self.config.max_wait,
                checkpoint_s3_uri=self.config.checkpoint_uri,
            )
            self.estimator = estimator

        return self.estimator

    def _set_hyperparameters(self) -> sagemaker.estimator.Estimator:
        """Set the hyperparameters after the tuning."""

        if self.estimator is None:
            _ = self._build()
        params_conf = OmegaConf.load(Filepath.hyperparameters_path)
        params_obj = OmegaConf.to_object(params_conf)
        self.estimator.set_hyperparameters(**params_obj)

        return self.estimator

    def trainjob(self) -> sagemaker.estimator.Estimator:
        estimator = self._set_hyperparameters()
        estimator.fit(
            {
                "train": self.config.s3_training_file,
                "test": self.config.s3_test_file,
            }
        )
        return self.estimator


def create_endpoint_from_training_job(config: DictConfig) -> str:
    """Create the Endpoint from a specified existing and completed training job."""

    sm = boto3.client("sagemaker")

    # Get job info
    training_info = sm.describe_training_job(TrainingJobName=config.job_name)
    model_artifact = training_info["ModelArtifacts"]["S3ModelArtifacts"]
    container_image = training_info["AlgorithmSpecification"]["TrainingImage"]

    # Create the model
    sm.create_model(
        ModelName=config.model_name,
        PrimaryContainer={"Image": container_image, "ModelDataUrl": model_artifact},
        ExecutionRoleArn=config.role,
    )

    # Create Endpoint configuration
    sm.create_endpoint_config(
        EndpointConfigName=config.endpoint_cfg_name,
        ProductionVariants=[
            {
                "VariantName": "AllTraffic",
                "ModelName": config.model_name,
                "InitialInstanceCount": config.instance_count,
                "InstanceType": config.inference_instance_type,
            }
        ],
    )

    # Create the Endpoint
    sm.create_endpoint(EndpointName=config.endpoint_name, EndpointConfigName=config.endpoint_cfg_name)

    return config.endpoint_name


def delete_endpoint(config: DictConfig) -> None:
    """A full cleanup (Endpoint - Config - Model)"""

    sm = boto3.client("sagemaker")

    # Delete the Endpoint
    sm.delete_endpoint(EndpointName=config.endpoint_name)

    # Remove Endpoint configuration
    sm.delete_endpoint_config(EndpointConfigName=config.endpoint_cfg_name)

    # Delete the model created
    sm.delete_model(ModelName=config.model_name)
