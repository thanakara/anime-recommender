import boto3
import sagemaker


def region_name_resolver() -> str:
    sess = boto3.session.Session()
    return sess.region_name


def execution_role_resolver() -> str:
    return sagemaker.get_execution_role()


def get_latest_job_name() -> str:
    sm = boto3.client("sagemaker")
    response = sm.list_training_jobs(MaxResults=1)
    return response["TrainingJobSummaries"][0]["TrainingJobName"]
