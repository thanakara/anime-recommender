import boto3
import sagemaker


def region_name_resolver() -> str:
    sess = boto3.session.Session()
    return sess.region_name


def execution_role_resolver() -> str:
    return sagemaker.get_execution_role()
