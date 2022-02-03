import os
import logging
import tarfile
import datetime
import google.auth

from PIL import Image

from google.api_core.exceptions import NotFound
from google.cloud import artifactregistry_v1beta2
from google.cloud.artifactregistry_v1beta2 import CreateRepositoryRequest, Repository
from google.cloud.devtools import cloudbuild_v1
from google.cloud import storage as google_storage
from google.cloud.devtools.cloudbuild_v1 import Source, StorageSource

from label_studio_tools.core.utils.params import get_env
from label_studio_tools.core.utils.io import get_local_path

DATA_UNDEFINED_NAME = '$undefined$'

logger = logging.getLogger(__name__)


def get_single_tag_keys(parsed_label_config, control_type, object_type):
    """
    Gets parsed label config, and returns data keys related to the single control tag and the single object tag schema
    (e.g. one "Choices" with one "Text")
    :param parsed_label_config: parsed label config returned by "label_studio.misc.parse_config" function
    :param control_type: control tag str as it written in label config (e.g. 'Choices')
    :param object_type: object tag str as it written in label config (e.g. 'Text')
    :return: 3 string keys and 1 array of string labels: (from_name, to_name, value, labels)
    """
    assert len(parsed_label_config) == 1
    from_name, info = list(parsed_label_config.items())[0]
    assert info['type'] == control_type, 'Label config has control tag "<' + info['type'] + '>" but "<' + control_type + '>" is expected for this model.'  # noqa

    assert len(info['to_name']) == 1
    assert len(info['inputs']) == 1
    assert info['inputs'][0]['type'] == object_type
    to_name = info['to_name'][0]
    value = info['inputs'][0]['value']
    return from_name, to_name, value, info['labels']


def is_skipped(completion):
    if len(completion['annotations']) != 1:
        return False
    completion = completion['annotations'][0]
    return completion.get('skipped', False) or completion.get('was_cancelled', False)


def get_choice(completion):
    return completion['annotations'][0]['result'][0]['value']['choices'][0]


def get_image_local_path(url, image_cache_dir=None, project_dir=None, image_dir=None):
    return get_local_path(url, image_cache_dir, project_dir, get_env('HOSTNAME'), image_dir)


def get_image_size(filepath):
    return Image.open(filepath).size


def deploy_to_gcp(args):
    # Setup env before hand: https://cloud.google.com/run/docs/setup
    # Prepare dirs with code and docker file
    # Set configuration params
    region = args.gcp_region or os.environ.get("GCP_REGION", "us-central1")
    project_id = args.gcp_project or os.environ.get("GCP_PROJECT")
    service_name = args.project_name
    output_dir = os.path.join(args.root_dir, args.project_name)
    time_stamp = str(datetime.now().timestamp())
    # create tgz file to upload
    output_filename = os.path.join(output_dir, f"{time_stamp}.tgz")
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(output_dir, arcname=".")
    # get current credentials and project
    credentials, project = google.auth.load_credentials_from_file(r"C:\projects\Heartex\TestData\gcs\i-portfolio-339416-807c5a11ea6f.json")
    artifact_registry_name = 'cloud-run-source-deploy'
    # Upload artifacts to GCP
    # Get registry
    registry_client = artifactregistry_v1beta2.ArtifactRegistryClient(credentials=credentials)
    try:
        repo_name = f"projects/{project_id}/locations/{region}/repositories/{artifact_registry_name}"
        repo = registry_client.get_repository(name=repo_name)
    except NotFound:
        if not repo:
            create_repository_request = CreateRepositoryRequest()
            create_repository_request.repository = Repository()
            create_repository_request.repository.name = repo_name
            create_repository_request.repository.description = 'Cloud Run Source Deployments'
            create_repository_request.repository.format_ = Repository.Format.DOCKER
            repo = registry_client.create_repository(create_repository_request)
    except Exception as e:
        logger.error("Error while creating Artifact Repository.", exc_info=True)
        logger.error(e)

    # Get storage link
    storage_client = google_storage.Client(project=project_id, credentials=credentials)
    bucket_name = f"{project_id}_cloudbuild"
    bucket = storage_client.lookup_bucket(bucket_name)
    if not bucket:
        bucket = storage_client.create_bucket(bucket_name, project=project_id)

    # Upload files
    with open(output_filename, mode='rb') as file:
        blob = bucket.blob(f"{time_stamp}.tgz")
        blob.upload_from_file(file)

    # Post build
    build_client = cloudbuild_v1.CloudBuildClient(credentials=credentials)
    build = cloudbuild_v1.Build()
    build.images = [f"us-central1-docker.pkg.dev/{project_id}/{artifact_registry_name}/{service_name}"]
    build.source = Source()
    build.source.storage_source = StorageSource()
    build.source.storage_source.bucket = bucket_name
    build.source.storage_source.generation = blob.generation
    build.source.storage_source.object_ = f"{time_stamp}.tgz"
    build.steps = [{"args": ["build", "--network", "cloudbuild", "--no-cache", "-t",
                            f"us-central1-docker.pkg.dev/{project_id}/{artifact_registry_name}/{service_name}",
                            "."], "name": "gcr.io/cloud-builders/docker"}]

    build_operation = build_client.create_build(project_id=project_id, build=build)
    build_result = build_operation.result()
    artifact = build_result.artifacts.images[0]
    return artifact
