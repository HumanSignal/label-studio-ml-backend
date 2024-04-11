import label_studio_sdk
from typing import List, Dict


def download_ls_dataset(api_url: str, api_token: str, project_id: int) -> List[Dict]:
    """
    Download all labeled tasks from project using the Label Studio SDK.
    Read more about SDK here https://labelstud.io/sdk/
    :param project: project ID
    :return:
    """
    ls = label_studio_sdk.Client(api_url, api_token)
    project = ls.get_project(id=project_id)
    tasks = project.get_labeled_tasks()
    return tasks
