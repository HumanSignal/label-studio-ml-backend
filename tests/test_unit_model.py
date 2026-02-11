import copy
import pytest
from unittest.mock import patch, mock_open, MagicMock
from label_studio_ml.model import LabelStudioMLBase


@pytest.fixture
def model():
    return LabelStudioMLBase(label_config="""<View>
        <Text name="Text" value="$text"/>
        <Labels name="Labels" toName="Text">
            <Label value="label1"/>
            <Label value="label2"/>
        </Labels>
    </View>
    """)


@pytest.mark.parametrize("url, expected_result", [
    ("s3://bucket/test/1.jpg", "data"),
    ("gs://bucket/test/1.jpg", "data"),
    ("azure-blob://bucket/test/1.jpg", "data"),
    ("/data/local-files?d=test", "data"),
    ("/upload/123.jpg", "data"),
    ("c:\\test.jpg", "data"),
    ("/home/user/123.jpg", "data"),
    ("this is text", "this is text"),
])
@patch("builtins.open", new_callable=mock_open, read_data="body")
@patch("label_studio_ml.model.LabelStudioMLBase.get_local_path", return_value="path")
@patch("os.path.exists", return_value=True)  # Mock os.path.exists to return True
def test_preload_task_data(mock_exists, mock_get_local_path, mock_file, model, url, expected_result):
    task = {"id": 1, "data": {"url": url}}
    result = model.preload_task_data(task, value=copy.deepcopy(task['data']))
    assert result == {"url": "body"}
    mock_get_local_path.assert_called_once_with(url=task["data"]["url"], task_id=task["id"])
    mock_file.assert_called_once_with("path", "r")
    print(result)


@patch("builtins.open", new_callable=mock_open, read_data="body")
@patch("label_studio_ml.model.LabelStudioMLBase.get_local_path", return_value="path")
def test_preload_task_data_complex_structure(mock_get_local_path, mock_file, model):
    def generate_task(url):
        return {
            "id": 1,
            "data": {
                "root": {
                    "url": url,
                    "urls": [url, url, url],
                    "text": "this is text",
                },
                "int": 123,
                "float": 42.42,
                "bool": False,
                "null": None,
            }
        }

    url = "s3://bucket/test/1.jpg"
    task = generate_task(url)
    expected = generate_task("body")
    result = model.preload_task_data(task, value=copy.deepcopy(task['data']))
    assert result == expected['data']
    mock_get_local_path.assert_called_with(url=url, task_id=task["id"])
    mock_file.assert_called_with("path", "r")
    print(result)


@patch("label_studio_ml.model.LabelStudio")
def test_label_studio_client_initialized_and_cached(mock_label_studio, model):
    client_instance = MagicMock()
    mock_label_studio.return_value = client_instance

    with patch.dict("os.environ", {"LABEL_STUDIO_URL": "http://localhost:8080"}, clear=False):
        # First access should initialize SDK client
        first = model._get_label_studio_client()
        # Second access should reuse cached client
        second = model._get_label_studio_client()

    assert first is client_instance
    assert second is client_instance
    mock_label_studio.assert_called_once_with(base_url="http://localhost:8080")


@patch("label_studio_ml.model.LabelStudio", side_effect=RuntimeError("boom"))
def test_label_studio_client_init_failure_falls_back_to_env_token(mock_label_studio, model):
    with patch.dict(
        "os.environ",
        {
            "LABEL_STUDIO_URL": "http://localhost:8080",
            "LABEL_STUDIO_API_KEY": "legacy-token",
        },
        clear=False,
    ):
        # Initialization failure should not crash token retrieval
        token = model.get_label_studio_access_token()
        token_second = model.get_label_studio_access_token()

    assert token == "legacy-token"
    assert token_second == "legacy-token"
    # Failed init should be remembered, so constructor is only attempted once.
    mock_label_studio.assert_called_once_with(base_url="http://localhost:8080")