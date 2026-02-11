import importlib
import os
from unittest.mock import MagicMock

import pytest
from label_studio_ml.response import ModelResponse

# Ensure the Label Studio SDK inside the Deepgram example sees harmless defaults.
os.environ.setdefault('LABEL_STUDIO_URL', 'http://localhost')
os.environ.setdefault('LABEL_STUDIO_API_KEY', 'test-token')

try:
    deepgram_module = importlib.import_module('label_studio_ml.examples.deepgram.model')
except ImportError:
    deepgram_module = importlib.import_module('model')

DeepgramModelCls = deepgram_module.DeepgramModel


@pytest.fixture
def env_settings(monkeypatch):
    """Provide the environment variables required by the Deepgram example."""
    settings = {
        'DEEPGRAM_API_KEY': 'dg-key',
        'AWS_DEFAULT_REGION': 'us-east-1',
        'S3_BUCKET': 'test-bucket',
        'S3_FOLDER': 'tts',
    }
    for key, value in settings.items():
        monkeypatch.setenv(key, value)
    return settings


@pytest.fixture
def patched_clients(monkeypatch):
    """Patch the Deepgram SDK, boto3 client, and Label Studio SDK with mocks."""
    mock_deepgram_client = MagicMock(name='DeepgramClientInstance')
    mock_deepgram_ctor = MagicMock(return_value=mock_deepgram_client)
    monkeypatch.setattr(deepgram_module, 'DeepgramClient', mock_deepgram_ctor)

    mock_s3_client = MagicMock(name='S3Client')
    monkeypatch.setattr(deepgram_module.boto3, 'client', MagicMock(return_value=mock_s3_client))

    mock_ls_client = MagicMock(name='LabelStudioClient')
    monkeypatch.setattr(
        DeepgramModelCls,
        'get_label_studio_client',
        MagicMock(return_value=mock_ls_client),
        raising=False,
    )

    return {
        'deepgram_client': mock_deepgram_client,
        'deepgram_ctor': mock_deepgram_ctor,
        's3_client': mock_s3_client,
        'ls_client': mock_ls_client,
    }


def test_setup_raises_without_api_key(monkeypatch):
    """
    Scenario: setup is called without DEEPGRAM_API_KEY.
    Steps   : remove the env var and instantiate the model (setup runs in __init__).
    Checks  : verify ValueError is raised mentioning the missing key.
    """
    monkeypatch.delenv('DEEPGRAM_API_KEY', raising=False)

    with pytest.raises(ValueError, match='DEEPGRAM_API_KEY'):
        DeepgramModelCls()


def test_setup_initializes_clients_with_api_key(env_settings, patched_clients):
    """
    Scenario: setup receives valid env vars.
    Steps   : call setup after patching external clients.
    Checks  : ensure Deepgram & S3 clients plus region/bucket/folder are stored.
    """
    model = DeepgramModelCls()
    model.setup()

    assert patched_clients['deepgram_ctor'].called
    assert model.deepgram_client is patched_clients['deepgram_client']
    assert model.s3_client is patched_clients['s3_client']
    assert model.s3_region == env_settings['AWS_DEFAULT_REGION']
    assert model.s3_bucket == env_settings['S3_BUCKET']
    assert model.s3_folder == env_settings['S3_FOLDER']


def test_setup_falls_back_to_access_token(env_settings, patched_clients):
    """
    Scenario: the Deepgram SDK rejects the api_key kwarg.
    Steps   : make the first constructor call raise TypeError, then succeed on retry.
    Checks  : setup retries using access_token and keeps the final client (setup runs in __init__).
    """
    patched_clients['deepgram_ctor'].side_effect = [
        TypeError('unexpected kwarg'),
        patched_clients['deepgram_client'],
    ]
    model = DeepgramModelCls()

    assert patched_clients['deepgram_ctor'].call_count == 2
    first_call_kwargs = patched_clients['deepgram_ctor'].call_args_list[0].kwargs
    second_call_kwargs = patched_clients['deepgram_ctor'].call_args_list[1].kwargs
    assert 'api_key' in first_call_kwargs
    assert 'access_token' in second_call_kwargs
    assert model.deepgram_client is patched_clients['deepgram_client']


def test_predict_no_context_returns_empty_modelresponse(env_settings, patched_clients):
    """
    Scenario: predict is invoked before the user submits any text.
    Steps   : set up env vars and mocks, then call predict with empty context/result payloads.
    Checks  : confirm an empty ModelResponse is returned immediately without calling external services.
    """
    model = DeepgramModelCls()
    tasks = [{'id': 1}]

    response = model.predict(tasks=tasks, context=None)

    assert isinstance(response, ModelResponse)
    assert response.predictions == []
    # Verify no external calls were made
    patched_clients['deepgram_client'].speak.v1.audio.generate.assert_not_called()
    patched_clients['s3_client'].upload_file.assert_not_called()


def test_predict_generates_audio_uploads_to_s3_and_updates_task(env_settings, patched_clients):
    """
    Scenario: predict handles a happy path request.
    Steps   : mock Deepgram audio chunks, S3 upload, and Label Studio update.
    Checks  : verify Deepgram is called, S3 upload args are correct, ls.tasks.update
              receives the S3 URL, and the temporary file is deleted.
    """
    patched_clients['deepgram_client'].speak.v1.audio.generate.return_value = [b'chunk-a', b'chunk-b']
    model = DeepgramModelCls()
    model.setup()

    tasks = [{'id': 123}]
    context = {
        'user_id': 'user-7',
        'result': [{'value': {'text': ['Hello Deepgram']}}],
    }

    model.predict(tasks=tasks, context=context)

    patched_clients['deepgram_client'].speak.v1.audio.generate.assert_called_once_with(text='Hello Deepgram')
    assert patched_clients['s3_client'].upload_file.call_count == 1

    upload_args = patched_clients['s3_client'].upload_file.call_args.kwargs
    local_path = patched_clients['s3_client'].upload_file.call_args.args[0]
    assert upload_args['ExtraArgs']['ContentType'] == 'audio/mpeg'
    assert upload_args['ExtraArgs']['ACL'] == 'public-read'
    assert upload_args['ExtraArgs']['CacheControl'].startswith('public')

    expected_key = f"{env_settings['S3_FOLDER']}/123_user-7.mp3"
    assert patched_clients['s3_client'].upload_file.call_args.args[2] == expected_key

    expected_url = f"https://{env_settings['S3_BUCKET']}.s3.{env_settings['AWS_DEFAULT_REGION']}.amazonaws.com/{expected_key}"
    patched_clients['ls_client'].tasks.update.assert_called_once_with(
        id=123,
        data={'text': 'Hello Deepgram', 'audio': expected_url},
    )

    assert not os.path.exists(local_path)


def test_predict_s3_failure_raises_and_cleans_up_temp_file(env_settings, patched_clients):
    """
    Scenario: the S3 upload raises an exception.
    Steps   : let Deepgram produce chunks, force upload_file to fail.
    Checks  : ensure the exception bubbles up, temp file is removed, and Label Studio
              is never updated.
    """
    patched_clients['deepgram_client'].speak.v1.audio.generate.return_value = [b'chunk']
    patched_clients['s3_client'].upload_file.side_effect = RuntimeError('s3 boom')
    model = DeepgramModelCls()
    model.setup()

    tasks = [{'id': 999}]
    context = {
        'user_id': 'user-1',
        'result': [{'value': {'text': ['Explode']}}],
    }

    with pytest.raises(RuntimeError, match='s3 boom'):
        model.predict(tasks=tasks, context=context)

    local_path = patched_clients['s3_client'].upload_file.call_args.args[0]
    assert not os.path.exists(local_path)
    patched_clients['ls_client'].tasks.update.assert_not_called()


def test_setup_in_test_mode_uses_stub_clients(monkeypatch):
    """
    Scenario: TEST_ENV is enabled to activate internal stubs.
    Steps   : set TEST_ENV, instantiate the model, and inspect configured clients.
    Checks  : ensure real Deepgram constructor is not called and stub clients exist with defaults.
    """
    monkeypatch.setenv('TEST_ENV', '1')
    monkeypatch.setenv('S3_BUCKET', 'test-bucket')
    monkeypatch.setenv('S3_FOLDER', 'tts')
    ctor = MagicMock()
    monkeypatch.setattr(deepgram_module, 'DeepgramClient', ctor)

    model = DeepgramModelCls()

    assert model.test_mode is True
    ctor.assert_not_called()
    assert callable(model.deepgram_client.speak.v1.audio.generate)
    assert model.s3_bucket == 'test-bucket'
    assert model.s3_folder == 'tts'


def test_predict_test_mode_skips_label_studio_update(monkeypatch):
    """
    Scenario: predict runs in test mode so external Label Studio updates should be skipped.
    Steps   : enable TEST_ENV, patch ls.tasks.update, run predict with valid context.
    Checks  : confirm stub S3 upload runs without raising and Label Studio update is not invoked.
    """
    monkeypatch.setenv('TEST_ENV', '1')
    model = DeepgramModelCls()
    mocked_ls_client = MagicMock()
    mocked_get_client = MagicMock(return_value=mocked_ls_client)
    monkeypatch.setattr(model, 'get_label_studio_client', mocked_get_client, raising=False)

    tasks = [{'id': 321}]
    context = {
        'user_id': 'tester',
        'result': [{'value': {'text': ['Hello from test mode']}}],
    }

    model.predict(tasks=tasks, context=context)

    mocked_get_client.assert_not_called()
    mocked_ls_client.tasks.update.assert_not_called()

