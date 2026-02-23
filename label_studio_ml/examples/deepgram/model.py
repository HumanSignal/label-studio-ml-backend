import os
import sys
import pathlib
from types import SimpleNamespace
from typing import List, Dict, Optional
from werkzeug.utils import secure_filename
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse
from deepgram import DeepgramClient
import boto3


class DeepgramModel(LabelStudioMLBase):
    """Custom ML Backend model for Deepgram
    """

    def setup(self):
        """Initialize the Deepgram client with API key from environment"""
        self.test_mode = self._is_test_mode_enabled()
        if self.test_mode:
            self._setup_test_clients()
            return

        api_key = os.getenv('DEEPGRAM_API_KEY')
        if not api_key:
            raise ValueError("DEEPGRAM_API_KEY environment variable is not set")
        print(f"Initializing Deepgram client with API key: {api_key[:10]}...")  # Debug: show first 10 chars
        # Deepgram SDK uses 'api_key' parameter in newer versions, 'access_token' in older
        # Try both to ensure compatibility
        try:
            self.deepgram_client = DeepgramClient(api_key=api_key)
        except (TypeError, ValueError):
            # Fallback to access_token for older SDK versions
            self.deepgram_client = DeepgramClient(access_token=api_key)
        
        # Initialize S3 client for uploading audio files
        self.s3_client = boto3.client('s3')
        self.s3_region = os.getenv('AWS_DEFAULT_REGION')
        self.s3_bucket = os.getenv('S3_BUCKET')
        self.s3_folder = os.getenv('S3_FOLDER')

    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> ModelResponse:
        """ Returns the predicted mask for a smart keypoint that has been placed."""

        if not context or not context.get('result'):
            # if there is no context, no interaction has happened yet
            return ModelResponse(predictions=[])

        task_id = tasks[0]['id']
        text = context['result'][0]['value']['text'][0]
        response = self.deepgram_client.speak.v1.audio.generate(
            text=text
        )
        
        # Generate unique filename for the audio file - task_id and user_id are unique identifiers for the task and user
        safe_task_id = secure_filename(str(task_id))
        safe_user_id = secure_filename(str(context['user_id']))
        audio_filename = f"{safe_task_id}_{safe_user_id}.mp3"
        local_audio_path = os.path.normpath(os.path.join("/tmp", audio_filename))
        # Ensure the final path is within /tmp
        if not local_audio_path.startswith(os.path.abspath("/tmp") + os.sep):
            raise ValueError("Invalid path: attempted directory traversal in filename")
        # Write audio chunks to local file
        with open(local_audio_path, "wb") as audio_file:
            for chunk in response:
                audio_file.write(chunk)
        
        # Upload to S3 with public-read ACL for wide open CORS access
        s3_key = f"{self.s3_folder}/{audio_filename}"
        try:
            self.s3_client.upload_file(
                local_audio_path,
                self.s3_bucket,
                s3_key,
                ExtraArgs={
                    'ContentType': 'audio/mpeg',
                    'ACL': 'public-read',  # Make object publicly readable for wide open access
                    'CacheControl': 'public, max-age=31536000',  # Cache for 1 year
                }
            )
            # Generate S3 URL
            s3_url = f"https://{self.s3_bucket}.s3.{self.s3_region}.amazonaws.com/{self.s3_folder}/{audio_filename}"
            print(f"Uploaded audio to S3: {s3_url}")
            
            # Update task with S3 URL
            if self.test_mode:
                print(f"[TEST MODE] Would update task {task_id} with audio {s3_url}")
            else:
                ls_client = self.get_label_studio_client()
                if not ls_client:
                    raise RuntimeError("Unable to initialize Label Studio SDK client")
                ls_client.tasks.update(id=task_id, data={"text": text, "audio": s3_url})
        except Exception as e:
            print(f"Error uploading to S3: {e}")
            raise
        finally:
            # Clean up local file
            if os.path.exists(local_audio_path):
                os.remove(local_audio_path)

    def _is_test_mode_enabled(self) -> bool:
        """Check environment variables to decide if the model should use local stubs."""
        truthy = {'1', 'true', 'TRUE', 'True', 'yes', 'on'}
        explicit_flag = os.getenv('DEEPGRAM_TEST_MODE')
        test_env_flag = os.getenv('TEST_ENV')
        return (explicit_flag in truthy) or (test_env_flag in truthy)

    def _setup_test_clients(self):
        """Configure lightweight stub clients so docker/CI runs do not need real secrets."""
        print("[TEST MODE] DeepgramModel using stubbed Deepgram/S3 clients.")

        def fake_generate(text: str):
            # Produce deterministic fake audio bytes for predictable tests.
            preview = text[:10] if text else ''
            return [f"fake-audio-{preview}".encode('utf-8')]

        fake_audio = SimpleNamespace(generate=fake_generate)
        fake_speak = SimpleNamespace(v1=SimpleNamespace(audio=fake_audio))
        self.deepgram_client = SimpleNamespace(speak=fake_speak)

        class _StubS3Client:
            """Minimal S3 client replacement for test environments."""
            def upload_file(self, filename, bucket, key, ExtraArgs=None):
                print(f"[TEST MODE] Pretend upload of {filename} to s3://{bucket}/{key}")

        self.s3_client = _StubS3Client()
        # Provide sensible defaults so downstream URL building still works.
        self.s3_region = os.getenv('AWS_DEFAULT_REGION', 'us-east-1')
        self.s3_bucket = os.getenv('S3_BUCKET', 'test-bucket')
        self.s3_folder = os.getenv('S3_FOLDER', 'tts')

