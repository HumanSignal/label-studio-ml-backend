import os
import logging
import sys
import time
import json
import redis
import attr
import io
try:
    import torch.multiprocessing as mp
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass
except ImportError:
    import multiprocessing as mp
import importlib
import importlib.util
import inspect

from typing import Dict
from abc import ABC, abstractmethod
from datetime import datetime
from contextlib import contextmanager
from redis import Redis
from rq import Queue, get_current_job
from rq.registry import StartedJobRegistry, FinishedJobRegistry, FailedJobRegistry
from rq.job import Job
from colorama import Fore

from label_studio_tools.core.utils.params import get_bool_env, get_env
from label_studio_tools.core.label_config import parse_config
from label_studio_tools.core.utils.io import get_local_path

logger = logging.getLogger(__name__)

AUTO_UPDATE_DEFAULT = False


@attr.s
class ModelWrapper(object):
    model = attr.ib()
    model_version = attr.ib()
    is_training = attr.attrib(default=False)


class JobManager(object):
    """Job Manager provides a facility to spin up background jobs for LabelStudioMLBase models"""

    def get_result(self, model_version=None):
        """Return job result based on specified model_version (=job_id)"""
        job_result = None
        if model_version:
            logger.debug(f'Get result based on model_version={model_version}')
            try:
                job_result = self.get_result_from_job_id(model_version)
            except Exception as exc:
                logger.error(exc, exc_info=True)
        else:
            logger.debug(f'Get result from last valid job')
            job_result = self.get_result_from_last_job()
        return job_result or {}

    def job(self, model_class, event: str, data: Dict, job_id: str):
        """
        Job function to be run in background. It actually does the following:
        1. Creates model_class instance, possibly by using artefacts from previously finished jobs
        2. Calls model_class.process_event() method
        :param model_class: LabelStudioMLBase instance
        :param event: event name (e.g. Label Studio webhook action name)
        :param data: job data (e.g. Label Studio webhook payload)
        :param job_id: user-defined job identifier
        :return: json-serializable job result
        """
        # with self.start_run(event, data, job_id):
        model_version = data.get('model_version') or data.get('project', {}).get('model_version')
        job_result = self.get_result(model_version)
        label_config = data.get('label_config') or data.get('project', {}).get('label_config')
        train_output = job_result
        logger.debug(f'Load model with label_config={label_config} and train_output={train_output}')
        model = model_class(label_config=label_config, train_output=train_output)
        additional_params = self.get_additional_params(event, data, job_id)
        result = model.process_event(event, data, job_id, additional_params)
        self.post_process(event, data, job_id, result)

        logger.debug(f'Finished processing event {event}! Return result: {result}')
        return result

    def get_additional_params(self, event, data, job_id):
        """Dict with aux parameters required to run process_event()"""
        return {}

    @contextmanager
    def start_run(self, event, data, job_id):
        """Context manager that can be used to create a separate 'run':
        Useful to perform ML tracking, i.e. by creating a separate workspace for each job call.
        For example, it could be implemented with MLFlow:
        ```python
        with mlflow.start_run() as run:
            yield run
        ```
        """
        raise NotImplementedError

    def get_result_from_job_id(self, job_id):
        """This is the actual function should be used further to ensure 'job_id' is included in results.
        DON'T OVERRIDE THIS FUNCTION! Instead, override _get_result_from_job_id
        """
        result = self._get_result_from_job_id(job_id)
        assert isinstance(result, dict), \
            f"Training job {job_id} was finished unsuccessfully. No result was saved in job folder." \
            f"Please clean up failed job folders to remove this error from log."
        result['job_id'] = job_id
        return result

    def _get_result_from_job_id(self, job_id):
        """Return job result by job id"""
        raise NotImplementedError

    def iter_finished_jobs(self):
        raise NotImplementedError

    def get_result_from_last_job(self, skip_empty_results=True):
        """Return job result by last successfully finished job
        when skip_empty_results is True, result is None are skipped (e.g. if fit() function makes `return` call)
        """
        for job_id in self.iter_finished_jobs():
            logger.debug(f'Try job_id={job_id}')
            try:
                result = self.get_result_from_job_id(job_id)
            except Exception as exc:
                logger.error(f'{job_id} job returns exception: {exc}', exc_info=True)
                continue
            if skip_empty_results and result is None:
                logger.debug(f'Skip empty result from job {job_id}')
                continue
            return result

        # if nothing found - return empty result
        return

    def post_process(self, event, data, job_id, result):
        """Post-processing hook after calling process_event()"""
        raise NotImplementedError

    def run_job(self, model_class, args: tuple):
        """Defines the logic to run job() in background"""
        raise NotImplementedError


class SimpleJobManager(JobManager):
    """Simple Job Manager doesn't require additional dependencies
    and uses a native python multiprocessing for running job in background.
    Job results / artefacts are stored as ordinary files, inside user-defined model directory:
    model_dir:
        |_ job_id
            |_ event.json
            |_ job_result.json
            |_ artefacts.bin
    Note, that this job manager should be used only for development purposes.
    """
    JOB_RESULT = 'job_result.json'  # in this file,

    def __init__(self, model_dir='.', async_job=False):
        self.model_dir = model_dir
        self.async_job = async_job

    @contextmanager
    def start_run(self, event, data, job_id):
        job_dir = self._job_dir(job_id)
        os.makedirs(job_dir, exist_ok=True)
        with open(os.path.join(job_dir, 'event.json'), mode='w') as f:
            event_data = {'event': event, 'job_id': job_id}
            if data:
                event_data['data'] = data
            json.dump(event_data, f, indent=2)
        yield job_dir

    def _job_dir(self, job_id):
        return os.path.join(self.model_dir, str(job_id))

    def get_additional_params(self, event, data, job_id):
        return {'workdir': self._job_dir(job_id)}

    def _get_result_from_job_id(self, job_id):
        """
        Return job result or {}
        @param job_id: Job id (also known as model version)
        @return: dict
        """
        job_dir = self._job_dir(job_id)
        if not os.path.exists(job_dir):
            logger.warning(f"=> Warning: {job_dir} dir doesn't exist. "
                           f"It seems that you don't have specified model dir.")
            return
        result_file = os.path.join(job_dir, self.JOB_RESULT)
        if not os.path.exists(result_file):
            logger.warning(f"=> Warning: {job_dir} dir doesn't contain result file {result_file} "
                           f"It seems that previous training session ended with error."
                           f"If you haven't implemented fit() method - ignore this message.")
            return
        logger.debug(f'Read result from {result_file}')
        with open(result_file) as f:
            result = json.load(f)
        return result

    def iter_finished_jobs(self):
        logger.debug(f'Try fetching last valid job id from directory {self.model_dir}')
        return reversed(sorted(map(int, filter(lambda d: d.isdigit(), os.listdir(self.model_dir)))))

    def post_process(self, event, data, job_id, result):
        if not result or not isinstance(result, dict):
            logger.warning(f'Cannot save result {result}')
            return

        job_dir = self._job_dir(job_id)
        os.makedirs(job_dir, exist_ok=True)
        with open(os.path.join(job_dir, 'event.json'), mode='w') as f:
            event_data = {'event': event, 'job_id': job_id}
            if data:
                event_data['data'] = data
            json.dump(event_data, f, indent=2)

        result_file = os.path.join(job_dir, self.JOB_RESULT)
        logger.debug(f'Saving job {job_id} result to file: {result_file}')
        with open(result_file, mode='w') as f:
            json.dump(result, f)

    def run_job(self, model_class, args: tuple):
        if self.async_job:
            proc = mp.Process(target=self.job, args=tuple([model_class] + list(args)))
            proc.daemon = True
            proc.start()
            logger.info(f'Subprocess {proc.pid} has been started with args={args}')
        else:
            self.job(model_class, *args)


class RQJobManager(JobManager):
    """
    RQ-based Job Manager runs all background jobs in RQ workers and requires Redis server to be installed.
    All jobs results are be stored and could be retrieved from Redis queue.
    """

    MAX_QUEUE_LEN = 1  # Controls a maximal amount of simultaneous jobs running in queue.
    # If exceeded, new jobs are ignored

    def __init__(self, redis_host, redis_port, redis_queue):
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.redis_queue = redis_queue

    def _get_redis(self, host, port, raise_on_error=False):
        r = Redis(host=host, port=port)
        try:
            r.ping()
        except redis.ConnectionError:
            if raise_on_error:
                raise
            return None
        else:
            return r

    @contextmanager
    def start_run(self, event, data, job_id):
        # Each "job" record in queue already encapsulates each run
        yield

    def run_job(self, model_class, args: tuple):
        # Launch background job with RQ (production mode)
        event, data, job_id = args
        redis = self._get_redis(self.redis_host, self.redis_port)
        queue = Queue(name=self.redis_queue, connection=redis)
        if len(queue) >= self.MAX_QUEUE_LEN:
            logger.warning(f'Maximal RQ queue len {self.MAX_QUEUE_LEN} reached. Job is not started.')
            return
        job = queue.enqueue(
            self.job,
            args=(model_class, event, data, job_id),
            job_id=job_id,
            job_timeout='365d',
            ttl=None,
            result_ttl=-1,
            failure_ttl=300)
        assert job.id == job_id
        logger.info(f'RQ job {job_id} has been started for event={event}')

    def _get_result_from_job_id(self, job_id):
        redis = self._get_redis(self.redis_host, self.redis_port)
        job = Job.fetch(job_id, connection=redis)
        return job.result

    def iter_finished_jobs(self):
        redis = self._get_redis(self.redis_host, self.redis_port)
        finished_jobs = FinishedJobRegistry(self.redis_queue, redis)
        jobs = []
        for job_id in finished_jobs.get_job_ids():
            job = Job.fetch(job_id, connection=redis)
            jobs.append((job_id, job.ended_at))
        return (j[0] for j in reversed(sorted(jobs, key=lambda job: job[1])))

    def post_process(self, event, data, job_id, result):
        pass


class LabelStudioMLBase(ABC):
    
    TRAIN_EVENTS = (
        'ANNOTATION_CREATED',
        'ANNOTATION_UPDATED',
        'ANNOTATION_DELETED',
        'PROJECT_UPDATED'
    )

    def __init__(self, label_config=None, train_output=None, **kwargs):
        """Model loader"""
        self.label_config = label_config
        self.parsed_label_config = parse_config(self.label_config) if self.label_config else {}
        self.train_output = train_output or {}
        self.hostname = kwargs.get('hostname', '') or get_env('HOSTNAME')
        self.access_token = kwargs.get('access_token', '') or get_env('ACCESS_TOKEN') or get_env('API_KEY')

    @abstractmethod
    def predict(self, tasks, **kwargs):
        pass

    def process_event(self, event, data, job_id, additional_params):
        if event in self.TRAIN_EVENTS:
            logger.debug(f'Job {job_id}: Received event={event}: calling {self.__class__.__name__}.fit()')
            train_output = self.fit(event=event, data=data, job_id=job_id, **additional_params)
            logger.debug(f'Job {job_id}: Train finished.')
            return train_output

    def fit(self, event, data, job_id, **additional_params):
        return {}

    def get_local_path(self, url, project_dir=None):
        return get_local_path(url, project_dir=project_dir, hostname=self.hostname, access_token=self.access_token)


class LabelStudioMLManager(object):

    model_class = None
    model_dir = None
    redis_host = None
    redis_port = None
    redis_queue = None
    train_kwargs = None

    _redis = None
    _redis_queue = None
    _current_model = None

    @classmethod
    def initialize(
        cls, model_class, model_dir=None, redis_host='localhost', redis_port=6379, redis_queue='default',
        **init_kwargs
    ):
        if not issubclass(model_class, LabelStudioMLBase):
            raise ValueError('Inference class should be the subclass of ' + LabelStudioMLBase.__class__.__name__)

        cls.model_class = model_class
        cls.redis_queue = redis_queue
        cls.model_dir = model_dir
        cls.init_kwargs = init_kwargs
        cls.redis_host = redis_host
        cls.redis_port = redis_port

        if cls.model_dir:
            cls.model_dir = os.path.expanduser(cls.model_dir)
            os.makedirs(cls.model_dir, exist_ok=True)

        cls._redis = None
        if get_bool_env('USE_REDIS', False):
            cls._redis = cls._get_redis(redis_host, redis_port)
        if cls._redis:
            cls._redis_queue = Queue(name=redis_queue, connection=cls._redis)

    @classmethod
    def get_initialization_params(cls):
        return dict(
            model_class=cls.model_class,
            model_dir=cls.model_dir,
            redis_host=cls.redis_host,
            redis_port=cls.redis_port,
            redis_queue=cls.redis_queue,
            **cls.init_kwargs
        )

    @classmethod
    def without_redis(cls):
        return cls._redis is None

    @classmethod
    def _get_redis(cls, host, port, raise_on_error=False):
        r = Redis(host=host, port=port)
        try:
            r.ping()
        except redis.ConnectionError:
            if raise_on_error:
                raise
            return None
        else:
            return r

    @classmethod
    def _generate_version(cls):
        return str(int(datetime.now().timestamp()))

    @classmethod
    def _get_tasks_key(cls, project):
        return 'project:' + str(project) + ':tasks'

    @classmethod
    def _get_job_results_key(cls, project):
        return 'project:' + str(project) + ':job_results'

    @classmethod
    def _remove_jobs(cls, project):
        started_registry = StartedJobRegistry(cls._redis_queue.name, cls._redis_queue.connection)
        finished_registry = FinishedJobRegistry(cls._redis_queue.name, cls._redis_queue.connection)
        for job_id in started_registry.get_job_ids() + finished_registry.get_job_ids():
            job = Job.fetch(job_id, connection=cls._redis)
            if job.meta.get('project') != project:
                continue
            logger.info('Deleting job_id ' + job_id)
            job.delete()

    @classmethod
    def _get_latest_job_result_from_redis(cls, project):
        job_results_key = cls._get_job_results_key(project)
        try:
            num_finished_jobs = cls._redis.llen(job_results_key)
            if num_finished_jobs == 0:
                logger.info('Job queue is empty')
                return
            latest_job = cls._redis.lindex(job_results_key, -1)
        except redis.exceptions.ConnectionError as exc:
            logger.error(exc)
            return
        else:
            return json.loads(latest_job)

    @classmethod
    def _get_latest_job_result_from_workdir(cls, project):
        project_model_dir = os.path.join(cls.model_dir, project or '')
        if not os.path.exists(project_model_dir):
            return

        # sort directories by decreasing timestamps
        for subdir in reversed(sorted(map(int, filter(lambda d: d.isdigit(), os.listdir(project_model_dir))))):
            job_result_file = os.path.join(project_model_dir, str(subdir), 'job_result.json')
            if not os.path.exists(job_result_file):
                logger.error('The latest job result file ' + job_result_file + ' doesn\'t exist')
                continue
            with open(job_result_file) as f:
                return json.load(f)

    @classmethod
    def _key(cls, project):
        return project, os.getpid()

    @classmethod
    def has_active_model(cls, project):
        return cls._current_model is not None

    @classmethod
    def get_current_model_version(cls):
        if cls._current_model:
            return cls._current_model.model_version

    @classmethod
    def get(cls, project):
        return cls._current_model

    @classmethod
    def create(cls, project=None, label_config=None, train_output=None, version=None, **kwargs):
        key = cls._key(project)
        logger.debug('Create project ' + str(key))
        kwargs.update(cls.init_kwargs)
        cls._current_model = ModelWrapper(
            model=cls.model_class(label_config=label_config, train_output=train_output, **kwargs),
            model_version=version or cls._generate_version()
        )
        return cls._current_model

    @classmethod
    def get_or_create(
        cls, project=None, label_config=None, force_reload=False, train_output=None, version=None, **kwargs
    ):
        m = cls.get(project)
        # reload new model if model is not loaded into memory OR force_reload=True OR model versions are mismatched
        if not cls.has_active_model(project) or \
                force_reload or \
                m is not None or \
                (m.model_version != version and version is not None):  # noqa
            logger.debug('Reload model for project={project} with version={version}'.format(
                project=project, version=version))
            cls.create(project, label_config, train_output, version, **kwargs)
        return cls.get(project)

    @classmethod
    def fetch(cls, project=None, label_config=None, force_reload=False, **kwargs):
        """
        Fetch the model

        @param project: Project
        @param label_config: Project label config
        @param force_reload: Force reload the model
        @param kwargs: additional params
        @return:
        Current model
        """
        # update kwargs with init kwargs from class (e.g. --with start arg)
        kwargs.update(cls.init_kwargs)

        model_version = kwargs.get('model_version')
        if not cls._current_model or (model_version != cls._current_model.model_version and model_version is not None) or \
                os.getenv('AUTO_UPDATE', default=AUTO_UPDATE_DEFAULT):
            jm = cls.get_job_manager()
            model_version = kwargs.get('model_version')
            job_result = jm.get_result(model_version)
            if job_result:
                logger.debug(f'Found job result: {job_result}')
                model = cls.model_class(label_config=label_config, train_output=job_result, **kwargs)
                cls._current_model = ModelWrapper(model=model, model_version=job_result['job_id'])
            else:
                logger.debug(f'Job result not found: create initial model')
                model = cls.model_class(label_config=label_config, **kwargs)
                cls._current_model = ModelWrapper(model=model, model_version='INITIAL')
        return cls._current_model

    @classmethod
    def job_status(cls, job_id):
        job = Job.fetch(job_id, connection=cls._redis)
        response = {
            'job_status': job.get_status(),
            'error': job.exc_info,
            'created_at': job.created_at,
            'enqueued_at': job.enqueued_at,
            'started_at': job.started_at,
            'ended_at': job.ended_at
        }
        if job.is_finished and isinstance(job.result, str):
            response['result'] = json.loads(job.result)
        return response

    @classmethod
    def predict(
        cls, tasks, **kwargs
    ):
        """
        Make prediction for tasks

        @param tasks: Serialized LS tasks
        @param project: Project ID (e.g. {project.id}.{created_timestamp}
        @param label_config: Label studio project label config
        @param force_reload: force reload the model
        @param try_fetch: if service should try to fetch the model
        @param kwargs: additional params
        @return:
        Predictions in LS format
        """
        if not cls._current_model:
            raise ValueError(f'Model is not loaded for {cls.__class__.__name__}: run setup() before using predict()')

        predictions = cls._current_model.model.predict(tasks, **kwargs)
        return predictions, cls._current_model

    @classmethod
    def get_job_manager(cls):
        if cls.without_redis():
            # Launch background job with fork (dev mode)
            job_man = SimpleJobManager(model_dir=cls.model_dir)
        else:
            # Launch background job with RQ (production mode)
            job_man = RQJobManager(redis_host=cls.redis_host, redis_port=cls.redis_port, redis_queue=cls.redis_queue)
        return job_man

    @classmethod
    def webhook(cls, event, data):
        job_id = cls._generate_version()
        cls.get_job_manager().run_job(cls.model_class, (event, data, job_id))
        return {'job_id': job_id}

    @classmethod
    def _get_models_from_workdir(cls):
        """
        Return current models
        @param project: Project ID (e.g. {project.id}.{created_timestamp}
        @return: List of model versions for current model
        """
        project_model_dir = cls.model_dir
        # get directories with training results
        final_models = []
        for subdir in map(int, filter(lambda d: d.isdigit(), os.listdir(project_model_dir))):
            job_result_file = os.path.join(project_model_dir, str(subdir), 'job_result.json')
            # check if there is job result
            if not os.path.exists(job_result_file):
                continue
            with open(job_result_file) as f:
                # Add model version if status is ok
                final_models.append(subdir)
        return final_models


def get_all_classes_inherited_LabelStudioMLBase(script_file):
    names = set()
    abs_path = os.path.abspath(script_file)
    module_name = os.path.splitext(os.path.basename(script_file))[0]
    sys.path.append(os.path.dirname(abs_path))
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as e:
        print(Fore.RED + 'Can\'t import module "' + module_name + f'", reason: {e}.\n'
              'If you are looking for examples, you can find a dummy model.py here:\n' +
              Fore.LIGHTYELLOW_EX + 'https://labelstud.io/tutorials/dummy_model.html')
        module = None
        exit(-1)

    for name, obj in inspect.getmembers(module, inspect.isclass):
        if name == LabelStudioMLBase.__name__:
            continue
        if issubclass(obj, LabelStudioMLBase):
            names.add(name)
        for base in obj.__bases__:
            if LabelStudioMLBase.__name__ == base.__name__:
                names.add(name)
    sys.path.pop()
    return list(names)
