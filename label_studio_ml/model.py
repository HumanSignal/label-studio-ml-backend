import os
import logging
import sys
import time
import json
import redis
import attr
import io
import multiprocessing as mp
import importlib
import importlib.util
import inspect

from typing import Dict
from abc import ABC, abstractmethod
from datetime import datetime
from itertools import tee
from contextlib import contextmanager
from redis import Redis
from rq import Queue, get_current_job
from rq.registry import StartedJobRegistry, FinishedJobRegistry, FailedJobRegistry
from rq.job import Job
from colorama import Fore

from label_studio_tools.core.utils.params import get_bool_env
from label_studio_tools.core.label_config import parse_config

logger = logging.getLogger(__name__)

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
        with self.start_run(event, data, job_id):
            model_version = data.get('model_version') or data.get('project', {}).get('model_version')
            job_result = self.get_result(model_version)
            label_config = data.get('label_config') or data.get('project', {}).get('label_config')
            train_output = job_result.get('train_output')
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
        assert isinstance(result, dict)
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

    def __init__(self, model_dir='.'):
        self.model_dir = model_dir

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
        job_dir = self._job_dir(job_id)
        if not os.path.exists(job_dir):
            raise IOError(f'Run directory {job_dir} specified by model_version doesn\'t exist')
        result_file = os.path.join(job_dir, self.JOB_RESULT)
        if not os.path.exists(result_file):
            raise IOError(f'Result file {result_file} specified by model_version doesn\'t exist')
        logger.debug(f'Read result from {result_file}')
        with open(result_file) as f:
            result = json.load(f)
        return result

    def iter_finished_jobs(self):
        logger.debug(f'Try fetching last valid job id from directory {self.model_dir}')
        return reversed(sorted(map(int, filter(lambda d: d.isdigit(), os.listdir(self.model_dir)))))

    def post_process(self, event, data, job_id, result):
        if isinstance(result, dict):
            result_file = os.path.join(self._job_dir(job_id), self.JOB_RESULT)
            logger.debug(f'Saving job {job_id} result to file: {result_file}')
            with open(result_file, mode='w') as f:
                json.dump(result, f)
        else:
            logger.info(f'Cannot save result {result}')

    def run_job(self, model_class, args: tuple):
        proc = mp.Process(target=self.job, args=tuple([model_class] + list(args)))
        proc.daemon = True
        proc.start()
        logger.info(f'Subprocess {proc.pid} has been started with args={args}')


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
        'ANNOTATION_DELETED'
    )

    def __init__(self, label_config=None, train_output=None, **kwargs):
        """Model loader"""
        self.label_config = label_config
        self.parsed_label_config = parse_config(self.label_config) if self.label_config else {}
        self.train_output = train_output or {}
        self.hostname = kwargs.get('hostname', '')
        self.access_token = kwargs.get('access_token', '')

    @abstractmethod
    def predict(self, tasks, **kwargs):
        pass

    def process_event(self, event, data, job_id, additional_params):
        if event in self.TRAIN_EVENTS:
            logger.debug(f'Job {job_id}: Received event={event}: calling {self.__class__.__name__}.fit()')
            train_output = self.fit((), event=event, data=data, job_id=job_id, **additional_params)
            logger.debug(f'Job {job_id}: Train finished.')
            return train_output

    def fit(self, tasks, workdir=None, **kwargs):
        return {}

    def get_local_path(self, url, project_dir=None):
        from label_studio_ml.utils import get_local_path
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
    _current_model = {}

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
        return cls._key(project) in cls._current_model

    @classmethod
    def get(cls, project):
        key = cls._key(project)
        logger.debug('Get project ' + str(key))
        return cls._current_model.get(key)

    @classmethod
    def create(cls, project=None, label_config=None, train_output=None, version=None, **kwargs):
        key = cls._key(project)
        logger.debug('Create project ' + str(key))
        kwargs.update(cls.init_kwargs)
        cls._current_model[key] = ModelWrapper(
            model=cls.model_class(label_config=label_config, train_output=train_output, **kwargs),
            model_version=version or cls._generate_version()
        )
        return cls._current_model[key]

    @classmethod
    def get_or_create(
        cls, project=None, label_config=None, force_reload=False, train_output=None, version=None, **kwargs
    ):
        # reload new model if model is not loaded into memory OR force_reload=True OR model versions are mismatched
        if not cls.has_active_model(project) or force_reload or (cls.get(project).model_version != version and version is not None):  # noqa
            logger.debug('Reload model for project={project} with version={version}'.format(
                project=project, version=version))
            cls.create(project, label_config, train_output, version, **kwargs)
        return cls.get(project)

    @classmethod
    def fetch(cls, project=None, label_config=None, force_reload=False, **kwargs):
        if not os.getenv('LABEL_STUDIO_ML_BACKEND_V2'):
            # TODO: Deprecated branch
            if cls.without_redis():
                logger.debug('Fetch ' + project + ' from local directory')
                job_result = cls._get_latest_job_result_from_workdir(project) or {}
            else:
                logger.debug('Fetch ' + project + ' from Redis')
                job_result = cls._get_latest_job_result_from_redis(project) or {}
            train_output = job_result.get('train_output')
            version = job_result.get('version')
            return cls.get_or_create(project, label_config, force_reload, train_output, version, **kwargs)

        model_version = kwargs.get('model_version')
        if not cls._current_model or model_version != cls._current_model.model_version:
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
    def is_training(cls, project):
        if not cls.has_active_model(project):
            return {'is_training': False}
        m = cls.get(project)
        if cls.without_redis():
            return {
                'is_training': m.is_training,
                'backend': 'none',
                'model_version': m.model_version
            }
        else:
            started_jobs = StartedJobRegistry(cls._redis_queue.name, cls._redis_queue.connection).get_job_ids()
            finished_jobs = FinishedJobRegistry(cls._redis_queue.name, cls._redis_queue.connection).get_job_ids()
            failed_jobs = FailedJobRegistry(cls._redis_queue.name, cls._redis_queue.connection).get_job_ids()
            running_jobs = list(set(started_jobs) - set(finished_jobs + failed_jobs))
            logger.debug('Running jobs: ' + str(running_jobs))
            for job_id in running_jobs:
                job = Job.fetch(job_id, connection=cls._redis)
                if job.meta.get('project') == project:
                    return {
                        'is_training': True,
                        'job_id': job_id,
                        'backend': 'redis',
                        'model_version': m.model_version,
                    }
            return {
                'is_training': False,
                'backend': 'redis',
                'model_version': m.model_version
            }

    @classmethod
    def predict(
        cls, tasks, project=None, label_config=None, force_reload=False, try_fetch=True, **kwargs
    ):
        if not os.getenv('LABEL_STUDIO_ML_BACKEND_V2'):
            if try_fetch:
                m = cls.fetch(project, label_config, force_reload)
            else:
                m = cls.get(project)
                if not m:
                    raise FileNotFoundError('No model loaded. Specify "try_fetch=True" option.')
            predictions = m.model.predict(tasks, **kwargs)
            return predictions, m

        if not cls._current_model:
            raise ValueError(f'Model is not loaded for {cls.__class__.__name__}: run setup() before using predict()')

        predictions = cls._current_model.model.predict(tasks, **kwargs)
        return predictions, cls._current_model

    @classmethod
    def create_data_snapshot(cls, data_iter, workdir):
        data = list(data_iter)
        data_file = os.path.join(workdir, 'train_data.json')
        with io.open(data_file, mode='w') as fout:
            json.dump(data, fout, ensure_ascii=False)

        info_file = os.path.join(workdir, 'train_data_info.json')
        with io.open(info_file, mode='w') as fout:
            json.dump({'count': len(data)}, fout)

    @classmethod
    def train_script_wrapper(
        cls, project, label_config, train_kwargs, initialization_params=None, tasks=()
    ):

        if initialization_params:
            # Reinitialize new cls instance for using in RQ context
            initialization_params = initialization_params or {}
            cls.initialize(**initialization_params)

        # fetching the latest model version before we generate the next one
        t = time.time()
        m = cls.fetch(project, label_config)
        m.is_training = True

        version = cls._generate_version()

        if cls.model_dir:
            logger.debug('Running in model dir: ' + cls.model_dir)
            project_model_dir = os.path.join(cls.model_dir, project or '')
            workdir = os.path.join(project_model_dir, version)
            os.makedirs(workdir, exist_ok=True)
        else:
            logger.debug('Running without model dir')
            workdir = None

        if cls.without_redis():
            data_stream = tasks
        else:
            data_stream = (json.loads(t) for t in cls._redis.lrange(cls._get_tasks_key(project), 0, -1))

        if workdir:
            data_stream, snapshot = tee(data_stream)
            cls.create_data_snapshot(snapshot, workdir)

        try:
            train_output = m.model.fit(data_stream, workdir, **train_kwargs)
            if cls.without_redis():
                job_id = None
            else:
                job_id = get_current_job().id
            job_result = json.dumps({
                'status': 'ok',
                'train_output': train_output,
                'project': project,
                'workdir': workdir,
                'version': version,
                'job_id': job_id,
                'time': time.time() - t
            })
            if workdir:
                job_result_file = os.path.join(workdir, 'job_result.json')
                with open(job_result_file, mode='w') as fout:
                    fout.write(job_result)
            if not cls.without_redis():
                cls._redis.rpush(cls._get_job_results_key(project), job_result)
        except:
            raise
        finally:
            m.is_training = False
        return job_result

    @classmethod
    def _start_training_job(cls, project, label_config, train_kwargs):
        job = cls._redis_queue.enqueue(
            cls.train_script_wrapper,
            args=(project, label_config, train_kwargs, cls.get_initialization_params()),
            job_timeout='365d',
            ttl=None,
            result_ttl=-1,
            failure_ttl=300,
            meta={'project': project},
        )
        logger.info('Training job {job} started for project {project}'.format(job=job, project=project))
        return job

    @classmethod
    def train(cls, tasks, project=None, label_config=None, **kwargs):
        job = None
        if cls.without_redis():
            job_result = cls.train_script_wrapper(
                project, label_config, train_kwargs=kwargs, tasks=tasks)
            train_output = json.loads(job_result)['train_output']
            cls.get_or_create(project, label_config, force_reload=True, train_output=train_output)
        else:
            tasks_key = cls._get_tasks_key(project)
            cls._redis.delete(tasks_key)
            for task in tasks:
                cls._redis.rpush(tasks_key, json.dumps(task))
            job = cls._start_training_job(project, label_config, kwargs)
        return job

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


def get_all_classes_inherited_LabelStudioMLBase(script_file):
    names = []
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
            names.append(name)
    sys.path.pop()
    return names