import os
import sqlite3
from abc import ABC, abstractmethod
from threading import Lock
from functools import lru_cache


class BaseCache(ABC):

    def __init__(self, path):
        self.path = path

    @abstractmethod
    def __getitem__(self, project_id_key: tuple):
        """
        Get value from cache
        :param project_id_key: tuple (project_id, key)
        :return:
        """

    @abstractmethod
    def __setitem__(self, project_id_key: tuple, value):
        """
        Set value to cache
        :param project_id_key: tuple (project_id, key)
        :param value:
        :return:
        """

    @abstractmethod
    def __contains__(self, project_id_key: tuple):
        """
        Check if value exists in cache
        :param project_id_key: tuple (project_id, key)
        :return:
        """

    @abstractmethod
    def __delitem__(self, project_id_key: tuple):
        """
        Delete value from cache
        :param project_id_key: tuple (project_id, key)
        :return:
        """


class SqliteCache(BaseCache):
    def __init__(self, path: str, db_name: str = 'cache.db'):
        super(SqliteCache, self).__init__(path)
        os.makedirs(self.path, exist_ok=True)
        self.db_name = os.path.join(self.path, db_name)
        self.lock = Lock()

        # Establish a connection and create table if it doesn't exist
        with self.lock, sqlite3.connect(self.db_name) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS cache (
                    project_id TEXT NOT NULL,
                    key TEXT NOT NULL,  
                    value TEXT NOT NULL,
                    PRIMARY KEY (project_id, key)
                );
            ''')

    @lru_cache(maxsize=100)
    def __getitem__(self, project_id_key):
        project_id, key = project_id_key
        with self.lock, sqlite3.connect(self.db_name) as conn:
            cursor = conn.cursor()
            cursor.execute(
                'SELECT value FROM cache WHERE project_id = ? AND key = ?;',
                (project_id, key))
            result = cursor.fetchone()
            if result is None:
                return result
            return result[0]

    def __setitem__(self, project_id_key, value):
        project_id, key = project_id_key
        if not isinstance(value, str):
            raise ValueError('Value must be a string')
        with self.lock, sqlite3.connect(self.db_name) as conn:
            cursor = conn.cursor()
            cursor.execute('REPLACE INTO cache (project_id, key, value) VALUES (?, ?, ?);',
                           (project_id, key, value))
        self.__getitem__.cache_clear()

    def __delitem__(self, project_id_key):
        project_id, key = project_id_key
        with self.lock, sqlite3.connect(self.db_name) as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM cache WHERE project_id = ? AND key = ?;',
                           (project_id, key))
        self.__getitem__.cache_clear()

    def __contains__(self, project_id_key):
        project_id, key = project_id_key
        with self.lock, sqlite3.connect(self.db_name) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT 1 FROM cache WHERE project_id = ? AND key = ?;',
                           (project_id, key))
            return cursor.fetchone() is not None


def create_cache(cache_type, path, **kwargs):
    if cache_type == 'sqlite':
        return SqliteCache(path, **kwargs)
    else:
        raise ValueError(f"Unsupported cache type: {cache_type}")
