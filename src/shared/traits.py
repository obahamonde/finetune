import json
import logging
import sys
import time
from dataclasses import dataclass, field
from functools import cached_property, wraps
from typing import Callable, Generic, Optional, TypeVar

from typing_extensions import ParamSpec

from .exceptions import (
    ColumnNotFound,
    DataFrameNotFound,
    ExtraData,
    InvalidColumn,
    InvalidPath,
    JobException,
    ParameterNotFound,
    S3ObjectNotFound,
)

T = TypeVar("T")
P = ParamSpec("P")


class JobReportTool:
    @cached_property
    def logger(self):
        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            json.dumps({"time": "%(asctime)s", "message": "%(message)s"}, indent=4),
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def log(self, func: Callable[P, T]) -> Callable[P, T]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            self.logger.info("Proceso %s iniciado", func.__name__)
            result = func(*args, **kwargs)
            self.logger.info("Resultado del proceso %s: %s", func.__name__, result)
            return result

        return wrapper

    def timer(self, func: Callable[P, T]) -> Callable[P, T]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            self.logger.info(
                "Proceso %s completado en %s segundos",
                func.__name__,
                round(end_time - start_time, 2),
            )
            return result

        return wrapper

    def exception(self, func: Callable[P, T]) -> Callable[P, T]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            try:
                return func(*args, **kwargs)
            except AttributeError as e:
                raise InvalidColumn(str(e)) from e
            except RuntimeError as e:
                raise InvalidPath(str(e)) from e
            except TypeError as e:
                raise ParameterNotFound(str(e)) from e
            except KeyError as e:
                raise ColumnNotFound(str(e)) from e
            except ValueError as e:
                raise ExtraData(str(e)) from e
            except FileNotFoundError as e:
                raise DataFrameNotFound(str(e)) from e
            except (ConnectionError, TimeoutError) as e:
                raise S3ObjectNotFound(str(e)) from e
            except IndexError as e:
                raise DataFrameNotFound(str(e)) from e
            except Exception as e:
                raise JobException(str(e), 500) from e

        return wrapper

    def __call__(self, func: Callable[P, T]) -> Callable[P, T]:
        @self.log
        @self.timer
        @self.exception
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            wrapper.__name__ = func.__name__
            return func(*args, **kwargs)

        return wrapper


@dataclass
class ETLJob(Generic[T]):
    data: Optional[T] = field(default=None)
    report: JobReportTool = field(default_factory=JobReportTool)

    def extract(self, func: Callable[P, T]) -> Callable[P, T]:
        @self.report
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs):
            wrapper.__name__ = func.__name__
            return func(*args, **kwargs)

        return wrapper

    def transform(self, func: Callable[P, T]) -> Callable[P, T]:
        @self.report
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs):
            wrapper.__name__ = func.__name__
            return func(*args, **kwargs)

        return wrapper

    def load(self, func: Callable[P, T]) -> Callable[P, T]:
        @self.report
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs):
            wrapper.__name__ = func.__name__
            return func(*args, **kwargs)

        return wrapper

    def __call__(self, func: Callable[P, T]):
        @self.report
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs):
            wrapper.__name__ = func.__name__
            return func(*args, **kwargs)

        return wrapper
