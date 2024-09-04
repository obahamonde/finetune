from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, field
from functools import cached_property
from typing import Optional, TypeVar

from typing_extensions import ParamSpec

T = TypeVar("T")
P = ParamSpec("P")


@dataclass
class JobException(BaseException):
    code: int = field(default=500)
    message: str = field(default="Error on ETL Job")

    def __init__(self, message: Optional[str] = None, code: Optional[int] = None):
        if message:
            self.message = message
        if code:
            self.code = code
        self.logger.error(json.dumps({"error": self.message, "code": self.code}))
        sys.exit(self.code)

    @cached_property
    def logger(self):
        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger


@dataclass
class InputNotFound(JobException):
    def __init__(self, message: str = "Input wasn't provided"):
        super().__init__(message=message, code=404)


@dataclass
class ColumnNotFound(JobException):
    def __init__(self, message: str = "Column not found in DataFrame"):
        super().__init__(message=message, code=404)


@dataclass
class DataFrameEmpty(JobException):
    def __init__(self, message: str = "The DataFrame is empty"):
        super().__init__(message=message, code=204)


@dataclass
class ExtraData(JobException):
    def __init__(self, message: str = "Duplicate column in DataFrame"):
        super().__init__(message=message, code=400)


@dataclass
class InvalidColumn(JobException):
    def __init__(self, message: str = "Invalid column in DataFrame"):
        super().__init__(message=message, code=400)


@dataclass
class InvalidDataFrame(JobException):
    def __init__(self, message: str = "Invalid DataFrame"):
        super().__init__(message=message, code=400)


@dataclass
class DataFrameNotFound(JobException):
    def __init__(self, message: str = "DataFrame not found"):
        super().__init__(message=message, code=404)


@dataclass
class InvalidPath(JobException):
    def __init__(self, message: str = "Invalid path"):
        super().__init__(message=message, code=400)


@dataclass
class ParameterNotFound(JobException):
    def __init__(self, message: str = "Parameter not found"):
        super().__init__(message=message, code=404)


@dataclass
class S3ObjectNotFound(JobException):
    def __init__(self, message: str):
        super().__init__(message=f"S3 Object not found: {message}", code=404)


__all__ = [
    "JobException",
    "InputNotFound",
    "ColumnNotFound",
    "DataFrameEmpty",
    "InvalidColumn",
    "InvalidDataFrame",
    "DataFrameNotFound",
    "InvalidPath",
    "ParameterNotFound",
    "S3ObjectNotFound",
]
