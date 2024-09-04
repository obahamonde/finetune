from typing import Literal, Optional

import pandas as pd  # type: ignore
from typing_extensions import Required, TypedDict


class S3AdditonalKwargs(TypedDict, total=False):
    ServerSideEncryption: Required[Literal["AES256", "aws:kms"]]
    SSEKMSKeyId: Optional[str]
    StorageClass: Optional[
        Literal[
            "STANDARD",
            "INTELLIGENT_TIERING",
            "STANDARD_IA",
            "ONEZONE_IA",
            "INTELLIGENT_TIERING_AFTER_DELETION",
            "GLACIER",
            "DEEP_ARCHIVE",
            "OUTPOSTS",
        ]
    ]
    SSECustomerAlgorithm: Optional[Literal["AES256", "aws:kms"]]
    SSECustomerKey: Optional[str]
    SSECustomerKeyMD5: Optional[str]
    SSEKMSKeyId: Optional[str]
    BucketKeyEnabled: Optional[bool]
    ChecksumMode: Optional[Literal["ENABLED", "DISABLED"]]
    ChecksumModeKey: Optional[str]
    ChecksumModeMD5: Optional[str]
    Tagging: Optional[str]
    TagDirective: Optional[Literal["COPY", "REPLACE"]]
    MetadataDirective: Optional[Literal["COPY", "REPLACE"]]
    SSEContext: Optional[str]
    SSEKMSContext: Optional[str]
    SSECustomerAlgorithmContext: Optional[str]
    SSECustomerKeyContext: Optional[str]
    SSECustomerKeyMD5Context: Optional[str]
    ChecksumModeContext: Optional[str]
    S3StorageClassContext: Optional[str]
    S3TaggingContext: Optional[str]
    S3MetadataDirectiveContext: Optional[str]
    S3TaggingDirectiveContext: Optional[str]


class ReadParams(TypedDict, total=False):
    path: Required[str]
    dataset: Required[bool]
    index: Required[bool]
    mode: Required[Literal["overwrite", "append", "overwrite_partitions"]]
    compression: Required[Literal["gzip", "snappy", "none"]]
    schema_evolution: Required[bool]
    s3_additional_kwargs: Required[S3AdditonalKwargs]
    table: Optional[str]
    catalog_id: Optional[str]
    database: Optional[str]
    table_type: Optional[Literal["table", "view"]]
    partition_by: Optional[list[str]]
    partition_number: Optional[int]


class WriteParams(TypedDict, total=False):
    df: Required[pd.DataFrame]
    path: Required[str]
    dataset: Required[bool]
    index: Required[bool]
    mode: Required[Literal["overwrite", "append", "overwrite_partitions"]]
    compression: Required[Literal["gzip", "snappy", "none"]]
    schema_evolution: Required[bool]
    s3_additional_kwargs: Required[S3AdditonalKwargs]
    table: Optional[str]
    catalog_id: Optional[str]
    database: Optional[str]
    table_type: Optional[Literal["table", "view"]]
    partition_by: Optional[list[str]]
    partition_number: Optional[int]
