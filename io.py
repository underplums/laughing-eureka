import logging
from contextlib import contextmanager
from datetime import datetime, timedelta
from functools import lru_cache
from io import BytesIO
from typing import Dict, Generator, List, cast, Optional

import sqlalchemy as sa
from sqlalchemy.pool import NullPool
from decouple import config
from pydantic import BaseModel, SecretStr, computed_field
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from s3fs import S3FileSystem
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.engine.url import URL
from typing_extensions import Self

from cvm_model.object_url import ObjectPath


class CredsDatabase(BaseModel, arbitrary_types_allowed=True):
    driver_name: str
    username: str
    password: SecretStr
    host: str
    port: int
    database: str
    params: Dict[str, str]

    @computed_field
    @property
    def sa_engine(self) -> Engine:
        engine = sa.create_engine(
            url=URL.create(
                drivername=self.driver_name,
                username=self.username,
                password=self.password.get_secret_value(),
                host=self.host,
                port=self.port,
                database=self.database,
                query=self.params,
            ),
            poolclass=NullPool,
        )
        return engine

    @computed_field
    @property
    def jdbc_url(self) -> str:
        dbms = self.driver_name.split('+')[0]
        return f'jdbc:{dbms}://{self.host}:{self.port}/{self.database}'

    @contextmanager
    def sa_connect(self) -> Generator[Connection, None, None]:
        with self.sa_engine.connect() as conn:
            yield conn


class CredsMlFlow(BaseModel, arbitrary_types_allowed=True):
    username: str
    password: SecretStr
    tracking_uri: Optional[str] = None
    insecure_tls: str = 'false'
    server_cert_path: Optional[str] = None

    @computed_field
    @property
    def environment(self) -> Dict[str, str]:
        env = {
            'MLFLOW_TRACKING_USERNAME': self.username,
            'MLFLOW_TRACKING_PASSWORD': self.password.get_secret_value(),
            'MLFLOW_TRACKING_INSECURE_TLS': self.insecure_tls,
        }

        if self.server_cert_path:
            env['MLFLOW_TRACKING_SERVER_CERT_PATH'] = self.server_cert_path

        return env


class CredsS3(BaseModel, arbitrary_types_allowed=True):
    endpoint: str
    region: str
    access_key: SecretStr
    secret_key: SecretStr

    @computed_field
    @property
    def s3fs(self) -> S3FileSystem:
        return S3FileSystem(
            client_kwargs={
                'endpoint_url': self.endpoint,
                'region_name': self.region,
            },
            key=self.access_key.get_secret_value(),
            secret=self.secret_key.get_secret_value(),
        )

    @computed_field
    @property
    def storage_options(self) -> Dict[str, str]:
        storage_options = {
            'aws_access_key_id': self.access_key.get_secret_value(),
            'aws_secret_access_key': self.secret_key.get_secret_value(),
            'aws_endpoint': self.endpoint,
            'aws_region': self.region,
        }
        return storage_options

    def rm_dir(self, path: ObjectPath, missing_ok: bool = True):
        logging.info(f'trying to delete dir: {path}...')
        try:
            self.s3fs.rm(path, recursive=True)
        except FileNotFoundError as e:
            if missing_ok:
                logging.info(f'dir doesn\'t exist: {path}')
            else:
                raise e

    def rm(self, path: ObjectPath, missing_ok: bool = True):
        logging.info(f'trying to delete file: {path}')
        try:
            self.s3fs.rm(path, recursive=False)
        except FileNotFoundError as e:
            if missing_ok:
                logging.info(f'file doesn\'t exist: {path}')
            else:
                raise e

    def ls(self, path: ObjectPath) -> List[ObjectPath]:
        logging.info(f'trying to ls: {path}')
        result = [
            ObjectPath(p) for p in cast(List[str], self.s3fs.ls(path, detail=False))
        ]
        logging.info(f'Got {len(result):_} paths')
        return result

    def read_bytes(self, path: ObjectPath) -> BytesIO:
        logging.info(f'trying to read bytes of: {path}')
        buffer = BytesIO()
        file = self.s3fs.cat_file(path)
        assert isinstance(file, bytes)
        buffer.write(file)
        buffer.seek(0)
        return buffer


def get_secret_strings(model: BaseModel) -> List[SecretStr]:
    result: List[SecretStr] = []

    for field_name, _ in model.__class__.model_fields.items():
        field_value = getattr(model, field_name)
        if isinstance(field_value, SecretStr):
            result.append(field_value)
        elif isinstance(field_value, BaseModel):
            result += get_secret_strings(field_value)
        else:
            pass

    return result


class Credentials(BaseModel, arbitrary_types_allowed=True):
    '''Хранилище секретов и подключений'''

    loyalty_gp: CredsDatabase
    cvm_clickhouse: CredsDatabase
    cvm_s3: CredsS3
    mlflow: CredsMlFlow

    @classmethod
    @lru_cache(1)
    def from_env(cls) -> Self:
        return cls(
            loyalty_gp=CredsDatabase(
                driver_name='postgresql+psycopg',
                username=config('GP_LOYALTY_USER'),
                password=config('GP_LOYALTY_PASSWORD', cast=SecretStr),
                host=config('GP_LOYALTY_HOST'),
                port=config('GP_LOYALTY_PORT', cast=int),
                database=config('GP_LOYALTY_DB'),
                params={
                    'keepalives': str(1),
                    'keepalives_idle': str(13),
                    'keepalives_interval': str(3),
                    'keepalives_count': str(6),
                    'tcp_user_timeout': str(
                        int(timedelta(minutes=3).total_seconds() * 1_000)
                    ),
                },
            ),
            cvm_clickhouse=CredsDatabase(
                driver_name='clickhouse+native',
                username=config('CH_CVM_USER'),
                password=config('CH_CVM_PASSWORD', cast=SecretStr),
                host=config('CH_CVM_HOST'),
                port=config('CH_CVM_PORT', cast=int),
                database=config('CH_CVM_DB'),
                params={
                    'connect_timeout': str(timedelta(hours=1).seconds),
                    'distributed_ddl_task_timeout': str(-1),
                },
            ),
            cvm_s3=CredsS3(
                endpoint=config('AWS_ENDPOINT_URL'),
                region=config('AWS_DEFAULT_REGION'),
                access_key=config('AWS_ACCESS_KEY_ID', cast=SecretStr),
                secret_key=config('AWS_SECRET_ACCESS_KEY', cast=SecretStr),
            ),
            mlflow=CredsMlFlow(
                username=config('MLFLOW_TRACKING_USERNAME'),
                password=config('MLFLOW_TRACKING_PASSWORD', cast=SecretStr),
                tracking_uri=config('MLFLOW_TRACKING_URI', default=None),
                insecure_tls=config('MLFLOW_TRACKING_INSECURE_TLS', default='false'),
                server_cert_path=config(
                    'MLFLOW_TRACKING_SERVER_CERT_PATH',
                    default=config('MLFLOW_TRACKING_SERVER_CER_PATH', default=None),
                ),
            ),
        )

    def dump_secrets(self) -> List[SecretStr]:
        return get_secret_strings(self)


class Settings(BaseModel, arbitrary_types_allowed=True):
    s3_permanent_prefix: ObjectPath
    s3_temporary_prefix: ObjectPath
    feature_store_s3_root: ObjectPath

    @classmethod
    def from_env(cls) -> Self:
        return cls(
            s3_permanent_prefix=config('S3_PERMANENT_STORAGE_PREFIX', cast=ObjectPath),
            s3_temporary_prefix=config('S3_TEMPORARY_STORAGE_PREFIX', cast=ObjectPath),
            feature_store_s3_root=config(
                'FEATURE_STORE_S3_ROOT',
                cast=ObjectPath,
                default=ObjectPath('s3://cvm-prod/feature_store/feature_views'),
            ),
        )

    def get_prefix(
        self,
        temp: bool = True,
        event_timestamp: Optional[datetime] = None,
        suffix: Optional[str] = None,
    ) -> ObjectPath:
        prefix = self.s3_temporary_prefix if temp else self.s3_permanent_prefix
        
        if event_timestamp is not None:
            if suffix is not None:
                return prefix / event_timestamp.date().isoformat() / suffix
            else:
                return prefix / event_timestamp.date().isoformat()
        else:
            if suffix is not None:
                return prefix / suffix
            else:
                return prefix

    def get_feature_view_prefix(
        self,
        feature_view: str,
        event_timestamp: Optional[datetime] = None,
    ) -> ObjectPath:
        if event_timestamp is not None:
            return (
                self.feature_store_s3_root
                / feature_view
                / f'event_timestamp={event_timestamp.date().isoformat()}'
            )

        return self.feature_store_s3_root / feature_view


class Spark(BaseModel, arbitrary_types_allowed=True):
    '''Обёртка над `pyspark.sql.SparkSession`

    Attributes:
        session: сама сессия
    '''

    session: SparkSession

    @classmethod
    def from_env(cls, settings: Settings) -> Self:
        _ = settings
        conf = SparkConf()
        conf.setAll(
            [
                # ('spark.sql.parquet.mergeSchema', 'true')
                # ('spark.sql.catalog.dwh', 'org.apache.iceberg.spark.SparkCatalog'),
                # ('spark.sql.catalog.dwh.type', 'hadoop'),
                # (
                #     'spark.sql.catalog.dwh.warehouse',
                #     settings.feature_view_prefix.s3a_protocol,
                # ),
                # (
                #     'spark.sql.extensions',
                #     'org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions',
                # ),
                ('spark.executor.cores', '1'),
                ('spark.dynamicAllocation.maxExecutors', '32'),
                ('spark.sql.execution.arrow.pyspark.enabled', 'true'),
                ('mapreduce.fileoutputcommitter.marksuccessfuljobs', 'false'),
                ('spark.master', 'yarn'),
                ('spark.dynamicAllocation.enabled', 'true'),
                ('spark.executor.memory', '32g'),
                ('spark.driver.memory', '32g'),
                ('spark.driver.maxResultSize', '32g'),
            ]
        )

        sc = SparkContext(conf=conf)
        session = SparkSession(sc).builder.getOrCreate()

        return cls(session=session)


class State(BaseModel, arbitrary_types_allowed=True):
    credentials: Credentials
    spark: Spark
    settings: Settings

    @classmethod
    def from_env(cls) -> Self:
        settings = Settings.from_env()
        return cls(
            credentials=Credentials.from_env(),
            spark=Spark.from_env(settings),
            settings=settings,
        )
