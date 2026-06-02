from datetime import datetime

import logging
import os

import mlflow
import pandas as pd
from mlflow import MlflowClient

import cvm_model.functions as func
import cvm_model.utils as utils
from cvm_model.io import State
from cvm_model.parameters import (
    features,
    inference_data_stat_suffix,
    input_suffix,
    model_predictions_suffix,
    model_type,
    project_name,
    score,
    target,
    template,
)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
pd.set_option('display.max_columns', None)


def inference(event_timestamp: datetime, train_event_timestamp: datetime):
    logging.info('Start inference')

    # Set connections
    state = State.from_env()
    session = state.spark.session
    os.environ.update(state.credentials.mlflow.environment)
    if state.credentials.mlflow.tracking_uri:
        mlflow.set_tracking_uri(state.credentials.mlflow.tracking_uri)

    client = MlflowClient()
    event_date = event_timestamp.date().isoformat()
    train_event_date = train_event_timestamp.date().isoformat()

    # Resolve inference input path
    input_prefix = state.settings.preprocess_prefix(event_timestamp) / input_suffix / 'inference'
    input_bucket = input_prefix.split('//')[1].split('/')[0]
    input_prefix = '/'.join(input_prefix.split('//')[1].split('/')[1:])
    input_path = template.format(bucket=input_bucket, prefix=input_prefix)

    # Load inference dataset prepared by preprocess_inference.py
    logging.info(f'Loading inference dataset from {input_path}')
    df = session.read.parquet(input_path).toPandas()

    assert len(df) > 0, 'Inference dataset is empty'
    assert df['contact_id'].nunique() == len(df), 'Inference dataset contains duplicate contact_id values'

    missing_features = set(features) - set(df.columns)
    assert not missing_features, f'Inference dataset is missing features: {missing_features}'

    # Load the latest registered model version for this project.
    model_name = f'{project_name}_{model_type}'
    versions = client.search_model_versions(
        f'name = \'{model_name}\'',
        order_by=['version_number DESC'],
    )
    assert len(versions) > 0, f'Registered model was not found: {model_name}'

    model_version = versions[0]
    logging.info(f'Loading model {model_version.name} version {model_version.version}')
    model = mlflow.sklearn.load_model(f'models:/{model_version.name}/{model_version.version}')

    # Score all current DAC clients. The saved model is already calibrated.
    logging.info('Scoring inference dataset')
    df[score] = model.predict_proba(df[features])[:, 1]

    # Save predictions to permanent S3 storage.
    predictions_prefix = state.settings.get_prefix(temp=False, suffix=model_predictions_suffix)
    predictions_bucket = predictions_prefix.split('//')[1].split('/')[0]
    predictions_prefix = '/'.join(predictions_prefix.split('//')[1].split('/')[1:])
    predictions_prefix = f'{predictions_prefix}/{event_date}/'

    score_df = pd.DataFrame(
        {
            'model_name': model_version.name,
            'target': target,
            'model_type': model_type,
            'version_id': model_version.version,
            'contact_id': df['contact_id'].astype('int64'),
            score: df[score],
            'dac_segment_12m': df['dac_segment_12m'] if 'dac_segment_12m' in df.columns else None,
            'segment': df['segment'] if 'segment' in df.columns else None,
            'model_inference_date': event_date,
            'model_train_date': model_version.tags.get('event_timestamp', train_event_date),
        }
    )

    logging.info(f'Saving predictions to s3://{predictions_bucket}/{predictions_prefix}')
    utils.save_df_to_s3(score_df, state.credentials.cvm_s3, predictions_bucket, predictions_prefix)

    # Save inference data and score statistics to permanent S3 storage.
    data_stat_prefix = state.settings.get_prefix(temp=False, suffix=inference_data_stat_suffix)
    data_stat_bucket = data_stat_prefix.split('//')[1].split('/')[0]
    data_stat_prefix = '/'.join(data_stat_prefix.split('//')[1].split('/')[1:])
    data_stat_prefix = f'{data_stat_prefix}/{event_date}/'

    logging.info(f'Saving inference data report to s3://{data_stat_bucket}/{data_stat_prefix}')
    report = func.get_data_report(df, features + [score])
    utils.save_df_to_s3(report, state.credentials.cvm_s3, data_stat_bucket, data_stat_prefix)

    logging.info('End inference')
