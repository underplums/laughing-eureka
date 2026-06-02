<a id="readme-top"></a>

<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#about-the-project">About The Project</a></li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#data-storage">Data Storage</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>

## About The Project

`cvm_churn-from-dac_binary-class_churn-dac` is a binary classification project for predicting DAC churn.

The model scores clients who were DAC in the base month and estimates the probability that they will stop being DAC in the next month.

DAC is defined as a client with transaction activity and at least one digital activity signal in the same month:

```sql
has_transaction_activity = 1
and (
    has_mobapp_activity = 1
    or has_pwa_activity = 1
    or vcoff_trn_cnt > 0
)
```

Target:

```text
target_churn_from_dac = 1, if the client was DAC in base_month and is not DAC in target_month.
target_churn_from_dac = 0, if the client was DAC in base_month and remains DAC in target_month.
```

Model type: CatBoost binary classifier with isotonic score calibration.

Detailed project description: add Confluence link after business approval.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Getting Started

### Prerequisites

The project is intended to run on the stream Dataproc/JupyterHub environment with access to GP, S3, Spark and MLflow credentials.

Required environment variables are loaded by `cvm_model.io.State.from_env()`:

- `GP_LOYALTY_USER`
- `GP_LOYALTY_PASSWORD`
- `GP_LOYALTY_HOST`
- `GP_LOYALTY_PORT`
- `GP_LOYALTY_DB`
- `CH_CVM_USER`
- `CH_CVM_PASSWORD`
- `CH_CVM_HOST`
- `CH_CVM_PORT`
- `CH_CVM_DB`
- `AWS_ENDPOINT_URL`
- `AWS_DEFAULT_REGION`
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `PXF_SERVER_PROFILE`
- `MLFLOW_TRACKING_USERNAME`
- `MLFLOW_TRACKING_PASSWORD`
- `MLFLOW_TRACKING_URI`
- `MLFLOW_TRACKING_INSECURE_TLS`
- `MLFLOW_TRACKING_SERVER_CERT_PATH` or `MLFLOW_TRACKING_SERVER_CER_PATH`
- `S3_PERMANENT_STORAGE_PREFIX`
- `S3_TEMPORARY_STORAGE_PREFIX`
- `FEATURE_STORE_S3_ROOT`

### Installation

Install project dependencies with Poetry:

```bash
poetry install
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Usage

All commands use `event_timestamp` as the logical pipeline date.

Train preprocessing:

```bash
poetry run python -m cvm_model_entrypoint train preprocess --event-timestamp 2026-06-01
```

Train:

```bash
poetry run python -m cvm_model_entrypoint train run --event-timestamp 2026-06-01
```

Inference preprocessing:

```bash
poetry run python -m cvm_model_entrypoint inference preprocess --event-timestamp 2026-06-01
```

Inference:

```bash
poetry run python -m cvm_model_entrypoint inference run --event-timestamp 2026-06-01 --train-event-timestamp 2026-06-01
```

Date logic for `event_timestamp = 2026-06-01`:

- train audience: DAC clients from `2026-04-01`;
- train target: churn from DAC in `2026-05-01`;
- train features: history before `2026-05-01`;
- inference audience: DAC clients from `2026-05-01`;
- inference features: history before `2026-06-01`;
- inference scores: probability of DAC churn in June.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Data Storage

Train and holdout datasets are saved to temporary S3 storage:

```text
S3_TEMPORARY_STORAGE_PREFIX/<event_timestamp>/input/train/
S3_TEMPORARY_STORAGE_PREFIX/<event_timestamp>/input/holdout/
```

Inference dataset is saved to temporary S3 storage:

```text
S3_TEMPORARY_STORAGE_PREFIX/<event_timestamp>/input/inference/
```

Model predictions are saved to permanent S3 storage:

```text
S3_PERMANENT_STORAGE_PREFIX/model_predictions/<event_timestamp>/
```

Train and inference data statistics are saved to permanent S3 storage:

```text
S3_PERMANENT_STORAGE_PREFIX/data_stat/train/<event_timestamp>/
S3_PERMANENT_STORAGE_PREFIX/data_stat/inference/<event_timestamp>/
```

The model is logged and registered in MLflow with the name:

```text
cvm_churn-from-dac_binary-class_churn-dac_catboost
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Contact

Owner: gorelova_i_v

Project repository: `cvm_churn-from-dac_binary-class_churn-dac`

<p align="right">(<a href="#readme-top">back to top</a>)</p>
