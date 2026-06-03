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

[Confluence](https://it-portal.corp.tander.ru/pages/viewpage.action?pageId=2160460511)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Getting Started

### Installation

1. Clone the repository

```bash
git clone git@coderepo.corp.tander.ru:cvm_de/cvm-ml-platform/rnd/cvm_churn-from-dac_binary-class_churn-dac.git
```
2. Navigate to the project directory

```bash
cd cvm_churn-from-dac_binary-class_churn-dac
```
3. Initialize and update submodules

```bash
git submodule update --init --recursive
```
4. Install project dependencies with Poetry:

```bash
poetry install
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Usage

All commands use `<EVENT_TIMESTAMP>` as the logical pipeline date.

Train preprocessing:

```bash
poetry run python -m cvm_model_entrypoint train preprocess --event-timestamp <EVENT_TIMESTAMP>
```
[Notebook](https://coderepo.corp.tander.ru/cvm_de/cvm-ml-platform/rnd/cvm_churn-from-dac_binary-class_churn-dac/-/blob/feature/CVMB-24239/src/cvm_model/notebooks/train_preprocess_pipe.ipynb)

Train:

```bash
poetry run python -m cvm_model_entrypoint train run --event-timestamp <EVENT_TIMESTAMP>
```

[Notebook](https://coderepo.corp.tander.ru/cvm_de/cvm-ml-platform/rnd/cvm_churn-from-dac_binary-class_churn-dac/-/blob/feature/CVMB-24239/src/cvm_model/notebooks/train_pipe.ipynb)

Inference preprocessing:

```bash
poetry run python -m cvm_model_entrypoint inference preprocess --event-timestamp <EVENT_TIMESTAMP>
```

[Notebook](https://coderepo.corp.tander.ru/cvm_de/cvm-ml-platform/rnd/cvm_churn-from-dac_binary-class_churn-dac/-/blob/feature/CVMB-24239/src/cvm_model/notebooks/inference_preprocess_pipe.ipynb)

Inference:

```bash
poetry run python -m cvm_model_entrypoint inference run --event-timestamp <EVENT_TIMESTAMP> --train-event-timestamp <TRAIN_EVENT_TIMESTAMP>
```
[Notebook](https://coderepo.corp.tander.ru/cvm_de/cvm-ml-platform/rnd/cvm_churn-from-dac_binary-class_churn-dac/-/blob/feature/CVMB-24239/src/cvm_model/notebooks/inference_pipe.ipynb)

Date logic for `event_timestamp = 2026-06-01`:

- train audience: DAC clients from `2026-04-01`;
- train target: DAC churn in `2026-05-01`;
- train features: history before `2026-05-01`;
- inference audience: DAC clients from `2026-05-01`;
- inference features: history before `2026-06-01`;
- inference scores: probability of DAC churn in June.

## Contact

Горелова Ирина (gorelova_i_v) - gorelova_i_v@magnit.ru

[Project Link](https://coderepo.corp.tander.ru/cvm_de/cvm-ml-platform/rnd/cvm_churn-from-dac_binary-class_churn-dac)

<p align="right">(<a href="#readme-top">back to top</a>)</p>
