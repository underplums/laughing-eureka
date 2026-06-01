import os
import numpy as np
import pandas as pd

from typing import List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import shap

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    roc_curve, 
    precision_recall_curve, 
    confusion_matrix,
    PrecisionRecallDisplay,
    ConfusionMatrixDisplay
)
from sklearn.calibration import calibration_curve

from scipy.stats import skew, kurtosis

def get_data_report(df: pd.DataFrame, features: List[str], source_col: Union[str, None] = None) -> pd.DataFrame:
    report = pd.DataFrame()
    if source_col:
        report['source'] = list(df[source_col].unique()) * len(features)
    report['feature_name'] = features
    
    for row in report.iterrows():
        f = row[1]['feature_name']
        fs = df[f]
        
        report.loc[row[0], 'is_categorical'] = False
        report.loc[row[0], 'null_share'] = fs.isna().sum() / len(df)
        report.loc[row[0], 'zero_share'] = (fs == 0).sum() / len(df)
        
        fs = fs[fs > 0]
        
        if len(fs) > 0:
            mean = fs.mean()
            report.loc[row[0], 'mean'] = fs.mean()

            report.loc[row[0], 'min'] = fs.min()
            report.loc[row[0], 'max'] = fs.max()
            report.loc[row[0], 'mode'] = fs.mode()[0]

            median = fs.quantile(0.5)
            report.loc[row[0], 'median'] = median

            for k in [10, 25, 75, 90, 95, 99]:
                quantile = fs.quantile(k / 100)
                report.loc[row[0], f'{k}_perc'] = quantile
                report.loc[row[0], f'{k}_perc_to_median'] = quantile / median

            report.loc[row[0], 'std_to_mean'] = fs.std() / mean
            
            if fs.nunique(dropna=True) > 1:
                report.loc[row[0], 'skew'] = skew(fs)
                report.loc[row[0], 'kurt'] = kurtosis(fs)
            else:
                report.loc[row[0], 'skew'] = 0
                report.loc[row[0], 'kurt'] = 0
    return report

def plot_precision_recall_curve(
    y_val, 
    predicts, 
    title_suffix: str = "", 
    artifacts_dir: Union[None, str] = None,
):
    precision, recall, _ = precision_recall_curve(y_val, predicts)
    
    prauc = average_precision_score(y_val, predicts)
    target_mean = y_val.mean()
    net_gain = prauc - target_mean
    net_gain_pct = round(prauc / target_mean, 1)

    legend_label = (
        f"PRAUC: {prauc:.2f}\n"
        f"Base: {target_mean:.2f}\n"
        f"Net Gain: {net_gain:.2f} ({net_gain_pct}x)"
    )

    display = PrecisionRecallDisplay(precision=precision, recall=recall)
    display.plot(label=legend_label)
    
    title = f"Precision-Recall Curve {title_suffix}".strip()
    plt.title(title)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc='upper right')
    
    if artifacts_dir is not None:
        plt.savefig(os.path.join(artifacts_dir, title), bbox_inches='tight')


def plot_roc(
    y_val, 
    predicts, 
    title_suffix: str = "", 
    artifacts_dir: Union[None, str] = None,
):

    score_auc = roc_auc_score(y_val, predicts)

    fpr, tpr, _ = roc_curve(y_val, predicts)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {score_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Специфичность)')
    plt.ylabel('True Positive Rate (Чувствительность)')
    title = f"Receiver Operating Characteristic (ROC) {title_suffix}".strip()
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    
    if artifacts_dir is not None:
        plt.savefig(os.path.join(artifacts_dir, title), bbox_inches='tight')

def plot_confusion_matrix(
    y_true, 
    y_pred, 
    artifacts_dir: Union[None, str] = None,
):
    ConfusionMatrixDisplay(
            confusion_matrix=confusion_matrix(y_true, y_pred),
            display_labels=['not churn', 'churn'],
        ).plot(cmap=plt.cm.Blues, values_format='d')
    
    title = 'Confusion Matrix'
    plt.title(title)

    if artifacts_dir is not None:
        plt.savefig(os.path.join(artifacts_dir, title), bbox_inches='tight')


def get_feature_importance(model, features: List[str]) -> pd.DataFrame:
    importance = pd.DataFrame(
        {
            'feature': features,
            'importance': model.get_feature_importance(),
        }
    ).sort_values('importance', ascending=False)

    return importance


def plot_feature_importance(
    model,
    features: List[str],
    top_n: int = 40,
    artifacts_dir: Union[None, str] = None,
) -> pd.DataFrame:
    importance = get_feature_importance(model, features)

    plt.figure(figsize=(10, max(6, top_n * 0.25)))
    top_importance = importance.head(top_n).sort_values('importance')
    plt.barh(top_importance['feature'], top_importance['importance'])
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    title = f'Top {top_n} Feature Importances'
    plt.title(title)
    plt.tight_layout()

    if artifacts_dir is not None:
        plt.savefig(os.path.join(artifacts_dir, 'feature_importance.png'), bbox_inches='tight')

    return importance


def plot_shap_summary(
    model,
    df: pd.DataFrame,
    features: List[str],
    sample_size: int = 50_000,
    artifacts_dir: Union[None, str] = None,
):
    sample = df[features].sample(min(sample_size, len(df)), random_state=42)
    shap.summary_plot(shap.TreeExplainer(model).shap_values(sample), sample, show=False)
    plt.title('SHAP Summary Plot')

    if artifacts_dir is not None:
        plt.savefig(os.path.join(artifacts_dir, 'shap_summary_plot.png'), bbox_inches='tight')

    plt.close()


def _get_top_two_features(
    model,
    features: List[str],
    feature_importance: Optional[pd.DataFrame] = None,
) -> Tuple[str, str]:
    if feature_importance is None:
        feature_importance = get_feature_importance(model, features)

    top_features = feature_importance['feature'].head(2).tolist()
    if len(top_features) < 2:
        raise ValueError('At least two features are required for decision scatter plot')

    return top_features[0], top_features[1]


def plot_decision_scatter(
    model,
    df: pd.DataFrame,
    features: List[str],
    target_col: str,
    score_col: Optional[str] = None,
    feature_importance: Optional[pd.DataFrame] = None,
    threshold: Optional[float] = None,
    sample_size: int = 50_000,
    artifacts_dir: Union[None, str] = None,
):
    x_feature, y_feature = _get_top_two_features(model, features, feature_importance)
    sample = df.sample(min(sample_size, len(df)), random_state=42).copy()

    if score_col is not None and score_col in sample.columns:
        sample_score = sample[score_col]
    else:
        sample_score = model.predict_proba(sample[features])[:, 1]

    plt.figure(figsize=(9, 7))
    scatter = plt.scatter(
        sample[x_feature],
        sample[y_feature],
        c=sample_score,
        s=8,
        alpha=0.35,
        cmap='viridis',
    )
    plt.colorbar(scatter, label='Score')
    plt.xlabel(x_feature)
    plt.ylabel(y_feature)
    title = f'Decision Scatter: {x_feature} vs {y_feature}'
    plt.title(title)

    if threshold is not None:
        selected_share = (sample_score >= threshold).mean()
        plt.text(
            0.02,
            0.98,
            f'threshold={threshold:.3f}\nselected={selected_share:.2%}',
            transform=plt.gca().transAxes,
            va='top',
            bbox={'facecolor': 'white', 'alpha': 0.8, 'edgecolor': 'none'},
        )

    if artifacts_dir is not None:
        plt.savefig(os.path.join(artifacts_dir, 'decision_scatter.png'), bbox_inches='tight')


def plot_calibration_curve(
    y_true,
    y_score,
    n_bins: int = 10,
    artifacts_dir: Union[None, str] = None,
):
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true,
        y_score,
        n_bins=n_bins,
        strategy='quantile',
    )

    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect calibration')
    plt.plot(mean_predicted_value, fraction_of_positives, marker='o', label='Model')
    plt.xlabel('Mean predicted probability')
    plt.ylabel('Fraction of positives')
    title = 'Calibration Curve'
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)

    if artifacts_dir is not None:
        plt.savefig(os.path.join(artifacts_dir, 'calibration_curve.png'), bbox_inches='tight')


def plot_target_distribution(
    df: pd.DataFrame,
    target_col: str,
    artifacts_dir: Union[None, str] = None,
):
    counts = df[target_col].value_counts(dropna=False).sort_index()
    shares = counts / counts.sum()

    plt.figure(figsize=(7, 5))
    bars = plt.bar(counts.index.astype(str), counts.values)
    plt.xlabel(target_col)
    plt.ylabel('Rows')
    title = 'Target Distribution'
    plt.title(title)

    for bar, share in zip(bars, shares):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f'{share:.1%}',
            ha='center',
            va='bottom',
        )

    if artifacts_dir is not None:
        plt.savefig(os.path.join(artifacts_dir, 'target_distribution.png'), bbox_inches='tight')


def _get_dac_segment(df: pd.DataFrame, segment_col: Optional[str] = None) -> pd.Series:
    if segment_col is not None and segment_col in df.columns:
        return df[segment_col].astype(str)

    if 'dac_months_last_12' in df.columns:
        return pd.cut(
            df['dac_months_last_12'].fillna(0),
            bins=[-1, 1, 5, 9, 12],
            labels=['new_1m', 'unstable_2_5m', 'regular_6_9m', 'stable_10_12m'],
        ).astype(str)

    segment_parts = []
    for col, label in [
        ('is_new_dac', 'new'),
        ('is_unstable_dac', 'unstable'),
        ('is_regular_dac', 'regular'),
        ('is_stable_dac', 'stable'),
    ]:
        if col in df.columns:
            segment_parts.append((df[col] == 1, label))

    if segment_parts:
        result = pd.Series('other', index=df.index)
        for mask, label in segment_parts:
            result.loc[mask] = label
        return result

    raise ValueError('DAC segment column or DAC segment features were not found')


def plot_dac_segment_distribution(
    df: pd.DataFrame,
    target_col: str,
    segment_col: Optional[str] = None,
    artifacts_dir: Union[None, str] = None,
) -> pd.DataFrame:
    segment = _get_dac_segment(df, segment_col)
    report = (
        pd.DataFrame({'segment': segment, target_col: df[target_col]})
        .groupby('segment')
        .agg(rows=(target_col, 'size'), target_rate=(target_col, 'mean'))
        .reset_index()
        .sort_values('rows', ascending=False)
    )

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.bar(report['segment'], report['rows'], alpha=0.7)
    ax1.set_ylabel('Rows')
    ax1.set_xlabel('DAC segment')
    ax1.tick_params(axis='x', rotation=30)

    ax2 = ax1.twinx()
    ax2.plot(report['segment'], report['target_rate'], color='red', marker='o')
    ax2.set_ylabel('Target rate')

    title = 'DAC Segment Distribution'
    plt.title(title)
    fig.tight_layout()

    if artifacts_dir is not None:
        plt.savefig(os.path.join(artifacts_dir, 'dac_segment_distribution.png'), bbox_inches='tight')

    return report


def plot_score_distribution(
    df: pd.DataFrame,
    target_col: str,
    score_col: str,
    artifacts_dir: Union[None, str] = None,
):
    plt.figure(figsize=(9, 6))
    for target_value in sorted(df[target_col].dropna().unique()):
        plt.hist(
            df.loc[df[target_col] == target_value, score_col],
            bins=50,
            alpha=0.45,
            density=True,
            label=f'{target_col}={target_value}',
        )

    plt.xlabel(score_col)
    plt.ylabel('Density')
    title = 'Score Distribution by Target'
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)

    if artifacts_dir is not None:
        plt.savefig(os.path.join(artifacts_dir, 'score_distribution_by_target.png'), bbox_inches='tight')


def plot_decile_report(
    df: pd.DataFrame,
    target_col: str,
    score_col: str,
    artifacts_dir: Union[None, str] = None,
) -> pd.DataFrame:
    report_df = df[[target_col, score_col]].copy()
    report_df['score_decile'] = pd.qcut(
        report_df[score_col].rank(method='first'),
        10,
        labels=False,
    ) + 1

    report = (
        report_df
        .groupby('score_decile')
        .agg(
            rows=(target_col, 'size'),
            target_rate=(target_col, 'mean'),
            score_min=(score_col, 'min'),
            score_max=(score_col, 'max'),
        )
        .reset_index()
        .sort_values('score_decile')
    )

    plt.figure(figsize=(9, 6))
    plt.plot(report['score_decile'], report['target_rate'], marker='o')
    plt.xlabel('Score decile')
    plt.ylabel('Target rate')
    title = 'Target Rate by Score Decile'
    plt.title(title)
    plt.grid(alpha=0.3)

    if artifacts_dir is not None:
        plt.savefig(os.path.join(artifacts_dir, 'target_rate_by_score_decile.png'), bbox_inches='tight')

    return report


def plot_model_diagnostics(
    model,
    base_model,
    df: pd.DataFrame,
    features: List[str],
    target_col: str,
    score_col: str,
    threshold: Optional[float] = None,
    dac_segment_col: Optional[str] = None,
    artifacts_dir: Union[None, str] = None,
):
    importance = plot_feature_importance(base_model, features, artifacts_dir=artifacts_dir)
    plot_shap_summary(base_model, df, features, artifacts_dir=artifacts_dir)
    plot_decision_scatter(
        model,
        df,
        features,
        target_col,
        score_col=score_col,
        feature_importance=importance,
        threshold=threshold,
        artifacts_dir=artifacts_dir,
    )
    plot_calibration_curve(df[target_col], df[score_col], artifacts_dir=artifacts_dir)
    plot_target_distribution(df, target_col, artifacts_dir=artifacts_dir)
    segment_report = plot_dac_segment_distribution(
        df,
        target_col,
        segment_col=dac_segment_col,
        artifacts_dir=artifacts_dir,
    )
    plot_score_distribution(df, target_col, score_col, artifacts_dir=artifacts_dir)
    decile_report = plot_decile_report(df, target_col, score_col, artifacts_dir=artifacts_dir)

    return {
        'feature_importance': importance,
        'segment_report': segment_report,
        'decile_report': decile_report,
    }
