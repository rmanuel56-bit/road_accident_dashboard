"""Monthly forecasting and simple, in-memory downloads."""
from io import BytesIO
import json
import math
import pickle
import zipfile
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from part_3_models import count_metrics

def forecast_values(series, horizon, method):
    y = series.to_numpy(dtype=float)
    if method == 'Last value': prediction = np.repeat(y[-1], horizon)
    elif method == 'Seasonal naive': prediction = np.array([y[-12 + i % 12] for i in range(horizon)])
    elif method.startswith('Smoothing'):
        alpha = float(method.split()[-1]); level = y[0]
        for value in y[1:]: level = alpha * value + (1 - alpha) * level
        prediction = np.repeat(level, horizon)
    elif method == 'Trend regression':
        future_dates = pd.date_range(series.index[-1] + pd.offsets.MonthBegin(1), periods=horizon, freq='MS')
        def design(dates, start):
            return np.column_stack([np.arange(start, start + len(dates)),
                np.sin(2 * np.pi * dates.month / 12), np.cos(2 * np.pi * dates.month / 12)])
        model = make_pipeline(StandardScaler(), Ridge(alpha=10))
        model.fit(design(series.index, 0), y)
        prediction = model.predict(design(future_dates, len(y)))
    else: raise ValueError('Unknown forecasting method.')
    return np.clip(prediction, 0, None)

def forecasting_analysis(daily_counts, horizon=12):
    monthly = daily_counts.resample('MS').sum()
    if len(monthly) < 30: raise ValueError('Forecast comparison needs at least 30 months of reporting history.')
    test_size = 12 if len(monthly) >= 36 else 6
    dev, test = monthly.iloc[:-test_size], monthly.iloc[-test_size:]
    methods = ['Last value', 'Seasonal naive', 'Smoothing 0.2', 'Smoothing 0.5', 'Trend regression']
    scores, errors = [], []
    for cut in sorted(set([len(dev) - 2 * test_size, len(dev) - test_size])):
        if cut < 12: continue
        train, validation = dev.iloc[:cut], dev.iloc[cut:cut + test_size]
        for method in methods:
            pred = forecast_values(train, len(validation), method)
            scores.append({'model': method, 'training_months': cut, **count_metrics(validation, pred)})
            errors.extend({'model': method, 'error': float(a - b)} for a, b in zip(validation, pred))
    if not scores: raise ValueError('Not enough earlier months for forecast validation.')
    scores = pd.DataFrame(scores)
    cv = scores.groupby('model').MAE.agg(['mean', 'std']).reset_index().sort_values('mean')
    selected = cv.iloc[0].model
    pred = forecast_values(dev, len(test), selected)
    rows = []
    scale = np.mean(np.abs(dev.to_numpy()[12:] - dev.to_numpy()[:-12]))
    for label, method in [(selected, selected), ('Seasonal naive (reference)', 'Seasonal naive'), ('Last value (reference)', 'Last value')]:
        values = forecast_values(dev, len(test), method)
        row = {'model': label, **count_metrics(test, values)}
        row['seasonal_MASE'] = row['MAE'] / scale if scale > 0 else np.nan; rows.append(row)
    calibration = np.array([abs(e['error']) for e in errors if e['model'] == selected])
    quantile = min(1, math.ceil((len(calibration) + 1) * .9) / len(calibration))
    radius = float(np.quantile(calibration, quantile, method='higher'))
    holdout = pd.DataFrame({'date': test.index, 'actual': test.to_numpy(), 'predicted': pred,
                           'approx_90pct_low': np.maximum(0, pred - radius), 'approx_90pct_high': pred + radius})
    future_dates = pd.date_range(monthly.index[-1] + pd.offsets.MonthBegin(1), periods=horizon, freq='MS')
    future_pred = forecast_values(monthly, horizon, selected)
    future = pd.DataFrame({'date': future_dates, 'predicted': future_pred,
                          'approx_90pct_low': np.maximum(0, future_pred - radius), 'approx_90pct_high': future_pred + radius})
    coverage = np.mean((holdout.actual >= holdout.approx_90pct_low) & (holdout.actual <= holdout.approx_90pct_high))
    return {'selected_name': selected, 'metrics': pd.DataFrame(rows), 'cv': cv, 'backtests': scores,
            'holdout': holdout, 'future': future, 'monthly': monthly.reset_index(),
            'interval_coverage': float(coverage), 'calibration_errors': len(calibration)}

def download_package(prepared, results, source_name):
    """Export what has actually been run, without temporary folders or closure serialization."""
    buffer = BytesIO()
    notes = ['RACC analysis results', 'Uploaded source: ' + source_name,
             'Only analyses already run in the app are included.',
             'Counts are reported rows, not verified unique incidents or exposure-adjusted rates.',
             'Context-dependent date fixes and reporting-calendar completeness require confirmation.',
             'Consciousness labels are not injury grades. Explanations are not causal effects.',
             'Evaluation models expect their documented cleaned features. Load pickle files only from trusted sources.']
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, value in prepared.items():
            if isinstance(value, pd.DataFrame): archive.writestr('data/' + name + '.csv', value.to_csv(index=False))
        for task, result in results.items():
            for name, value in result.items():
                if isinstance(value, pd.DataFrame): archive.writestr(task + '/' + name + '.csv', value.to_csv(index=False))
                elif name == 'rules': archive.writestr(task + '/reference_tree_rules.txt', value)
                elif name == 'model': archive.writestr(task + '/evaluation_model.pkl', pickle.dumps(value))
        summary = {name: result.get('selected_name', 'descriptive analysis') for name, result in results.items()}
        archive.writestr('completed_analyses.json', json.dumps(summary, indent=2))
        archive.writestr('README.txt', '\n'.join(notes))
    return buffer.getvalue()
