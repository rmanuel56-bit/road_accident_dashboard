"""Small, ordinary functions for classification, numeric prediction and explanations."""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, average_precision_score, brier_score_loss, log_loss,
    classification_report, mean_absolute_error, mean_squared_error, r2_score, mean_poisson_deviance)
from part_2_data import features, injury_records, GRADES

def temporal_split(frame):
    """Keep whole dates together; prefer the latest complete recording year as holdout."""
    frame = frame.sort_values('date').reset_index(drop=True)
    year = frame.date.dt.year
    train, test = frame.loc[year.lt(year.max())], frame.loc[year.eq(year.max())]
    if len(train) < 40 or len(test) < 10:
        dates = np.sort(frame.date.unique())
        if len(dates) < 12: raise ValueError('Need more distinct dates for chronological validation.')
        cutoff = dates[max(1, int(.8 * len(dates)))]
        train, test = frame.loc[frame.date.lt(cutoff)], frame.loc[frame.date.ge(cutoff)]
    if len(train) < 30 or len(test) < 8: raise ValueError('Too few records for a reserved-date evaluation.')
    return train.reset_index(drop=True), test.reset_index(drop=True)

def validation_folds(frame):
    folds = []
    for year in sorted(frame.date.dt.year.unique())[1:]:
        a = np.flatnonzero(frame.date.dt.year.to_numpy() < year)
        b = np.flatnonzero(frame.date.dt.year.to_numpy() == year)
        if len(a) >= 30 and len(b) >= 10: folds.append((a, b))
    if not folds:
        dates = np.sort(frame.date.unique()); cutoff = dates[int(.7 * len(dates))]
        a = np.flatnonzero(frame.date.lt(cutoff)); b = np.flatnonzero(frame.date.ge(cutoff))
        if len(a) >= 20 and len(b) >= 8: folds.append((a, b))
    if not folds: raise ValueError('Not enough earlier dates for development validation.')
    return folds[-3:]

def classifier(model, columns):
    categorical = [c for c in columns if c in ['weather', 'road_condition', 'vehicle', 'area', 'gender_group']]
    numeric = [c for c in columns if c not in categorical]
    transforms = []
    if categorical: transforms.append(('categories', OneHotEncoder(handle_unknown='ignore'), categorical))
    if numeric: transforms.append(('numeric', Pipeline([('impute', SimpleImputer(strategy='median', keep_empty_features=True)),
                                                         ('scale', StandardScaler())]), numeric))
    return Pipeline([('prepare', ColumnTransformer(transforms)), ('model', clone(model))])

def binary_metrics(y, pred, probability):
    both = len(np.unique(y)) == 2
    return {'accuracy': accuracy_score(y, pred), 'balanced_accuracy': balanced_accuracy_score(y, pred),
        'macro_F1': f1_score(y, pred, labels=[0, 1], average='macro', zero_division=0),
        'severe_precision': precision_score(y, pred, zero_division=0),
        'severe_recall': recall_score(y, pred, zero_division=0),
        'ROC_AUC': roc_auc_score(y, probability) if both else np.nan,
        'average_precision': average_precision_score(y, probability) if both else np.nan,
        'Brier_score': brier_score_loss(y, probability),
        'log_loss': log_loss(y, np.column_stack([1 - probability, probability]), labels=[0, 1])}

def count_metrics(y, prediction):
    y = np.asarray(y, float); p = np.clip(np.asarray(prediction, float), 0, None)
    return {'MAE': mean_absolute_error(y, p), 'RMSE': np.sqrt(mean_squared_error(y, p)),
        'R2': r2_score(y, p) if len(y) > 1 else np.nan,
        'WAPE_percent': 100 * np.abs(y - p).sum() / y.sum() if y.sum() else np.nan,
        'Poisson_deviance': mean_poisson_deviance(y, np.maximum(p, 1e-9))}

def classification_analysis(clean):
    eligible, coverage = injury_records(clean)
    dev, test = temporal_split(eligible)
    if dev.severe.nunique() < 2: raise ValueError('Earlier training records need both serious and non-serious injury classes.')
    folds = [(a, b) for a, b in validation_folds(dev) if dev.severe.iloc[a].nunique() == 2]
    if not folds: raise ValueError('Development folds do not contain both injury classes.')
    x, xt = features(dev), features(test); y, yt = dev.severe.astype(int), test.severe.astype(int)
    candidates = {'Prior baseline': DummyClassifier(strategy='prior'),
        'Logistic regression': LogisticRegression(C=.3, class_weight='balanced', max_iter=1500, random_state=42),
        'Random forest': RandomForestClassifier(n_estimators=80, max_depth=6, min_samples_leaf=5,
                                                class_weight='balanced_subsample', random_state=42, n_jobs=1)}
    scores = []
    for name, estimator in candidates.items():
        for i, (a, b) in enumerate(folds):
            fit = classifier(estimator, list(x)).fit(x.iloc[a], y.iloc[a])
            scores.append({'model': name, 'fold': i + 1,
                'macro_F1': f1_score(y.iloc[b], fit.predict(x.iloc[b]), labels=[0, 1], average='macro', zero_division=0)})
    cv = pd.DataFrame(scores).groupby('model').macro_F1.agg(['mean', 'std']).reset_index().sort_values('mean', ascending=False)
    name = cv.iloc[0].model
    model = classifier(candidates[name], list(x)).fit(x, y)
    pred = model.predict(xt); probability = model.predict_proba(xt)[:, 1]
    baseline = classifier(candidates['Prior baseline'], list(x)).fit(x, y)
    metrics = pd.DataFrame([{'model': name, **binary_metrics(yt, pred, probability)},
        {'model': 'Prior baseline (reference)', **binary_metrics(yt, baseline.predict(xt), baseline.predict_proba(xt)[:, 1])}])
    report_dict = classification_report(yt, pred, labels=[0, 1],
        target_names=['minor / no injury', 'major / fatal'], output_dict=True, zero_division=0)
    report = pd.DataFrame({k: v for k, v in report_dict.items() if isinstance(v, dict)}).T.reset_index(names='class')
    confusion = pd.crosstab(pd.Series(yt.to_numpy(), name='actual'), pd.Series(pred, name='predicted')).reindex(index=[0, 1], columns=[0, 1], fill_value=0)
    reference = classifier(candidates['Random forest'], list(x)).fit(x, dev.injury_grade)
    multi_dict = classification_report(test.injury_grade, reference.predict(xt), labels=GRADES,
        output_dict=True, zero_division=0)
    multi = pd.DataFrame({k: v for k, v in multi_dict.items() if isinstance(v, dict)}).T.reset_index(names='class')
    importance = permutation_importance(model, xt, yt, scoring='f1_macro', n_repeats=5, random_state=42)
    importance = pd.DataFrame({'variable': x.columns, 'macro_F1_drop': importance.importances_mean,
                                'repeat_SD': importance.importances_std}).sort_values('macro_F1_drop', ascending=False)
    groups = {'context': ['weather', 'road_condition', 'area'], 'vehicle and gender': ['vehicle', 'gender_group'],
              'context + vehicle/gender': ['weather', 'road_condition', 'area', 'vehicle', 'gender_group'], 'all variables': list(x)}
    ablations = []
    for label, cols in groups.items():
        values = []
        for a, b in folds:
            fit = classifier(candidates[name], cols).fit(x[cols].iloc[a], y.iloc[a])
            values.append(f1_score(y.iloc[b], fit.predict(x[cols].iloc[b]), labels=[0, 1], average='macro', zero_division=0))
        ablations.append({'variable_combination': label, 'development_macro_F1': np.mean(values)})
    out_of_fold = []
    for a, b in folds:
        fit = classifier(candidates[name], list(x)).fit(x.iloc[a], y.iloc[a])
        frame = dev.iloc[b][['weather', 'road_condition', 'vehicle', 'severe', 'proxy']].copy()
        frame['probability'] = fit.predict_proba(x.iloc[b])[:, 1]; out_of_fold.append(frame)
    combos = pd.concat(out_of_fold).groupby(['weather', 'road_condition', 'vehicle'], dropna=False).agg(
        records=('severe', 'size'), proxy_support=('proxy', 'nunique'), observed_fraction=('severe', 'mean'),
        mean_out_of_fold_probability=('probability', 'mean')).reset_index()
    combos = combos.loc[combos.records.ge(10)].sort_values('records', ascending=False)
    tree = classifier(DecisionTreeClassifier(max_depth=3, min_samples_leaf=20, random_state=42), list(x)).fit(x, y)
    rules = export_text(tree.named_steps['model'], feature_names=list(tree.named_steps['prepare'].get_feature_names_out()))
    predictions = test[['source_row', 'date', 'proxy', 'injury_grade']].copy()
    predictions['predicted_severe'] = pred; predictions['probability_severe'] = probability
    return {'model': model, 'selected_name': name, 'metrics': metrics, 'cv': cv, 'report': report,
            'confusion': confusion, 'multiclass_report': multi, 'importance': importance,
            'variable_combinations': pd.DataFrame(ablations), 'observed_combinations': combos,
            'predictions': predictions, 'rules': rules, 'coverage': coverage,
            'split': pd.DataFrame([{'set': 'development', 'start': dev.date.min(), 'end': dev.date.max(), 'n': len(dev)},
                                   {'set': 'holdout', 'start': test.date.min(), 'end': test.date.max(), 'n': len(test)}])}

def daily_features(counts):
    f = pd.DataFrame({'date': counts.index, 'target': counts.to_numpy()}, index=counts.index)
    for label, v, period in [('weekday', counts.index.dayofweek, 7), ('month', counts.index.month, 12)]:
        f[label + '_sin'] = np.sin(2 * np.pi * v / period); f[label + '_cos'] = np.cos(2 * np.pi * v / period)
    for lag in [1, 7, 28]: f['lag_' + str(lag)] = counts.shift(lag)
    for window in [7, 28]: f['past_mean_' + str(window)] = counts.shift(1).rolling(window).mean()
    return f.iloc[28:].reset_index(drop=True)

def regression_pipeline(estimator):
    return Pipeline([('scale', StandardScaler()), ('model', clone(estimator))])

def prediction_analysis(counts):
    data = daily_features(counts); dev, test = temporal_split(data); folds = validation_folds(dev)
    cols = [c for c in data if c not in ['date', 'target']]
    candidates = {'Mean baseline': DummyRegressor(strategy='mean'), 'Median baseline': DummyRegressor(strategy='median'),
        'Poisson regression': PoissonRegressor(alpha=1, max_iter=1000),
        'Random forest': RandomForestRegressor(n_estimators=80, max_depth=5, min_samples_leaf=15, random_state=42, n_jobs=1)}
    scores = []
    for name, estimator in candidates.items():
        for i, (a, b) in enumerate(folds):
            fit = regression_pipeline(estimator).fit(dev[cols].iloc[a], dev.target.iloc[a])
            scores.append({'model': name, 'fold': i + 1,
                           'MAE': mean_absolute_error(dev.target.iloc[b], np.clip(fit.predict(dev[cols].iloc[b]), 0, None))})
    cv = pd.DataFrame(scores).groupby('model').MAE.agg(['mean', 'std']).reset_index().sort_values('mean')
    name = cv.iloc[0].model; model = regression_pipeline(candidates[name]).fit(dev[cols], dev.target)
    learned_name = next(n for n in cv.model if 'baseline' not in n)
    learned = regression_pipeline(candidates[learned_name]).fit(dev[cols], dev.target)
    selected_pred = np.clip(model.predict(test[cols]), 0, None)
    predictions = test[['date', 'target']].rename(columns={'target': 'actual'}).copy()
    predictions['predicted'] = selected_pred
    rows = [{'model': name, **count_metrics(test.target, selected_pred)},
            {'model': learned_name + ' (learned reference)', **count_metrics(test.target, learned.predict(test[cols]))}]
    mean = regression_pipeline(candidates['Mean baseline']).fit(dev[cols], dev.target)
    rows += [{'model': 'Mean baseline (reference)', **count_metrics(test.target, mean.predict(test[cols]))},
             {'model': 'Previous-week reference', **count_metrics(test.target, test.lag_7)}]
    importance = permutation_importance(learned, test[cols], test.target, scoring='neg_mean_absolute_error', n_repeats=5, random_state=42)
    table = pd.DataFrame({'variable': cols, 'MAE_increase': importance.importances_mean,
                          'repeat_SD': importance.importances_std, 'inspected_model': learned_name}).sort_values('MAE_increase', ascending=False)
    groups = {'calendar': [c for c in cols if c.startswith(('weekday', 'month'))],
              'past counts': [c for c in cols if c.startswith(('lag', 'past'))], 'all variables': cols}
    ablations = []
    for label, subset in groups.items():
        errors = []
        for a, b in folds:
            fit = regression_pipeline(candidates[learned_name]).fit(dev[subset].iloc[a], dev.target.iloc[a])
            errors.append(mean_absolute_error(dev.target.iloc[b], fit.predict(dev[subset].iloc[b])))
        ablations.append({'variable_combination': label, 'development_MAE': np.mean(errors), 'model': learned_name})
    next_date = counts.index[-1] + pd.Timedelta(days=1)
    extended = pd.concat([counts, pd.Series([np.nan], index=[next_date])])
    next_x = daily_features(extended).iloc[[-1]][cols]
    deployed = regression_pipeline(candidates[name]).fit(data[cols], data.target)
    next_value = np.clip(deployed.predict(next_x), 0, None)[0]
    return {'model': model, 'selected_name': name, 'metrics': pd.DataFrame(rows), 'cv': cv,
            'predictions': predictions, 'importance': table, 'variable_combinations': pd.DataFrame(ablations),
            'next_day': pd.DataFrame({'date': [next_date], 'predicted_records': [next_value]}),
            'split': pd.DataFrame([{'set': 'development', 'start': dev.date.min(), 'end': dev.date.max()},
                                   {'set': 'holdout', 'start': test.date.min(), 'end': test.date.max()}])}
