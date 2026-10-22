"""Read, clean and describe RACC CSV data. No Streamlit or dynamic loading here."""
from io import BytesIO
from datetime import datetime
import re
import numpy as np
import pandas as pd
from scipy import stats

REQUIRED = ['DATE', 'TIME', 'LOCATION', 'WEATHER', 'ROAD CONDITION',
            'VEHICLE TYPE', 'CAUSE', 'SEVERITY', 'GENDER']
GRADES = ['no injury', 'minor injury', 'major injury', 'fatal']
# Context-dependent proposals for the supplied RACC file; retained in the audit.
DATE_FIXES = {'8/18/0201': '2021-08-18', '1/4/202': '2022-01-04',
              '1/14/202': '2022-01-14', '1/16/202': '2022-01-16',
              '11/2/1014': '2024-11-02', '8/3/1025': '2025-08-03',
              '8/10/1025': '2025-08-10'}

def parse_date(value, apply_fixes):
    if pd.isna(value): return pd.NaT, 'missing'
    value = str(value).strip()
    if value in DATE_FIXES:
        return (pd.Timestamp(DATE_FIXES[value]), 'provisional correction') if apply_fixes else (pd.NaT, 'quarantined')
    try:
        a, b, year = map(int, re.sub('/+', '/', value).split('/'))
        if not 2000 <= year <= 2100: raise ValueError('year')
        month, day = (a, b) if b > 12 and a <= 12 else (b, a)
        return pd.Timestamp(year=year, month=month, day=day), 'parsed'
    except (ValueError, TypeError): return pd.NaT, 'invalid'

def parse_hour(value):
    if pd.isna(value): return np.nan
    for fmt in ['%I:%M %p', '%I:%M:%S %p', '%I:%M%p', '%H:%M', '%H:%M:%S']:
        try:
            t = datetime.strptime(str(value).strip().upper(), fmt)
            return t.hour + t.minute / 60
        except ValueError: pass
    return np.nan

def gender_group(value):
    if pd.isna(value): return 'unknown'
    if 'other' in value: return 'partly unspecified'
    sexes = set(re.findall(r'\b(?:female|male)\b', value))
    if len(sexes) == 2: return 'mixed'
    if sexes == {'male'}: return 'male only'
    if sexes == {'female'}: return 'female only'
    return 'unknown'

def area_group(value):
    if pd.isna(value): return 'unknown'
    value = value.replace('bunawna', 'bunawan').replace('san trodoro', 'san teodoro')
    value = value.replace('pobalcion', 'poblacion').replace('mamabalili', 'mambalili')
    for area in ['san teodoro', 'san andres', 'san marcos', 'bunawan brook',
                 'consuelo', 'libertad', 'poblacion', 'mambalili', 'imelda', 'nueva era']:
        if area in value: return area
    if any(s in value for s in ['bun. brook', 'bun, brook', 'b-brook']): return 'bunawan brook'
    if re.search(r'\bpob\.?\b', value): return 'poblacion'
    return 'other / unresolved'

def prepare_data(content, apply_fixes=True, encoding='utf-8-sig', separator=','):
    raw = pd.read_csv(BytesIO(content), dtype='string', encoding=encoding, sep=separator)
    raw.columns = raw.columns.str.strip().str.upper()
    missing = [c for c in REQUIRED if c not in raw.columns]
    if missing: raise ValueError('Missing required columns: ' + ', '.join(missing))
    if raw.columns.duplicated().any(): raise ValueError('Duplicate column names. Rename them in the CSV.')
    raw = raw[REQUIRED].copy()
    if raw.empty: raise ValueError('The CSV has no data rows.')
    d = raw.copy()
    for c in REQUIRED:
        d[c] = d[c].str.strip().str.lower().str.replace(r'\s+', ' ', regex=True).replace('', pd.NA)
    d.columns = [c.lower().replace(' ', '_') for c in d.columns]
    dates = [parse_date(v, apply_fixes) for v in raw.DATE]
    d['date'] = pd.to_datetime([x[0] for x in dates])
    d['date_status'] = [x[1] for x in dates]
    d['hour'] = d['time'].map(parse_hour)
    d['source_row'] = np.arange(len(d)) + 2
    d['duplicate_flag'] = raw.duplicated(keep=False)
    d['severity'] = d['severity'].replace({'minor injry': 'minor injury'})
    d['injury_grade'] = d.severity.where(d.severity.isin(GRADES))
    d['severe'] = d.injury_grade.map({'no injury': 0, 'minor injury': 0, 'major injury': 1, 'fatal': 1})
    d['road_condition'] = d.road_condition.replace({'clear': 'clear (unverified)'})
    d['vehicle'] = d.vehicle_type.replace({'single motor': 'motorcycle', 'motor single': 'motorcycle', 'motorcyle': 'motorcycle'})
    multiple = d.vehicle.fillna('').str.contains(r'\bvs\.?|&|\band\b|^2\b', regex=True)
    d.loc[multiple, 'vehicle'] = 'multiple vehicles recorded'
    d['area'] = d.location.map(area_group)
    d['gender_group'] = d.gender.map(gender_group)
    d['year'] = d.date.dt.year
    location = d.location.fillna('unknown').str.replace(r'\bp\s*-?\s*(?=\d)', 'p', regex=True)
    d['proxy'] = 'row_' + d.source_row.astype(str)
    complete = d.date.notna() & d.hour.notna() & d.location.notna()
    d.loc[complete, 'proxy'] = (d.loc[complete, 'date'].dt.strftime('%Y-%m-%d') + '|' +
        d.loc[complete, 'hour'].round(5).astype(str) + '|' + location[complete])
    audit = pd.DataFrame({'variable': REQUIRED, 'missing': raw.isna().sum().values,
                          'unique_values': raw.nunique().values})
    review = pd.DataFrame({'source_row': d.source_row, 'original_date': raw.DATE,
                           'cleaned_date': d.date, 'status': d.date_status})
    review = review.loc[review.status.ne('parsed')]
    return {'raw': raw, 'clean': d, 'audit': audit, 'date_review': review,
            'exact_repeats': int(raw.duplicated().sum())}

def injury_records(d):
    """Exclude recording years dominated by non-injury status labels."""
    coverage = d.loc[d.date.notna()].groupby('year').injury_grade.agg(
        records='size', usable_grades='count')
    coverage['usable_fraction'] = coverage.usable_grades / coverage.records
    years = coverage.index[coverage.usable_fraction.ge(.5)]
    return d.loc[d.year.isin(years) & d.injury_grade.notna()].copy(), coverage.reset_index()

def make_counts(d):
    valid = d.loc[d.date.notna()]
    if valid.empty: raise ValueError('No usable dates for count modeling.')
    dates = pd.date_range(valid.date.min().to_period('M').start_time,
                          valid.date.max().to_period('M').end_time.normalize(), freq='D')
    counts = valid.groupby('date').size().reindex(dates, fill_value=0).astype(float)
    counts.index.name = 'date'
    return counts.rename('reported_records')

def features(d):
    x = d[['weather', 'road_condition', 'vehicle', 'area', 'gender_group']].fillna('unknown').astype(str).copy()
    for name, values, period in [('hour', d.hour, 24), ('month', d.date.dt.month, 12), ('weekday', d.date.dt.dayofweek, 7)]:
        x[name + '_sin'] = np.sin(2 * np.pi * values / period)
        x[name + '_cos'] = np.cos(2 * np.pi * values / period)
    return x

def statistical_analysis(d, permutations=499):
    frequencies = pd.concat([d[c].fillna('missing').value_counts().rename_axis('value').reset_index(name='n').assign(variable=c)
        for c in ['weather', 'road_condition', 'vehicle', 'area', 'gender_group', 'cause', 'severity']], ignore_index=True)
    frequencies['percent'] = 100 * frequencies.n / len(d)
    eligible, coverage = injury_records(d)
    fields = ['weather', 'road_condition', 'vehicle', 'area', 'gender_group', 'cause']
    consistent = lambda s: s.fillna('unknown').iloc[0] if s.fillna('unknown').nunique() == 1 else 'mixed records'
    grouped = eligible.groupby('proxy').agg({'severe': 'max', 'year': 'first', **{c: consistent for c in fields}})
    tests = []
    for c in fields:
        if grouped.empty or grouped.severe.nunique() < 2: continue
        v = grouped[c]; freq = v.value_counts(); v = v.where(v.map(freq).ge(10), 'other (rare)')
        x, levels = pd.factorize(v); y = grouped.severe.to_numpy(dtype=int)
        if len(levels) < 2: continue
        observed = np.bincount(2 * x + y, minlength=2 * len(levels)).reshape(-1, 2)
        chi, p, _, expected = stats.chi2_contingency(observed, correction=False)
        rng = np.random.default_rng(42); exceed = 0
        blocks = [np.flatnonzero(grouped.year.to_numpy() == yr) for yr in grouped.year.unique()]
        for _ in range(permutations):
            yp = y.copy()
            for idx in blocks: yp[idx] = rng.permutation(y[idx])
            simulated = np.bincount(2 * x + yp, minlength=observed.size).reshape(observed.shape)
            exceed += np.sum((simulated - expected) ** 2 / expected) >= chi - 1e-12
        tests.append({'feature': c, 'proxy_n': len(y), 'Cramers_V': np.sqrt(chi / len(y)),
                      'expected_cells_below_5_pct': 100 * np.mean(expected < 5),
                      'year_blocked_p': (exceed + 1) / (permutations + 1)})
    tests = pd.DataFrame(tests)
    if not tests.empty:
        order = np.argsort(tests.year_blocked_p.to_numpy()); m = len(tests)
        adjusted = np.minimum.accumulate((tests.year_blocked_p.to_numpy()[order] * m / np.arange(1, m + 1))[::-1])[::-1]
        tests['BH_adjusted_p'] = np.nan; tests.loc[order, 'BH_adjusted_p'] = np.minimum(adjusted, 1)
    return {'frequencies': frequencies, 'associations': tests, 'severity_coverage': coverage,
            'numeric_summary': d[['hour']].describe().T.reset_index(names='variable')}
