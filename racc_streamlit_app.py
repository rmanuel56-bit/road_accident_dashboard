"""Run: python -m streamlit run racc_streamlit_app.py
Keep the three part_*.py files in the same folder. Install requirements.txt first.
"""
from io import BytesIO
from datetime import date, time
import hashlib
import numpy as np
import pandas as pd
import streamlit as st
try:
    from part_2_data import REQUIRED, prepare_data, make_counts, statistical_analysis, features
    from part_3_models import classification_analysis, prediction_analysis
    from part_4_results import forecasting_analysis, download_package
except ModuleNotFoundError as exc:
    st.error('Missing file or Python package: ' + str(exc.name))
    st.info('Extract the complete ZIP into one folder, then install requirements.txt. Run the main file only.')
    st.stop()

def show_table(title, frame, key):
    st.subheader(title)
    if frame.empty:
        st.info('No rows meet the requirements for this table.'); return
    st.dataframe(frame, width='stretch')
    st.download_button('Download CSV', frame.to_csv(index=False).encode('utf-8'),
                       file_name=key + '.csv', mime='text/csv', key='download_' + key)

def run_analysis(bundle, key, function, *args):
    try:
        with st.spinner('Running ' + key + '…'):
            bundle['results'][key] = function(*args)
        st.success('Analysis completed.')
    except Exception as exc:
        st.error(key.capitalize() + ': ' + str(exc))
        with st.expander('Error details'): st.exception(exc)

def show_classification(result):
    st.write('Selected using development data:', result['selected_name'])
    show_table('Classification performance', result['metrics'], 'classification_metrics')
    if result['metrics'].iloc[0].ROC_AUC < .55:
        st.warning('Held-out discrimination is close to chance. Feature rankings are exploratory, not validated risk factors.')
    for title, key in [('Development comparison', 'cv'), ('Train/test dates', 'split'),
                       ('Class-level results', 'report'), ('Four-class reference results', 'multiclass_report')]:
        with st.expander(title): show_table(title, result[key], 'classification_' + key)
    st.write('Binary confusion matrix: 0 = minor/no injury; 1 = major/fatal.')
    st.dataframe(result['confusion'], width='stretch')
    st.subheader('Which variables matter to the selected model?')
    st.bar_chart(result['importance'].set_index('variable')['macro_F1_drop'])
    show_table('Variable importance', result['importance'], 'classification_importance')
    show_table('Variable-set comparisons', result['variable_combinations'], 'classification_variable_sets')
    show_table('Supported weather/road/vehicle combinations', result['observed_combinations'], 'observed_combinations')
    st.caption('Combinations use predictions from earlier-year training folds. Importance is post-evaluation and does not establish causation.')
    with st.expander('Simple reference-tree rules'): st.code(result['rules'], language='text')

def prediction_form(bundle):
    result = bundle['results'].get('classification')
    if result is None: return
    st.subheader('Try one severity prediction')
    with st.form('severity_form'):
        a, b = st.columns(2)
        when = a.date_input('Date', date(2024, 8, 15)); clock = b.time_input('Time', time(20, 0))
        location = a.text_input('Location', 'P1 San Teodoro')
        vehicle = b.text_input('Vehicle type', 'Motorcycle')
        weather = a.selectbox('Weather', sorted(bundle['data']['clean'].weather.dropna().unique()))
        road = b.selectbox('Road condition', sorted(bundle['data']['clean'].road_condition.dropna().unique()))
        gender = a.selectbox('Gender as recorded', ['Male', 'Female', 'Male & Female'])
        submitted = st.form_submit_button('Predict severity')
    if submitted:
        record = dict(zip(REQUIRED, [when.strftime('%d/%m/%Y'), clock.strftime('%I:%M %p'), location,
                                    weather, road, vehicle, '', '', gender]))
        incoming = prepare_data(pd.DataFrame([record]).to_csv(index=False).encode())['clean']
        probability = result['model'].predict_proba(features(incoming))[0, 1]
        st.metric('Model probability: major injury or fatal', f'{probability:.1%}')
        st.caption('Illustrative prediction among reported records. Review the performance table before use.')

def main():
    st.set_page_config(page_title='RACC analysis', layout='wide')
    st.title('RACC data analysis')
    st.caption('Upload your CSV and run the analysis you need. All results stay in this browser session.')
    upload = st.sidebar.file_uploader('Upload CSV dataset', type=['csv'], key='dataset_upload')
    with st.sidebar.expander('CSV and cleaning settings'):
        fix = st.checkbox('Apply documented provisional date corrections', True, key='date_fixes')
        encoding = st.selectbox('Encoding', ['utf-8-sig', 'utf-8', 'cp1252', 'latin-1'], key='encoding')
        separator_name = st.selectbox('Separator', ['Comma', 'Semicolon', 'Tab'], key='separator')
        separator = {'Comma': ',', 'Semicolon': ';', 'Tab': '\t'}[separator_name]
    coverage = st.sidebar.checkbox('Assume complete recording calendar', True, key='calendar_coverage')
    page = st.sidebar.radio('Analysis', ['1. Upload and cleaning', '2. Statistics and EDA',
        '3. Classification and prediction', '4. Forecasting and downloads'], key='analysis_page')
    if upload is None:
        st.session_state.pop('racc_bundle', None)
        st.info('Upload RACC.csv using the left panel.')
        st.code(','.join(REQUIRED), language='text'); return
    signature = hashlib.sha256(upload.getvalue() + str((fix, encoding, separator, coverage)).encode()).hexdigest()
    bundle = st.session_state.get('racc_bundle')
    if bundle is None or bundle['signature'] != signature:
        st.session_state.pop('racc_bundle', None)
        try:
            prepared = prepare_data(upload.getvalue(), fix, encoding, separator)
        except Exception as exc:
            st.error('Could not read this dataset: ' + str(exc)); return
        bundle = {'signature': signature, 'data': prepared, 'results': {}, 'filename': upload.name}
        st.session_state['racc_bundle'] = bundle
    d = bundle['data']['clean']; results = bundle['results']
    a, b, c = st.columns(3)
    a.metric('Records', len(d)); b.metric('Usable injury grades', int(d.injury_grade.notna().sum()))
    c.metric('Analyses completed', len(results))
    st.caption('Repeated rows are retained. Counts describe reported records, not verified unique accidents or accident rates.')

    if page.startswith('1.'):
        st.header('Upload and data cleaning')
        show_table('Data quality', bundle['data']['audit'], 'data_quality')
        st.write('Exact repeats after the first occurrence:', bundle['data']['exact_repeats'])
        st.write('Unusable dates:', int(d.date.isna().sum()), '| Missing/unusable times:', int(d.hour.isna().sum()))
        show_table('Dates requiring review', bundle['data']['date_review'], 'date_review')
        show_table('Cleaned data', d, 'cleaned_data')
        st.info('Proposed year/date repairs require source-owner confirmation. Conscious and Unconscious are not injury grades. Missing target labels are not imputed.')

    elif page.startswith('2.'):
        st.header('Statistical analysis and EDA')
        if st.button('Run statistical analysis', key='run_statistics'):
            run_analysis(bundle, 'statistics', statistical_analysis, d)
        if 'statistics' in results:
            for title, key in [('Category frequencies', 'frequencies'), ('Exploratory association tests', 'associations'),
                               ('Severity-label coverage by year', 'severity_coverage'), ('Numeric description', 'numeric_summary')]:
                show_table(title, results['statistics'][key], 'statistics_' + key)
            st.caption('Association tests collapse provisional incidents, permute labels within year and adjust p-values using Benjamini–Hochberg. Independence and reporting completeness remain assumptions.')
        category = st.selectbox('Explore a variable', ['weather', 'road_condition', 'vehicle', 'area', 'severity', 'gender_group', 'cause'], key='eda_variable')
        st.bar_chart(d[category].fillna('missing').value_counts().head(20))
        st.subheader('Missing values')
        st.bar_chart(d[['date', 'hour', 'location', 'injury_grade']].isna().mean().mul(100))
        dated = d.loc[d.date.notna()]
        if not dated.empty:
            st.subheader('Monthly reported rows')
            st.line_chart(dated.groupby(pd.Grouper(key='date', freq='MS')).size())
            st.subheader('Severity vocabulary by year')
            st.bar_chart(pd.crosstab(d.year, d.severity.fillna('missing')))

    elif page.startswith('3.'):
        st.header('Classification and numeric prediction')
        st.write('Severity classification uses injury-grade recording years with at least 50% valid grades. The latest eligible dates are held out; preprocessing is fitted within training folds.')
        if st.button('Run classification', key='run_classification'):
            run_analysis(bundle, 'classification', classification_analysis, d)
        if 'classification' in results:
            show_classification(results['classification']); prediction_form(bundle)
        st.divider(); st.subheader('Predict daily reported-row counts')
        st.caption('One-day-ahead predictions use observed earlier days. A calendar day absent from the file is assumed to have zero recorded rows. MAE selection can favor a zero-median baseline; compare all error metrics.')
        if not coverage: st.info('Enable the complete recording-calendar assumption to model counts.')
        if st.button('Run numeric prediction', disabled=not coverage, key='run_prediction'):
            try: counts = make_counts(d)
            except Exception as exc: st.error(str(exc))
            else: run_analysis(bundle, 'prediction', prediction_analysis, counts)
        if 'prediction' in results:
            r = results['prediction']
            show_table('Numeric prediction performance', r['metrics'], 'prediction_metrics')
            st.line_chart(r['predictions'].set_index('date')[['actual', 'predicted']])
            show_table('Next day after the dataset end', r['next_day'], 'next_day')
            show_table('Daily prediction variables', r['importance'], 'prediction_importance')
            show_table('Daily variable-set comparisons', r['variable_combinations'], 'prediction_variable_sets')
            with st.expander('Development comparison and split'):
                show_table('Development MAE', r['cv'], 'prediction_cv'); st.dataframe(r['split'])
            st.caption('Importance describes the best learned reference if a constant baseline wins; it does not show that the learned model improves on that baseline.')

    else:
        st.header('Forecasting, model performance and downloads')
        horizon = st.slider('Months beyond the dataset end', 1, 24, 12, key='forecast_horizon')
        if st.button('Run monthly forecasting', disabled=not coverage, key='run_forecast'):
            try: counts = make_counts(d)
            except Exception as exc: st.error(str(exc))
            else: run_analysis(bundle, 'forecast', forecasting_analysis, counts, horizon)
        if 'forecast' in results:
            r = results['forecast']
            show_table('Monthly forecast performance', r['metrics'], 'forecast_metrics')
            st.line_chart(r['holdout'].set_index('date')[['actual', 'predicted']])
            show_table('Forecast after the dataset end', r['future'], 'future_forecast')
            st.line_chart(r['future'].set_index('date')[['predicted', 'approx_90pct_low', 'approx_90pct_high']])
            st.caption(f"Approximate 90% intervals use {r['calibration_errors']} development errors. Holdout coverage: {r['interval_coverage']:.1%}. Temporal dependence and small samples limit coverage guarantees.")
            st.info('Forecast dates follow the last dataset month and may already be in the past. Update the data before current operational use.')
            with st.expander('Forecast development comparison'): show_table('Development MAE', r['cv'], 'forecast_cv')
        performance = [r['metrics'].assign(task=task) for task, r in results.items() if 'metrics' in r]
        if performance: show_table('All completed model evaluations', pd.concat(performance, ignore_index=True), 'all_model_metrics')
        st.download_button('Download all completed results (ZIP)', download_package(bundle['data'], results, upload.name),
            file_name='RACC_results.zip', mime='application/zip', key='download_results')
        st.caption('The ZIP includes cleaned data and the analyses you have actually run. Run statistics, classification, numeric prediction and forecasting to include every task.')

if __name__ == '__main__':
    main()
