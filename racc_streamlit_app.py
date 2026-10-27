"""Run: python -m streamlit run racc_streamlit_app.py
Keep the three part_*.py files in the same folder. Install requirements.txt first.
"""
from io import BytesIO
from datetime import date, time
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
import streamlit as st
try:
    from part_2_data import REQUIRED, prepare_data, make_counts, statistical_analysis, features
    from part_3_models import classification_analysis, prediction_analysis
    from part_4_results import forecasting_analysis, download_package
    from part_5_clustering import clustering_analysis, hotspot_analysis
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
    show_table('Classification performance (including Decision Tree)', result['metrics'], 'classification_metrics')
    show_table('Decision Tree precision, recall and F1 by class', result['decision_tree_report'], 'decision_tree_report')
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

    # Load the custom theme from style.css in the same folder as this script.
    css_path = Path(__file__).resolve().parent / 'style.css'
    if css_path.is_file():
        css = css_path.read_text(encoding='utf-8')
        st.markdown(f'<style>\n{css}\n</style>', unsafe_allow_html=True)
    else:
        st.warning(f'Custom theme not loaded: {css_path.name} was not found beside this app.')

    st.title('RACC data analysis')
    st.caption('Upload your CSV and run the analysis you need. All results stay in this browser session.')
    upload = st.sidebar.file_uploader('Upload CSV dataset', type=['csv'], key='dataset_upload')
    with st.sidebar.expander('CSV and cleaning settings'):
        fix = st.checkbox('Apply documented provisional date corrections', True, key='date_fixes')
        encoding = st.selectbox('Encoding', ['utf-8-sig', 'utf-8', 'cp1252', 'latin-1'], key='encoding')
        separator_name = st.selectbox('Separator', ['Comma', 'Semicolon', 'Tab'], key='separator')
        separator = {'Comma': ',', 'Semicolon': ';', 'Tab': '\t'}[separator_name]
