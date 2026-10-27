from pathlib import Path
import io, zipfile
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             classification_report, confusion_matrix, silhouette_score)
from sklearn.cluster import KMeans

st.set_page_config(page_title='RACC Decision Tree Dashboard', layout='wide')
REQUIRED = ['DATE','TIME','LOCATION','WEATHER','ROAD CONDITION','VEHICLE TYPE','CAUSE','SEVERITY','GENDER']
BARANGAYS = ['Bunawan Brook','Consuelo','Imelda','Libertad','Mambalili','Nueva Era','Poblacion','San Andres','San Marcos','San Teodoro']

@st.cache_data(show_spinner=False)
def load_data(raw_bytes, encoding='utf-8-sig', sep=','):
    df = pd.read_csv(io.BytesIO(raw_bytes), encoding=encoding, sep=sep)
    # Trim headers and cell whitespace without silently deleting rows.
    df.columns = [str(c).strip().upper() for c in df.columns]
    missing_cols = [c for c in REQUIRED if c not in df.columns]
    if missing_cols:
        raise ValueError('Missing required columns: ' + ', '.join(missing_cols))
    raw = df[REQUIRED].copy()
    raw['source_row'] = np.arange(2, len(raw)+2)
    d = raw.copy()
    d['date'] = pd.to_datetime(d['DATE'].astype('string').str.strip(), dayfirst=True, errors='coerce')
    d['hour'] = d['TIME'].map(parse_hour)
    for c in ['LOCATION','WEATHER','ROAD CONDITION','VEHICLE TYPE','CAUSE','SEVERITY','GENDER']:
        d[c] = d[c].astype('string').str.strip().replace('', pd.NA)
    d['severity_norm'] = d['SEVERITY'].str.lower().str.strip()
    d['severity_norm'] = d['severity_norm'].replace({
        'major injury':'major injury', 'minor injry':'minor injury', 'minor injury':'minor injury',
        'no injury':'no injury', 'fatal':'fatal', 'conscious':'conscious', 'unconscious':'unconscious'
    })
    d['injury_grade'] = d['severity_norm'].map({
        'minor injury':'minor injury','no injury':'no injury','major injury':'major injury','fatal':'fatal'
    })
    d['severe'] = d['injury_grade'].map({'minor injury':0,'no injury':0,'major injury':1,'fatal':1}).astype('Int64')
    d['year'] = d['date'].dt.year
    d['month'] = d['date'].dt.month
    d['weekday'] = d['date'].dt.dayofweek
    d['month_sin'] = np.sin(2*np.pi*d['month']/12)
    d['month_cos'] = np.cos(2*np.pi*d['month']/12)
    d['hour_sin'] = np.sin(2*np.pi*d['hour']/24)
    d['hour_cos'] = np.cos(2*np.pi*d['hour']/24)
    d['area'] = d['LOCATION'].map(normalize_location)
    d['weather'] = d['WEATHER'].str.lower().str.strip().replace({'sunny':'clear','light rain':'rainy'})
    d['road_condition'] = d['ROAD CONDITION'].str.lower().str.strip().replace({'wet':'wet','clear':'dry'})
    d['vehicle'] = d['VEHICLE TYPE'].str.lower().str.strip().replace({
        'motorcyle':'motorcycle','motor cycle':'motorcycle','motor single':'single motor'
    })
    d['cause'] = d['CAUSE'].str.lower().str.strip()
    d['gender_group'] = d['GENDER'].str.lower().str.strip()
    audit = pd.DataFrame({
        'check':['Original records','Exact duplicate rows (review before removal)','Missing date','Unusable time','Missing location','Missing weather','Missing road condition','Missing vehicle type','Missing cause','Missing severity','Missing gender','Usable injury-grade labels','Unclassified severity labels'],
        'count':[len(d),int(raw[REQUIRED].duplicated().sum()),int(d.date.isna().sum()),int(d.hour.isna().sum()),int(d.LOCATION.isna().sum()),int(d.WEATHER.isna().sum()),int(d['ROAD CONDITION'].isna().sum()),int(d['VEHICLE TYPE'].isna().sum()),int(d.CAUSE.isna().sum()),int(d.SEVERITY.isna().sum()),int(d.GENDER.isna().sum()),int(d.injury_grade.notna().sum()),int(d.injury_grade.isna().sum())]
    })
    return d, audit, raw

def parse_hour(value):
    if pd.isna(value): return np.nan
    s=str(value).strip()
    if not s: return np.nan
    t=pd.to_datetime(s, errors='coerce')
    if pd.notna(t): return float(t.hour + t.minute/60)
    # Also accept 24-hour strings such as 18:30 or numeric hour values.
    try:
        n=float(s)
        if 0 <= n <= 23: return n
    except Exception: pass
    return np.nan

def normalize_location(value):
    if pd.isna(value): return 'Unknown'
    s=' '.join(str(value).lower().replace('-', ' ').replace(',', ' ').split())
    for b in BARANGAYS:
        key=b.lower()
        if key in s:
            return b
    # Retain the original normalized location if no known barangay matched.
    return ' '.join(w.capitalize() for w in s.split())

def make_classifier(max_depth=5, min_leaf=8):
    cat=['weather','road_condition','vehicle','cause','area','gender_group']
    num=['hour_sin','hour_cos','month_sin','month_cos','weekday']
    pre=ColumnTransformer([
        ('cat',Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore'))]),cat),
        ('num',Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())]),num)
    ])
    tree=DecisionTreeClassifier(max_depth=max_depth,min_samples_leaf=min_leaf,class_weight='balanced',random_state=42)
    return Pipeline([('preprocess',pre),('tree',tree)])

def get_features(d):
    return d[['weather','road_condition','vehicle','cause','area','gender_group','hour_sin','hour_cos','month_sin','month_cos','weekday']].copy()

def chronological_split(frame):
    valid=frame.loc[frame.date.notna()].sort_values('date').copy()
    if valid.empty: raise ValueError('No valid dates are available for a chronological split.')
    last_year=valid.date.dt.year.max()
    train=valid.loc[valid.date.dt.year < last_year]
    test=valid.loc[valid.date.dt.year == last_year]
    if len(train)<40 or len(test)<15 or train.severe.nunique()<2 or test.severe.nunique()<2:
        cut=valid.date.quantile(.8)
        train=valid.loc[valid.date < cut]
        test=valid.loc[valid.date >= cut]
    train=train.loc[train.severe.notna()]
    test=test.loc[test.severe.notna()]
    if len(train)<30 or len(test)<10 or train.severe.nunique()<2:
        raise ValueError('Not enough dated, valid injury-grade records in both classes for a reliable holdout evaluation.')
    return train,test

def run_decision_tree(d, max_depth, min_leaf):
    eligible=d.loc[d.severe.notna() & d.date.notna()].copy()
    train,test=chronological_split(eligible)
    model=make_classifier(max_depth,min_leaf)
    Xtr=get_features(train); Xte=get_features(test); ytr=train.severe.astype(int); yte=test.severe.astype(int)
    model.fit(Xtr,ytr)
    pred=model.predict(Xte)
    proba=model.predict_proba(Xte)[:,list(model.named_steps['tree'].classes_).index(1)]
    base=DummyClassifier(strategy='prior').fit(np.zeros((len(ytr),1)),ytr)
    base_pred=base.predict(np.zeros((len(yte),1)))
    rows=[]
    for name,yp in [('Decision Tree',pred),('Majority-class baseline',base_pred)]:
        rows.append({'Model':name,'Accuracy':accuracy_score(yte,yp),'Precision (severe)':precision_score(yte,yp,zero_division=0),
                     'Recall (severe)':recall_score(yte,yp,zero_division=0),'F1-score (severe)':f1_score(yte,yp,zero_division=0),
                     'Macro F1':f1_score(yte,yp,average='macro',zero_division=0),'Test records':len(yte)})
    actual_pred=test[['source_row','date','LOCATION','WEATHER','ROAD CONDITION','VEHICLE TYPE','CAUSE','SEVERITY','injury_grade']].copy()
    actual_pred['actual_class']=yte.map({0:'Non-severe',1:'Severe'}).to_numpy()
    actual_pred['predicted_class']=pd.Series(pred,index=test.index).map({0:'Non-severe',1:'Severe'}).to_numpy()
    actual_pred['probability_severe']=proba
    actual_pred['correct_prediction']=actual_pred['actual_class'].eq(actual_pred['predicted_class'])
    cm=confusion_matrix(yte,pred,labels=[0,1])
    names=model.named_steps['preprocess'].get_feature_names_out()
    rules=export_text(model.named_steps['tree'],feature_names=list(names),decimals=3)
    importance=pd.DataFrame({'Feature':names,'Importance':model.named_steps['tree'].feature_importances_}).sort_values('Importance',ascending=False)
    report=pd.DataFrame(classification_report(yte,pred,labels=[0,1],target_names=['Non-severe','Severe'],output_dict=True,zero_division=0)).T.reset_index(names='Class')
    return {'model':model,'metrics':pd.DataFrame(rows),'actual_pred':actual_pred,'cm':cm,'rules':rules,'importance':importance,'report':report,'train':train,'test':test,'n_eligible':len(eligible)}

def run_kmeans(d, kmax=6):
    f=d.loc[d.date.notna()].copy()
    if len(f)<5: raise ValueError('At least five dated records are required for clustering.')
    cats=f[['weather','road_condition','vehicle','cause','area','gender_group']].fillna('Unknown')
    enc=OneHotEncoder(handle_unknown='ignore')
    mat=enc.fit_transform(cats)
    if mat.shape[0]>500:
        rng=np.random.RandomState(42); ix=np.sort(rng.choice(mat.shape[0],500,replace=False)); sample=mat[ix]
    else: sample=mat
    rows=[]
    for k in range(2,min(kmax, sample.shape[0]-1)+1):
        try:
            km=KMeans(n_clusters=k,random_state=42,n_init=10).fit(sample)
            score=silhouette_score(sample,km.labels_)
            rows.append({'k':k,'Silhouette score':score})
        except Exception: pass
    if not rows: raise ValueError('Could not calculate a silhouette score for this data.')
    best=max(rows,key=lambda x:x['Silhouette score'])['k']
    final=KMeans(n_clusters=best,random_state=42,n_init=10).fit(mat)
    f['Cluster']=final.labels_.astype(str)
    profiles=f.groupby('Cluster').agg(Records=('Cluster','size'),Most_common_weather=('weather',lambda x:x.mode().iat[0] if not x.mode().empty else 'Unknown'),Most_common_road=('road_condition',lambda x:x.mode().iat[0] if not x.mode().empty else 'Unknown'),Most_common_vehicle=('vehicle',lambda x:x.mode().iat[0] if not x.mode().empty else 'Unknown'),Most_common_cause=('cause',lambda x:x.mode().iat[0] if not x.mode().empty else 'Unknown'),Most_common_location=('area',lambda x:x.mode().iat[0] if not x.mode().empty else 'Unknown')).reset_index()
    return pd.DataFrame(rows),profiles,f

# Sidebar and upload
st.title('Road Accident Data Analysis Dashboard')
st.caption('Bunawan, Agusan del Sur | Decision Tree classification, K-Means clustering, and reported-accident concentration')
with st.sidebar:
    st.header('Dataset')
    uploaded=st.file_uploader('Upload RACC.csv',type=['csv'])
    st.caption('If no file is uploaded, the bundled RACC.csv is used.')
    max_depth=st.slider('Decision Tree maximum depth',2,8,5)
    min_leaf=st.slider('Minimum records per leaf',2,25,8)
    st.caption('Tree settings are chosen before evaluating the held-out data. Compare alternatives carefully to avoid tuning to the test set.')

if uploaded is not None:
    raw_bytes=uploaded.getvalue()
else:
    default_path=Path(__file__).with_name('RACC.csv')
    raw_bytes=default_path.read_bytes() if default_path.exists() else None
if raw_bytes is None:
    st.info('Upload the RACC.csv dataset to begin.'); st.stop()
try:
    d,audit,raw=load_data(raw_bytes)
except Exception as exc:
    st.error(str(exc)); st.stop()

# Overview
valid_sev=int(d.injury_grade.notna().sum())
col1,col2,col3,col4=st.columns(4)
col1.metric('Uploaded records',f'{len(d):,}')
col2.metric('Valid injury grades',f'{valid_sev:,}')
col3.metric('Records with valid dates',f'{int(d.date.notna().sum()):,}')
col4.metric('Exact duplicate rows to review',f'{int(raw[REQUIRED].duplicated().sum()):,}')
st.info('The model predicts severe vs. non-severe injury among records with recognized injury grades. “Conscious” and “Unconscious” are not automatically converted into injury grades. Exact duplicates are flagged for review, not automatically deleted.')

page=st.sidebar.radio('Pages',['Overview & cleaning','Decision Tree classification','K-Means clustering','Accident concentration','Data dictionary'])
if page=='Overview & cleaning':
    st.header('Data quality review')
    st.dataframe(audit,use_container_width=True,hide_index=True)
    st.subheader('Raw severity labels')
    st.bar_chart(d['SEVERITY'].fillna('Missing').astype(str).value_counts())
    st.subheader('Normalized location counts')
    st.bar_chart(d['area'].value_counts().head(15))
    st.subheader('Cleaned records')
    st.dataframe(d.drop(columns=['month_sin','month_cos','hour_sin','hour_cos']),use_container_width=True)
    st.download_button('Download cleaned dataset',d.to_csv(index=False).encode('utf-8'),file_name='RACC_cleaned.csv',mime='text/csv')
    st.download_button('Download data quality report',audit.to_csv(index=False).encode('utf-8'),file_name='RACC_data_quality_report.csv',mime='text/csv')
elif page=='Decision Tree classification':
    st.header('Decision Tree: severe versus non-severe injury')
    st.write('Severe = Major Injury or Fatal. Non-severe = Minor Injury or No Injury. Unrecognized severity labels are excluded from supervised training and evaluation, not imputed.')
    try:
        with st.spinner('Training and evaluating the Decision Tree on a chronological holdout…'):
            r=run_decision_tree(d,max_depth,min_leaf)
        m1,m2,m3,m4=st.columns(4)
        row=r['metrics'].iloc[0]
        m1.metric('Accuracy',f"{row['Accuracy']:.1%}")
        m2.metric('Precision (severe)',f"{row['Precision (severe)']:.1%}")
        m3.metric('Recall (severe)',f"{row['Recall (severe)']:.1%}")
        m4.metric('F1-score (severe)',f"{row['F1-score (severe)']:.1%}")
        st.caption(f"Training records: {len(r['train']):,} | Held-out records: {len(r['test']):,} | Eligible dated records: {r['n_eligible']:,}. Holdout dates: {r['test'].date.min().date()} to {r['test'].date.max().date()}.")
        st.subheader('Model performance')
        st.dataframe(r['metrics'].style.format({c:'{:.3f}' for c in ['Accuracy','Precision (severe)','Recall (severe)','F1-score (severe)','Macro F1']}),use_container_width=True,hide_index=True)
        st.subheader('Confusion matrix')
        cm=pd.DataFrame(r['cm'],index=['Actual non-severe','Actual severe'],columns=['Predicted non-severe','Predicted severe'])
        st.dataframe(cm,use_container_width=True)
        st.subheader('Decision Tree visualization')
        tree=r['model'].named_steps['tree']; feature_names=r['model'].named_steps['preprocess'].get_feature_names_out()
        fig,ax=plt.subplots(figsize=(22,10))
        plot_tree(tree,feature_names=feature_names,class_names=['Non-severe','Severe'],filled=False,rounded=True,fontsize=7,max_depth=3,ax=ax)
        ax.set_title('Top levels of the evaluated Decision Tree (display limited to depth 3)')
        st.pyplot(fig); plt.close(fig)
        st.caption('Visualization is limited to the first three levels for readability. Metrics are from the full evaluated tree.')
        st.subheader('Learned decision rules')
        st.code(r['rules'],language='text')
        st.subheader('Actual versus predicted holdout records')
        st.dataframe(r['actual_pred'],use_container_width=True,hide_index=True)
        st.download_button('Download actual vs predicted CSV',r['actual_pred'].to_csv(index=False).encode('utf-8'),file_name='decision_tree_actual_vs_predicted.csv',mime='text/csv')
        st.download_button('Download classification metrics CSV',r['metrics'].to_csv(index=False).encode('utf-8'),file_name='decision_tree_metrics.csv',mime='text/csv')
        st.subheader('Feature importance')
        st.bar_chart(r['importance'].head(15).set_index('Feature')['Importance'])
        st.dataframe(r['importance'],use_container_width=True,hide_index=True)
        st.caption('Feature importance describes splits used by this fitted tree; it does not prove causation. Results depend on label quality and reporting patterns.')
    except Exception as exc:
        st.error('Could not evaluate Decision Tree: '+str(exc))
elif page=='K-Means clustering':
    st.header('K-Means accident pattern clustering')
    st.caption('Clustering groups similar records based on categorical context. It does not predict future accidents or establish causes.')
    if st.button('Run K-Means analysis'):
        try:
            scores,profiles,assigned=run_kmeans(d)
            st.session_state['kmeans_result']=(scores,profiles,assigned)
        except Exception as exc: st.error(str(exc))
    if 'kmeans_result' in st.session_state:
        scores,profiles,assigned=st.session_state['kmeans_result']
        st.subheader('Silhouette score by number of clusters')
        st.dataframe(scores,use_container_width=True,hide_index=True)
        st.line_chart(scores.set_index('k')['Silhouette score'])
        st.subheader('Cluster profiles')
        st.dataframe(profiles,use_container_width=True,hide_index=True)
        st.download_button('Download cluster assignments',assigned.to_csv(index=False).encode('utf-8'),file_name='RACC_kmeans_assignments.csv',mime='text/csv')
elif page=='Accident concentration':
    st.header('Reported accident concentration by normalized location')
    counts=d.groupby('area').agg(Reported_records=('area','size'),Records_with_valid_severity=('injury_grade',lambda x:int(x.notna().sum())),Severe_labeled_records=('severe',lambda x:int((x==1).sum()))).sort_values('Reported_records',ascending=False).reset_index()
    st.bar_chart(counts.set_index('area')['Reported_records'])
    st.dataframe(counts,use_container_width=True,hide_index=True)
    st.download_button('Download location counts',counts.to_csv(index=False).encode('utf-8'),file_name='RACC_location_counts.csv',mime='text/csv')
    st.warning('These are counts by location text, not verified road-segment hotspots. The dataset does not contain coordinates, road-segment IDs, traffic exposure, or a denominator for accident rates.')
elif page=='Data dictionary':
    st.header('Fields used in this dashboard')
    dictionary=pd.DataFrame([
        ['DATE','Parsed as day/month/year; invalid dates remain missing.','Date for chronological evaluation and calendar features.'],
        ['TIME','Converted to hour when parseable.','Hour-of-day features.'],
        ['LOCATION','Original text retained; normalized to a recognized barangay when possible.','Location pattern and reported-record counts.'],
        ['WEATHER','Whitespace trimmed; common categories normalized for analysis.','Input feature.'],
        ['ROAD CONDITION','Whitespace trimmed and common capitalization variants normalized.','Input feature.'],
        ['VEHICLE TYPE','Common spelling variants normalized for analysis.','Input feature.'],
        ['CAUSE','Whitespace/case normalized for analysis.','Input feature; association is not causation.'],
        ['SEVERITY','Minor Injury/No Injury = non-severe; Major Injury/Fatal = severe.','Prediction target. Conscious/Unconscious and missing labels excluded.'],
        ['GENDER','Original text retained with lowercase analysis copy.','Optional contextual input feature.']
    ],columns=['Original field','Cleaning rule','Role'])
    st.dataframe(dictionary,use_container_width=True,hide_index=True)
    st.caption('Review all normalization rules with the source agency before treating them as official coding rules.')
