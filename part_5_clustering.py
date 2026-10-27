"""Exploratory K-Means patterns and descriptive Bunawan accident concentrations."""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from part_2_data import features

AREAS = ['bunawan brook','consuelo','imelda','libertad','mambalili','nueva era','poblacion','san andres','san marcos','san teodoro']

def clustering_analysis(clean, max_k=8):
    """K-Means on incident proxies; severity/cause excluded to avoid target leakage."""
    d = clean.loc[clean.date.notna()].copy()
    if d.empty: raise ValueError('No valid dated records for clustering.')
    d = d.sort_values(['date','source_row']).drop_duplicates('proxy').copy()
    if len(d) < 12: raise ValueError('At least 12 distinct incident proxies are needed.')
    x = features(d)
    categorical = ['weather','road_condition','vehicle','area','gender_group']
    numeric = [c for c in x if c not in categorical]
    prep = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical),
        ('num', Pipeline([('impute',SimpleImputer(strategy='constant',fill_value=0)),
                          ('scale',StandardScaler())]), numeric)
    ])
    matrix = prep.fit_transform(x)
    if np.unique(matrix,axis=0).shape[0] < 3: raise ValueError('Insufficient variation for meaningful clustering.')
    scores=[]; fits={}
    for k in range(2,min(max_k,len(d)-1)+1):
        model=KMeans(n_clusters=k,random_state=42,n_init=20)
        labels=model.fit_predict(matrix)
        if len(np.unique(labels)) < 2 or len(np.unique(labels)) >= len(d): continue
        score=silhouette_score(matrix,labels,sample_size=min(len(d),2000),random_state=42)
        scores.append({'k':k,'silhouette_score':float(score),'inertia':float(model.inertia_)})
        fits[k]=labels
    if not scores: raise ValueError('Could not produce valid clustering candidates.')
    scores=pd.DataFrame(scores).sort_values('silhouette_score',ascending=False)
    k=int(scores.iloc[0].k)
    assignments=d[['source_row','date','proxy','area','weather','road_condition','vehicle','cause','injury_grade']].copy()
    assignments['cluster']=fits[k]
    profile=assignments.groupby('cluster').agg(records=('proxy','size'),
        common_area=('area',lambda s:s.mode().iloc[0] if not s.mode().empty else 'unknown'),
        common_weather=('weather',lambda s:s.mode().iloc[0] if not s.mode().empty else 'unknown'),
        common_vehicle=('vehicle',lambda s:s.mode().iloc[0] if not s.mode().empty else 'unknown'),
        common_cause=('cause',lambda s:s.mode().iloc[0] if not s.mode().empty else 'unknown'),
        major_or_fatal=('injury_grade',lambda s:int(s.isin(['major injury','fatal']).sum()))).reset_index()
    profile['percent']=100*profile.records/len(assignments)
    return {'selected_name':f'K-Means (k={k})','silhouette':scores,'profiles':profile,
            'assignments':assignments,'sample_size':len(d)}

def hotspot_analysis(clean):
    """Barangay concentration only; not segment-level hotspots or risk rates."""
    d=clean.loc[clean.date.notna() & clean.area.isin(AREAS)].copy()
    if d.empty: raise ValueError('No valid dated records matched the ten specified barangays.')
    distinct=d.drop_duplicates('proxy')
    counts=distinct.groupby('area').agg(reported_incident_proxies=('proxy','nunique'),
        first_record=('date','min'),last_record=('date','max')).reset_index()
    counts['share_of_mapped_proxies_pct']=100*counts.reported_incident_proxies/counts.reported_incident_proxies.sum()
    counts=counts.sort_values('reported_incident_proxies',ascending=False).reset_index(drop=True)
    unresolved=clean.loc[~clean.area.isin(AREAS)].shape[0]
    return {'selected_name':'Barangay-level descriptive concentration', 'barangay_counts':counts,
            'data_limitations':pd.DataFrame([{'issue':'Rows outside ten named barangays (including unresolved)',
                                              'count':unresolved},
                                             {'issue':'Duplicate-proxy rows omitted from descriptive counts',
                                              'count':len(d)-len(distinct)}])}
