# Simple RACC Streamlit app

This version keeps the seven requested analysis tasks while using four short Python files,
ordinary imports, native Streamlit charts and five direct dependencies. There are no dynamic
module registries, shared-service factories, temporary output folders, Colab upload routines,
or Matplotlib dependencies. Analyses run independently when you click their buttons.

## Install and run

Use Python 3.10 or newer; Python 3.12 is recommended and was tested.

1. Extract the ZIP completely into one folder. Replace the previous four Python files with
   these four files. Do not mix files from the old and new versions.
2. Open a terminal in that folder and run:

       python -m pip install -r requirements.txt
       python -m streamlit run racc_streamlit_app.py

Keep these files together, with their exact names:

- racc_streamlit_app.py — the main interface; run this file only.
- part_2_data.py — upload parsing, cleaning and descriptive statistics.
- part_3_models.py — classification, daily count prediction and feature combinations.
- part_4_results.py — monthly forecasting and downloadable results.
- requirements.txt — package requirements.

For Streamlit Community Cloud, put all four Python files and requirements.txt in the same
repository folder and select racc_streamlit_app.py as the entry point. Select Python 3.12.

## Use all seven analysis tasks

1. Upload the RACC CSV in the sidebar. Page 1 loads it, cleans values and shows the data audit.
2. On page 2 click Run statistical analysis. Inspect frequencies, within-year permutation
   association tests, BH-adjusted p-values, category plots, missingness and time trends.
3. On page 3 click Run classification and Run numeric prediction separately. Each analysis
   uses chronological development validation and an untouched later-date holdout. Performance,
   feature importance and variable-set comparisons appear below their model results.
4. On page 4 click Run monthly forecasting. Review fixed-origin backtests, held-out performance,
   dataset-relative future forecasts and approximate intervals. Download the ZIP of all completed
   analyses. If an analysis has not been run, its results are not included in the download.

Required CSV headers:

    DATE,TIME,LOCATION,WEATHER,ROAD CONDITION,VEHICLE TYPE,CAUSE,SEVERITY,GENDER

Headers are trimmed and converted to uppercase. Use the sidebar to change CSV encoding or
separator. Uploading another file or changing cleaning/calendar settings clears previous results.
Small or unsuitable datasets receive a clear message; other tasks can still run.

## Analysis decisions

- Keep repeated rows and flag them: they may represent different people or vehicles.
- Explicit provisional fixes address nine malformed date rows in the original RACC dataset.
  They can be disabled and require owner confirmation. Other dates use the dominant day/month/year
  convention, with a month/day exception only when unambiguous. Unusable dates are flagged.
- Conscious and Unconscious are not injury grades. Injury modeling excludes recording years
  with fewer than 50% usable injury-grade labels; the coverage table makes this rule visible.
- Major/fatal vs minor/no injury is the primary classifier. A separate four-class random-forest
  reference is also evaluated. Class support matters, especially for fatal cases.
- Severity models compare prior baseline, logistic regression and random forest using earlier
  development folds. Choose by macro-F1, then evaluate the selected model on reserved dates.
- Daily count prediction compares mean/median baselines, Poisson regression and random forest
  using development MAE. Previous observed days update its lags: this is one-day-ahead prediction.
  A constant baseline can win MAE; inspect the other error metrics and reference results as well.
- Forecasting compares last-value and seasonal-naive baselines, simple smoothing and regularized
  trend/seasonal regression. Select on rolling-origin development backtests, evaluate on a later
  fixed-origin holdout, then refit for dates beyond the dataset end. At least 30 monthly observations
  are needed. Approximate 90% bands use pooled development forecast errors; they are not a formal
  guarantee under temporal dependence or incomplete reporting.
- Learned preprocessing is fitted inside training pipelines. Whole dates stay together. Count lags
  and rolling means use only earlier counts. Cause and recorded severity text are excluded from
  classification predictors. Variable-set comparisons use development folds; permutation importance
  is a post-evaluation explanation and is not used to choose new features on the holdout.
- Statistical tests collapse matching date/time/location proxy groups and permute the binary
  response within year, with BH multiple-testing adjustment. Proxy grouping does not prove
  independence. Statistical associations and model explanations do not establish causation.
- Counts measure reported rows, not verified unique accidents or exposure-adjusted accident rates.
  Missing dates and boundary months are treated as a complete reporting calendar only under the
  visible sidebar assumption. Forecast dates may already be in the past; update observations for
  current operational forecasts. Model performance can be weak and is shown as observed.

The simplified algorithms are smaller than the earlier notebook's model collection; model scores
can differ. All requested task categories remain available. Fitted evaluation pipelines can be
downloaded as pickle files; load them only from trusted sources in compatible environments.

## If deployment still fails

The earlier deployment error was not provided, so its exact cause cannot be confirmed. This
version removes the custom loading framework and rigid exact-version pins. If it fails, copy
the first full error from Streamlit's logs; it will identify a missing file/package, incompatible
environment or dataset issue. Splitting code does not increase a hosting service's memory limit.

Primary documentation:
https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/file-organization
https://docs.streamlit.io/develop/api-reference/widgets/st.file_uploader
https://scikit-learn.org/stable/common_pitfalls.html
