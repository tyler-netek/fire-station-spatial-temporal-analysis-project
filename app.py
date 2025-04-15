import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import os
import time
from itertools import product
import plotly.express as px
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

st.set_page_config(layout="wide", page_title="nyc fire risk analysis")


OPTIMIZATION_DATA_DIR= os.path.join(".", "Optimization", "Data")
PREDICTION_DATA_DIR = os.path.join(".", "Prediction", "Data")
PREDICTION_MODEL_DIR= os.path.join(".", "Prediction", "Models")
VISUALIZATIONS_PATH = os.path.join(".", "Visualizations")
WAVELET_FEATURES_FILE = os.path.join(".", "Prediction", "wavelet_features.csv")
FIRE_INCIDENCE_FILE = os.path.join("Misc", "Final_Fire_Incidence_Data_with_PopDensity.csv")
ACS_DATA_FILE= os.path.join(".", "ACSST5Y2023.S1901-Data.csv")
POTENTIAL_LOC_FILE = os.path.join(OPTIMIZATION_DATA_DIR, "Potential_location.csv")
OPTIMAL_LOC_FILE = os.path.join(OPTIMIZATION_DATA_DIR, "optimal_ga_locations.csv")
BUILDING_RISK_FILE = os.path.join(PREDICTION_DATA_DIR, "building_fire_risk.csv")
DEMO_DATA_PATH = os.path.join(PREDICTION_DATA_DIR, "nyc_demographic_data.csv")
ECO_DATA_PATH = os.path.join(PREDICTION_DATA_DIR, "nyc_economic_data.csv")
FDNY_STATIONS_PATH = os.path.join(VISUALIZATIONS_PATH, "FDNY_Firehouse_Listing_20250312.csv")
KDE_ZIP_OUTPUT_PATH = os.path.join(OPTIMIZATION_DATA_DIR, "kde_by_zipcode.csv")
PBI_ZIP_PATH = os.path.join(VISUALIZATIONS_PATH, "Team_139_PBI_Visualization.zip")


EXPECTED_LOGISTIC_FEATURES = [
    'high_mean_energy', 'high_max_energy', 'high_dominant_scale',
    'med_mean_energy', 'med_max_energy', 'med_dominant_scale',
    'low_mean_energy', 'low_max_energy', 'low_dominant_scale',
    'Month_1', 'Month_2', 'Month_3', 'Month_4', 'Month_5', 'Month_6',
    'Month_7', 'Month_8', 'Month_9', 'Month_10', 'Month_11', 'Month_12',
    'Date_1', 'Date_2', 'Date_3', 'Date_4', 'Date_5', 'Date_6', 'Date_7',
    'Date_8', 'Date_9', 'Date_10', 'Date_11', 'Date_12', 'Date_13',
    'Date_14', 'Date_15', 'Date_16', 'Date_17', 'Date_18', 'Date_19',
    'Date_20', 'Date_21', 'Date_22', 'Date_23', 'Date_24', 'Date_25',
    'Date_26', 'Date_27', 'Date_28', 'Date_29', 'Date_30', 'Date_31',
    'Hour_group_0', 'Hour_group_4', 'Hour_group_8', 'Hour_group_12',
    'Hour_group_16', 'Hour_group_20'
]


@st.cache_data
def load_csv(file_path):
    if not os.path.exists( file_path ):
        st.warning(f"data file missing: {file_path}.")
        return None
    try:
        df = pd.read_csv(file_path, low_memory=False)
        return df
    except Exception as e:
        st.error(f"couldn't load {file_path}: {e}")
        return None

@st.cache_resource
def load_pkl_model(model_fpath):
    if not os.path.exists( model_fpath ):
        st.warning(f"model file missing: {model_fpath}.")
        return None
    try:
        loaded_model = joblib.load(model_fpath)
        return loaded_model
    except Exception as e:
        st.error(f"problem loading pkl model {model_fpath}: {e}")
        return None

@st.cache_resource
def load_keras_model( model_fpath ):
    if not os.path.exists( model_fpath ):
        st.warning(f"model file missing: {model_fpath}.")
        return None
    try:
        from tensorflow.keras.models import load_model
        with st.spinner(f'loading dnn ({os.path.basename(model_fpath)})...'):
            model = load_model( model_fpath )
        return model
    except ImportError:
        st.error("tensorflow needed but not installed/found?")
        return None
    except Exception as e:
        st.error(f"problem loading keras model {model_fpath}: {e}")
        return None

def prep_data_for_prediction(zipcode, month, day, hour, wavelet_features):
    if wavelet_features is None: return None
    try:
        pd.Timestamp(year=2024, month=month, day=day )
    except ValueError:
         st.error(f"looks like an invalid date: {month}/{day}")
         return None

    hour_group_val = (hour // 4)*4
    input_data_dict = {'MODZCTA': [zipcode], 'Month': [month], 'Date': [day], 'Hour_group': [hour_group_val]}
    input_df_prep = pd.DataFrame( input_data_dict )
    input_df_encoded = pd.get_dummies(input_df_prep, columns=['Month', 'Date', 'Hour_group'])
    wavelet_select = wavelet_features[wavelet_features['MODZCTA'] == zipcode]
    if wavelet_select.empty:
        st.error(f"couldn't find wavelet features for zip {zipcode} in provided data.")
        return None
    merged_df_prep = pd.merge( input_df_encoded, wavelet_select, on='MODZCTA', how='left' )

    if merged_df_prep.empty or merged_df_prep.isnull().any().any():
         st.error(f"problem merging wavelet features for zip {zipcode}. check input file.")
         return None

    processed_df_prep = merged_df_prep.drop('MODZCTA', axis=1)

    required_cols = EXPECTED_LOGISTIC_FEATURES

    for col_name in required_cols:
        if col_name not in processed_df_prep.columns:
            processed_df_prep[col_name] = 0
    try:
        processed_df_prep = processed_df_prep[required_cols]
    except KeyError as e:
        st.error(f"column mismatch error: {e}. check hardcoded feature list vs generated dummies.")
        return None

    return processed_df_prep

if 'predict_loaded' not in st.session_state:
    st.session_state.predict_loaded = False
    st.session_state.models = {}
    st.session_state.wavelet_unique = None
    st.session_state.zip_list = []
if 'opt_loaded' not in st.session_state:
    st.session_state.opt_loaded = False
    st.session_state.opt_dfs = {}
if 'viz_loaded' not in st.session_state:
    st.session_state.viz_loaded = False
    st.session_state.viz_dfs = {}
if 'explore_last_loaded_df' not in st.session_state:
    st.session_state.explore_last_loaded_df = None
    st.session_state.explore_last_loaded_path = None
    st.session_state.explore_last_loaded_name = None

st.title("\U0001F692 NYC Fire Station Spatial-Temporal Analysis \U0001F5FD")

st.sidebar.header("Select View")
selected_page = st.sidebar.selectbox(
    "Section:",
    ["Welcome", "Explore Data", "Predict Risk", "Optimization", "Visualizations"]
)
st.sidebar.info(f"Last Refresh: {time.strftime('%Y-%m-%d %H:%M:%S')}")

if selected_page == "Welcome":
    st.header("Welcome")
    st.markdown("""
    This application provides an interactive interface for our CSE6242 project regarding NYC 🗽 fire station analysis.
    We investigated:
    * Spatial and temporal patterns of fire incidents.
    * Predictive modeling of fire risk using wavelet-based time-series features.
    * Optimization of potential new fire station locations using KDE and Genetic Algorithm techniques.

    Please use the sidebar to navigate the analysis sections. Click buttons within sections to load relevant data and models.
    """)
    st.caption("cs bit: emojis like 🚒 and 🗽 have unique hexadecimal codes (unicode points) behind the scenes - the statue is U+1F5FD!")
    st.caption("streamlit reruns the script on interactions, so caching heavy stuff like data loads (@st.cache_data) is key for speed, unlike maybe persistent objects in a compiled app.")
    st.info("Team 139: Tyler, Vinuka, Harshitha, Rishi, Madeleine, Kevin")


elif selected_page == "Explore Data":
    st.header("Dataset Overview")
    st.write("select a dataset from the dropdown below, then click load.")

    datasets_options = {
        "Select a dataset...": None,
        "Final Fire Incidents (w/ Pop Density)": FIRE_INCIDENCE_FILE,
        "Wavelet Features (Logistic Input)": WAVELET_FEATURES_FILE,
        "Candidate Station Locations (KDE/KMedoids Output)": POTENTIAL_LOC_FILE,
        "Optimal Station Locations (GA Output)": OPTIMAL_LOC_FILE,
        "Existing FDNY Stations": FDNY_STATIONS_PATH,
        "NYC Demographics": DEMO_DATA_PATH,
        "NYC Economics": ECO_DATA_PATH,
        "KDE Scores (by Zip Output)": KDE_ZIP_OUTPUT_PATH,
    }

    selected_dataset_name = st.selectbox(
        "Select Dataset:",
        options=list(datasets_options.keys())
        )

    if selected_dataset_name and selected_dataset_name != "Select a dataset...":
        filepath_to_load = datasets_options[selected_dataset_name]
        load_btn = st.button(f"Load Sample: {selected_dataset_name}", key=f"explore_btn_{selected_dataset_name}")
        if load_btn:
             df = load_csv(filepath_to_load)
             if df is not None:
                 st.session_state.explore_last_loaded_df = df
                 st.session_state.explore_last_loaded_path = filepath_to_load
                 st.session_state.explore_last_loaded_name = selected_dataset_name
             else:
                 st.session_state.explore_last_loaded_df = None
                 st.session_state.explore_last_loaded_path = filepath_to_load
                 st.session_state.explore_last_loaded_name = selected_dataset_name + " (Load Failed)"

    st.divider()

    if st.session_state.explore_last_loaded_df is not None:
        st.subheader(f"Showing Sample: {st.session_state.explore_last_loaded_name}")
        st.dataframe(st.session_state.explore_last_loaded_df.head())
        st.caption(f"shape: {st.session_state.explore_last_loaded_df.shape} | source: `{st.session_state.explore_last_loaded_path}`")
    elif st.session_state.explore_last_loaded_name is not None:
        st.subheader(st.session_state.explore_last_loaded_name)
        st.caption(f"file not found or load failed: `{st.session_state.explore_last_loaded_path}`")


elif selected_page == "Predict Risk":
    st.header("Predict Fire Incident Probability")

    if not st.session_state.predict_loaded:
        st.markdown("Models and features needed for prediction are not loaded yet.")
        if st.button("Load Prediction Environment", key="predict_load_btn"):
            wv_df = load_csv(WAVELET_FEATURES_FILE)
            if wv_df is not None:
                 st.session_state['wavelet_unique'] = wv_df.drop_duplicates(subset=['MODZCTA']).copy()
                 if 'MODZCTA' in st.session_state['wavelet_unique'].columns:
                      st.session_state['zip_list'] = sorted(st.session_state['wavelet_unique']['MODZCTA'].astype(int).unique())
                 else: st.error("modzcta column missing in wavelet file!")
            else: st.error(f"fatal: wavelet file ({WAVELET_FEATURES_FILE}) missing!")

            st.session_state.models['any'] = load_pkl_model(os.path.join(PREDICTION_MODEL_DIR,"any_risk_logistic.pkl"))
            st.session_state.models['high'] = load_pkl_model(os.path.join(PREDICTION_MODEL_DIR, "high_risk_logistic.pkl"))
            st.session_state.models['med'] = load_pkl_model(os.path.join(PREDICTION_MODEL_DIR,"med_risk_logistic.pkl"))
            st.session_state.models['low'] = load_pkl_model(os.path.join(PREDICTION_MODEL_DIR, "low_risk_logistic.pkl"))
            st.session_state.models['dnn'] = load_keras_model(os.path.join(PREDICTION_MODEL_DIR,"dnn_model.keras"))

            if st.session_state.get('wavelet_unique') is not None and st.session_state.models.get('high') is not None:
                st.session_state.predict_loaded = True
            else:
                st.error("failed to load necessary models or wavelet data for prediction.")

    if selected_page == "Predict Risk" and st.session_state.predict_loaded:
        st.markdown("Select a time and location to estimate the probability of a fire incident within that 4-hour window, based on the trained logistic models using wavelet features.")

        wavelet_unique_to_use = st.session_state.get('wavelet_unique')
        zip_list = st.session_state.get('zip_list', [])
        model_any_loaded = st.session_state.models.get('any')
        model_high_loaded = st.session_state.models.get('high')
        model_med_loaded = st.session_state.models.get('med')
        model_low_loaded = st.session_state.models.get('low')
        model_dnn_loaded = st.session_state.models.get('dnn')

        if wavelet_unique_to_use is None or not zip_list:
            st.error("wavelet data or zip list issue.")
        else:
            colA, colB = st.columns(2)
            colC, colD = st.columns(2)

            with colA:
                selected_zipcode = st.selectbox("Zip Code (MODZCTA):", zip_list)
            with colB:
                selected_month = st.selectbox("Month:", list(range(1, 13)), index= 4)
            with colC:
                selected_day = st.number_input("Day:", min_value=1, max_value=31, value= 10)
            with colD:
                selected_hour = st.number_input("Hour (0-23):", min_value=0, max_value=23, value= 16)

            run_predict_btn = st.button("Calculate Probability")

            if run_predict_btn:
                input_vector = prep_data_for_prediction(selected_zipcode, selected_month, selected_day, selected_hour, wavelet_unique_to_use)

                st.subheader("Prediction Results")
                if input_vector is not None:
                    results_map = {}
                    if model_any_loaded: results_map["Any Risk Prob."] = model_any_loaded.predict_proba(input_vector)[0, 1]
                    if model_high_loaded: results_map["High Risk Prob."] = model_high_loaded.predict_proba(input_vector)[0, 1]
                    if model_med_loaded: results_map["Medium Risk Prob."] = model_med_loaded.predict_proba(input_vector)[0, 1]
                    if model_low_loaded: results_map["Low Risk Prob."] = model_low_loaded.predict_proba(input_vector)[0, 1]

                    if results_map:
                        st.success(f"predictions for modzcta `{selected_zipcode}` on `{selected_month}/{selected_day}` hour `{selected_hour}`:")
                        prediction_results_dataframe = pd.DataFrame([results_map])
                        st.dataframe(prediction_results_dataframe.style.format("{:.4f}"))
                    else:
                        st.warning("no models available for prediction or prediction failed.")
                else:
                    st.error("couldn't prepare input data.")

            st.divider()
            st.subheader("DNN Model Info")
            if model_dnn_loaded is None:
                st.info("dnn model (.keras file) wasn't loaded.")
            else:
                st.info("note: the dnn prediction uses different input features (demographic, economic etc.) - prediction interface not built here.")


elif selected_page == "Optimization":
    st.header("Optimized Fire Station Locations")
    st.markdown("Load results from offline analyses first.")
    st.info("parameters used for generating these results (offline): NNEW=2 stations, Weights= H:3/M:2/L:1. live optimization is not feasible.")


    if not st.session_state.opt_loaded:
        if st.button("Load Optimization Results", key="opt_load_btn"):
            st.session_state.opt_dfs['candidate'] = load_csv(POTENTIAL_LOC_FILE)
            st.session_state.opt_dfs['optimal'] = load_csv(OPTIMAL_LOC_FILE)
            st.session_state.opt_dfs['kde_scores'] = load_csv(KDE_ZIP_OUTPUT_PATH)
            st.session_state.opt_loaded = True

    if selected_page == "Optimization" and st.session_state.opt_loaded:
        st.success("Optimization results loaded.")

        candidate_data = st.session_state.opt_dfs.get('candidate')
        optimal_data = st.session_state.opt_dfs.get('optimal')
        kde_data = st.session_state.opt_dfs.get('kde_scores')

        opt_tab1, opt_tab2, opt_tab3 = st.tabs(["Candidates", "GA Optimal", "KDE Scores"])

        with opt_tab1:
            st.subheader("Candidate Potential Locations")
            st.write("These 15 potential spots were shortlisted from high-risk vacant lots using KDE and K-Medoids (via offline notebook).")
            if candidate_data is not None:
                st.dataframe(candidate_data)
                if 'longitude' in candidate_data.columns and 'latitude' in candidate_data.columns:
                     map_dataframe = candidate_data[['latitude', 'longitude']].dropna().rename(columns={'latitude':'lat', 'longitude':'lon'})
                     if not map_dataframe.empty: st.map(map_dataframe, zoom=9)
            else:
                st.warning(f"missing candidate locations csv: `{POTENTIAL_LOC_FILE}`.")

        with opt_tab2:
            st.subheader("Optimal Locations (Genetic Algorithm)")
            st.write("the ga chose these 2 as best additions (result loaded from file).")
            if optimal_data is not None:
                st.dataframe(optimal_data)
                if 'longitude' in optimal_data.columns and 'latitude' in optimal_data.columns:
                     map_dataframe = optimal_data[['latitude', 'longitude']].dropna().rename(columns={'latitude':'lat', 'longitude':'lon'})
                     if not map_dataframe.empty: st.map(map_dataframe, zoom=9)
            else:
                st.warning(f"missing ga results csv: `{OPTIMAL_LOC_FILE}`. run ga script first!")

        with opt_tab3:
            st.subheader("KDE Scores by Zip Code")
            st.write("Shows the calculated fire risk density score per zip code (generated offline by KDE notebook).")
            if kde_data is not None:
                st.dataframe(kde_data.head(15))
                st.caption(f"loaded {len(kde_data)} zip code scores.")
            else:
                 st.warning(f"kde scores file not found: `{KDE_ZIP_OUTPUT_PATH}`. run kde notebook!")
            st.info("the visual kde map image needs to be generated offline and committed separately if desired.")


elif selected_page == "Visualizations":
    st.header("Project Visualizations Dashboard")
    st.markdown("Load relevant data first.")

    if not st.session_state.viz_loaded:
         if st.button("Load Visualization Data", key="viz_load_btn"):
             st.session_state.viz_dfs['candidate'] = load_csv(POTENTIAL_LOC_FILE)
             st.session_state.viz_dfs['optimal'] = load_csv(OPTIMAL_LOC_FILE)
             st.session_state.viz_loaded = True

    if selected_page == "Visualizations" and st.session_state.viz_loaded:
        st.success("Visualization data loaded.")

        candidate_data_viz = st.session_state.viz_dfs.get('candidate')
        optimal_data_viz = st.session_state.viz_dfs.get('optimal')

        st.subheader("Map of Candidate vs Optimal Stations")
        if candidate_data_viz is not None and optimal_data_viz is not None:
             if ('longitude' in candidate_data_viz.columns and 'latitude' in candidate_data_viz.columns and
                 'longitude' in optimal_data_viz.columns and 'latitude' in optimal_data_viz.columns):
                try:
                    pdf_plotting = candidate_data_viz.copy()
                    odf_plotting = optimal_data_viz.copy()
                    pdf_plotting['Type'] = 'Candidate (KDE/KMedoids)'
                    odf_plotting['Type'] = 'Optimal (GA)'
                    plot_dataframe = pd.concat([
                        pdf_plotting[['latitude', 'longitude', 'Type']],
                        odf_plotting[['latitude', 'longitude', 'Type']]
                    ], ignore_index=True).dropna()

                    location_map = px.scatter_mapbox(plot_dataframe,
                                                lat="latitude", lon="longitude", color="Type",
                                                title="Candidate vs Optimal Locations",
                                                mapbox_style="carto-positron", zoom=9.5,
                                                center={"lat": 40.7128, "lon": -74.0060},
                                                opacity= 0.8, size_max=10 )
                    location_map.update_layout(margin={"r":0,"t":50,"l":0,"b":0}, legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
                    st.plotly_chart(location_map, use_container_width=True)
                except Exception as e:
                    st.error(f"couldn't make the location map: {e}")
             else:
                 st.warning("missing lat/lon data for mapping.")
        else:
            st.warning("need both potential and optimal location data loaded for this map.")

        st.divider()

        st.subheader("Power BI Dashboard")
        st.info("""
        Below is a zip file with our PowerBI content as well as instructions for using the custom dashboard.


        We're excited to include this content as an extension of what we learned using Tableau in HW1:
        """)

        if os.path.exists(PBI_ZIP_PATH):
            try:
                with open(PBI_ZIP_PATH, "rb") as fp:
                    btn = st.download_button(
                        label="Download Power BI Content (.zip)",
                        data=fp,
                        file_name="Team_139_PBI_Visualization.zip",
                        mime="application/zip"
                    )
            except Exception as e:
                st.error(f"could not read zip file for download: {e}")
        else:
            st.warning(f"power bi zip file not found: {PBI_ZIP_PATH}")

