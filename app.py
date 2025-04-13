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
FIRE_INCIDENCE_FILE = os.path.join(".", "Final_Fire_Incidence_Data_with_PopDensity.csv")
ACS_DATA_FILE= os.path.join(".", "ACSST5Y2023.S1901-Data.csv")
POTENTIAL_LOC_FILE = os.path.join(OPTIMIZATION_DATA_DIR, "Potential_location.csv")
OPTIMAL_LOC_FILE = os.path.join(OPTIMIZATION_DATA_DIR, "optimal_ga_locations.csv")
BUILDING_RISK_FILE = os.path.join(PREDICTION_DATA_DIR, "building_fire_risk.csv")


@st.cache_data
def load_csv(file_path):
    if not os.path.exists( file_path ):
        st.warning(f"data file missing: {file_path}. some features might be unavailable.")
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
        if hasattr(loaded_model, 'feature_names_in_'):
            st.session_state[f'{os.path.basename(model_fpath)}_features'] = loaded_model.feature_names_in_
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


wavelet_data_full = load_csv(WAVELET_FEATURES_FILE)
wavelet_features_unique_df = None
zip_list_for_dropdown = []
if wavelet_data_full is not None:
    wavelet_features_unique_df = wavelet_data_full.drop_duplicates(subset=['MODZCTA']).copy()
    if 'MODZCTA' in wavelet_features_unique_df.columns:
        zip_list_for_dropdown = sorted(wavelet_features_unique_df['MODZCTA'].astype(int).unique())
    else:
        st.error("modzcta column not found in wavelet features file!")
else:
     st.error(f"fatal: wavelet file ({WAVELET_FEATURES_FILE}) is missing! predictions broken.")


fire_inc_dataframe = load_csv(FIRE_INCIDENCE_FILE)


any_risk_model_file = os.path.join(PREDICTION_MODEL_DIR,"any_risk_logistic.pkl")
high_risk_model_file = os.path.join(PREDICTION_MODEL_DIR, "high_risk_logistic.pkl")
med_risk_model_file = os.path.join(PREDICTION_MODEL_DIR,"med_risk_logistic.pkl")
low_risk_model_file = os.path.join(PREDICTION_MODEL_DIR, "low_risk_logistic.pkl")
keras_model_file = os.path.join(PREDICTION_MODEL_DIR,"dnn_model.keras")


model_any = load_pkl_model( any_risk_model_file )
model_high = load_pkl_model( high_risk_model_file )
model_med = load_pkl_model( med_risk_model_file )
model_low = load_pkl_model( low_risk_model_file )
model_dnn = load_keras_model( keras_model_file )


candidate_df = load_csv(POTENTIAL_LOC_FILE)
optimal_df = load_csv(OPTIMAL_LOC_FILE)


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
    merged_df_prep = pd.merge( input_df_encoded, wavelet_features, on='MODZCTA', how='left' )

    if merged_df_prep.empty or merged_df_prep.isnull().values.any():
         st.error(f"couldn't find wavelet features for zip {zipcode}? check input file.")
         return None

    processed_df_prep = merged_df_prep.drop('MODZCTA', axis=1)

    ref_model_key = f'{os.path.basename(high_risk_model_file)}_features'
    if ref_model_key in st.session_state:
        required_cols = st.session_state[ref_model_key]
        for col_name in required_cols:
            if col_name not in processed_df_prep.columns:
                processed_df_prep[col_name] = 0
        try:
            processed_df_prep = processed_df_prep[required_cols]
        except KeyError as e:
            st.error(f"column mismatch error: {e}.")
            return None
    else:
        if model_high is not None:
             st.error("can't verify model features. something is wrong.")
        return None

    return processed_df_prep


st.title("🚒 NYC Fire Station Analysis")


st.sidebar.header("Select View")
selected_page = st.sidebar.selectbox(
    "Section:",
    ["Welcome", "Explore Data", "Predict Risk", "Optimization", "Visualizations"]
)
st.sidebar.info(f"Last Refresh: {time.strftime('%Y-%m-%d %H:%M:%S')}")


if selected_page == "Welcome":
    st.header("Welcome")
    st.markdown("""
    This application provides an interactive interface for our CSE6242 project regarding NYC fire station analysis.
    We investigated:
    * Spatial and temporal patterns of fire incidents.
    * Predictive modeling of fire risk using wavelet-based time-series features.
    * Optimization of potential new fire station locations using KDE and Genetic Algorithm techniques.

    Please use the sidebar to navigate the analysis sections.
    """)
    st.info("Team 139: Tyler, Vinuka, Harshitha, Rishi, Madeleine, Kevin")


elif selected_page == "Explore Data":
    st.header("Dataset Overview")
    st.write("Samples from various data files used or generated in the project.")


    dnn_inputs_path = os.path.join(PREDICTION_DATA_DIR, "merged_dnn_input_features.csv")
    pred_outputs_path = os.path.join(PREDICTION_DATA_DIR, "fire_incident_prediction_output.csv")
    demo_data_path = os.path.join(PREDICTION_DATA_DIR, "nyc_demographic_data.csv")
    eco_data_path = os.path.join(PREDICTION_DATA_DIR, "nyc_economic_data.csv")
    bldg_risk_data_path = BUILDING_RISK_FILE
    fdny_stations_path = os.path.join(VISUALIZATIONS_PATH, "FDNY_Firehouse_Listing_20250312.csv")
    all_incidents_path = os.path.join(PREDICTION_DATA_DIR, "FireIncidenceAll.csv")
    high_risk_inc_path = os.path.join(PREDICTION_DATA_DIR, "HighRiskData.csv")
    merged_risk_pop_path = os.path.join(OPTIMIZATION_DATA_DIR, "Merged_Risk_PopDensity_SquareKilometers.csv")
    baseline_coverage_path = os.path.join(OPTIMIZATION_DATA_DIR, "all_zip_full_buffer_coverage_with_overlap.csv")
    kde_zip_output_path = os.path.join(OPTIMIZATION_DATA_DIR, "kde_by_zipcode.csv")
    ga_weights_results_path = os.path.join(OPTIMIZATION_DATA_DIR, "all_weight_combos_results.csv")


    datasets_map = {
        "Final Fire Incidents (w/ Pop Density)": (fire_inc_dataframe, FIRE_INCIDENCE_FILE),
        "Wavelet Features (Logistic Input)": (wavelet_data_full, WAVELET_FEATURES_FILE),
        "Logistic Prediction Output (Generated Offline)": (load_csv(pred_outputs_path), pred_outputs_path),
        "NYC Demographics": (load_csv(demo_data_path), demo_data_path),
        "NYC Economics": (load_csv(eco_data_path), eco_data_path),
        "Building Risk Definition (by Zip)": (load_csv(bldg_risk_data_path), bldg_risk_data_path),
        "Candidate Station Locations (KDE/KMedoids Output)": (candidate_df, POTENTIAL_LOC_FILE),
        "Optimal Station Locations (GA Output)": (optimal_df, OPTIMAL_LOC_FILE),
        "Existing FDNY Stations": (load_csv(fdny_stations_path), fdny_stations_path),
        "Raw Incident Counts (All)": (load_csv(all_incidents_path), all_incidents_path),
        "Raw Incident Counts (High Risk Sample)": (load_csv(high_risk_inc_path), high_risk_inc_path),
        "Merged Risk/Pop (GA Input)": (load_csv(merged_risk_pop_path), merged_risk_pop_path),
        "Baseline Coverage (GA Input)": (load_csv(baseline_coverage_path), baseline_coverage_path),
        "KDE Scores (by Zip Output)": (load_csv(kde_zip_output_path), kde_zip_output_path),
        "GA Weight Combo Results (Offline)": (load_csv(ga_weights_results_path), ga_weights_results_path),
        "Raw ACS Income Data (Sample)": (load_csv(ACS_DATA_FILE), ACS_DATA_FILE)
    }


    for display_name, (current_dataframe_obj, filepath) in datasets_map.items():
        st.subheader(display_name)
        if current_dataframe_obj is not None:
            st.dataframe(current_dataframe_obj.head())
            st.caption(f"shape: {current_dataframe_obj.shape} | source: `{filepath}`")
        else:
            st.caption(f"file not found or load failed: `{filepath}`")
        st.divider()


elif selected_page == "Predict Risk":
    st.header("Predict Fire Incident Probability")
    st.markdown("Select a time and location to estimate the probability of a fire incident within that 4-hour window, based on the trained logistic models using wavelet features.")

    if wavelet_features_unique_df is None:
        st.error("Wavelet features data is required for prediction!")
    else:
        colA, colB = st.columns(2)
        colC, colD = st.columns(2)

        with colA:
            selected_zipcode = st.selectbox("Zip Code (MODZCTA):", zip_list_for_dropdown)
        with colB:
            selected_month = st.selectbox("Month:", list(range(1, 13)), index= 4)
        with colC:
            selected_day = st.number_input("Day:", min_value=1, max_value=31, value= 10)
        with colD:
            selected_hour = st.number_input("Hour (0-23):", min_value=0, max_value=23, value= 16)

        run_predict_btn = st.button("Calculate Probability")

        if run_predict_btn:
            input_vector = prep_data_for_prediction(selected_zipcode, selected_month, selected_day, selected_hour, wavelet_features_unique_df)

            st.subheader("Prediction Results")
            if input_vector is not None:
                results_map = {}
                if model_any: results_map["Any Risk Prob."] = model_any.predict_proba(input_vector)[0, 1]
                if model_high: results_map["High Risk Prob."] = model_high.predict_proba(input_vector)[0, 1]
                if model_med: results_map["Medium Risk Prob."] = model_med.predict_proba(input_vector)[0, 1]
                if model_low: results_map["Low Risk Prob."] = model_low.predict_proba(input_vector)[0, 1]

                if results_map:
                    st.success(f"predictions for modzcta `{selected_zipcode}` on `{selected_month}/{selected_day}` hour `{selected_hour}`:")
                    prediction_results_dataframe = pd.DataFrame([results_map])
                    st.dataframe(prediction_results_dataframe.style.format("{:.4f}"))
                else:
                    st.warning("no models loaded or prediction failed.")
            else:
                st.error("couldn't prepare input data.")

        st.divider()
        st.subheader("DNN Model Info")
        if model_dnn is None:
            st.info("dnn model (.keras file) wasn't loaded.")
        else:
            st.info("note: the dnn prediction uses different input features (demographic, economic etc.) - prediction interface not built here.")


elif selected_page == "Optimization":
    st.header("Optimized Fire Station Locations")
    st.markdown("Displays candidate and optimal locations based on offline analyses.")

    opt_tab1, opt_tab2, opt_tab3 = st.tabs(["Candidates", "GA Optimal", "KDE Scores"])

    with opt_tab1:
        st.subheader("Candidate Potential Locations")
        st.write("These 15 potential spots were shortlisted from high-risk vacant lots using KDE and K-Medoids (via offline notebook).")
        if candidate_df is not None:
            st.dataframe(candidate_df)
            if 'longitude' in candidate_df.columns and 'latitude' in candidate_df.columns:
                 map_dataframe = candidate_df[['latitude', 'longitude']].dropna().rename(columns={'latitude':'lat', 'longitude':'lon'})
                 if not map_dataframe.empty: st.map(map_dataframe, zoom=9)
        else:
            st.warning(f"missing candidate locations csv: `{POTENTIAL_LOC_FILE}`.")

    with opt_tab2:
        st.subheader("Optimal Locations (Genetic Algorithm)")
        st.write("the ga chose these 2 as best additions (result loaded from file).")
        if optimal_df is not None:
            st.dataframe(optimal_df)
            if 'longitude' in optimal_df.columns and 'latitude' in optimal_df.columns:
                 map_dataframe = optimal_df[['latitude', 'longitude']].dropna().rename(columns={'latitude':'lat', 'longitude':'lon'})
                 if not map_dataframe.empty: st.map(map_dataframe, zoom=9)
        else:
            st.warning(f"missing ga results csv: `{OPTIMAL_LOC_FILE}`. run ga script first!")

    with opt_tab3:
        st.subheader("KDE Scores by Zip Code")
        st.write("Shows the calculated fire risk density score per zip code (generated offline by KDE notebook).")
        kde_zip_output_path = os.path.join(OPTIMIZATION_DATA_DIR, "kde_by_zipcode.csv")
        kde_zip_df = load_csv(kde_zip_output_path)
        if kde_zip_df is not None:
            st.dataframe(kde_zip_df.head(15))
            st.caption(f"loaded {len(kde_zip_df)} zip code scores.")
        else:
             st.warning(f"kde scores file not found: `{kde_zip_output_path}`. run kde notebook!")
        st.info("the visual kde map image needs to be generated offline and committed separately if desired.")


elif selected_page == "Visualizations":
    st.header("Project Visualizations Dashboard")

    st.subheader("Map of Candidate vs Optimal Stations")
    if candidate_df is not None and optimal_df is not None:
         if ('longitude' in candidate_df.columns and 'latitude' in candidate_df.columns and
             'longitude' in optimal_df.columns and 'latitude' in optimal_df.columns):
            try:
                pdf_plotting = candidate_df.copy()
                odf_plotting = optimal_df.copy()
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
    st.subheader("Add Custom Visualizations")
    st.info("this section is available for adding further project visualizations by editing the script.")
