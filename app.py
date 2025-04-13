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

st.set_page_config(layout="wide", page_title="NYC Fire Risk")

OPTIMIZATION_DATA_DIR = os.path.join(".", "Optimization", "Data")
PREDICTION_DATA_DIR = os.path.join(".", "Prediction", "Data")
PREDICTION_MODEL_DIR = os.path.join(".", "Prediction", "Models")
VISUALIZATIONS_DIR = os.path.join(".", "Visualizations")
OPTIMIZATION_VIS_DIR = os.path.join(".", "Optimization")
WAVELET_FEATURES_PATH = os.path.join(".", "Prediction", "wavelet_features.csv")


@st.cache_data
def load_csv(file_path):
    if not os.path.exists(file_path):
        st.warning(f"Data file missing: {file_path}. App might lack some features.")
        return None
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        st.error(f"Couldn't load {file_path}: {e}")
        return None

@st.cache_resource
def load_pkl_model(model_path):
    if not os.path.exists(model_path):
        st.warning(f"Model file missing: {model_path}.")
        return None
    try:
        model = joblib.load(model_path)
        if hasattr(model, 'feature_names_in_'):
            st.session_state[f'{os.path.basename(model_path)}_features'] = model.feature_names_in_
        return model
    except Exception as e:
        st.error(f"Problem loading sklearn model {model_path}: {e}")
        return None

@st.cache_resource
def load_keras_model(model_path):
    if not os.path.exists(model_path):
        st.warning(f"Model file missing: {model_path}.")
        return None
    try:
        with st.spinner(f'Loading DNN model ({os.path.basename(model_path)})...'):
            model = load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Problem loading Keras model {model_path}: {e}")
        return None


wavelet_df = load_csv(WAVELET_FEATURES_PATH)
wavelet_features_unique = None
available_zips = []
if wavelet_df is not None:
    wavelet_features_unique = wavelet_df.drop_duplicates(subset=['MODZCTA']).copy()
    if 'MODZCTA' in wavelet_features_unique.columns:
        available_zips = sorted(wavelet_features_unique['MODZCTA'].astype(int).unique())
    else:
        st.error("MODZCTA column not found in wavelet features file!")
else:
     st.error(f"Fatal: Wavelet features file ({WAVELET_FEATURES_PATH}) is missing. Predictions won't work.")


fire_inc_data_path = os.path.join(".", "Final_Fire_Incidence_Data_with_PopDensity.csv")
fire_inc_df = load_csv(fire_inc_data_path)


any_risk_model_path = os.path.join(PREDICTION_MODEL_DIR, "any_risk_logistic.pkl")
high_risk_model_path = os.path.join(PREDICTION_MODEL_DIR, "high_risk_logistic.pkl")
med_risk_model_path = os.path.join(PREDICTION_MODEL_DIR, "med_risk_logistic.pkl")
low_risk_model_path = os.path.join(PREDICTION_MODEL_DIR, "low_risk_logistic.pkl")
keras_model_path = os.path.join(PREDICTION_MODEL_DIR, "dnn_model.keras")

any_risk_model = load_pkl_model(any_risk_model_path)
high_risk_model = load_pkl_model(high_risk_model_path)
med_risk_model = load_pkl_model(med_risk_model_path)
low_risk_model = load_pkl_model(low_risk_model_path)
dnn_keras_model = load_keras_model(keras_model_path)


pot_loc_path = os.path.join(OPTIMIZATION_DATA_DIR, "Potential_location.csv")
opt_loc_path = os.path.join(OPTIMIZATION_DATA_DIR, "optimal_ga_locations.csv")
potential_locs_df = load_csv(pot_loc_path)
optimal_locs_df = load_csv(opt_loc_path)


kde_map_path = os.path.join(OPTIMIZATION_VIS_DIR, "kde_potential_stations_map.png")
dnn_perf_plot_path = os.path.join(PREDICTION_MODEL_DIR, "dnn_performance_scatter.png")


def prep_data_for_prediction(zipcode, month, day, hour, wavelet_feats):
    if wavelet_feats is None: return None
    try:
        pd.Timestamp(year=2024, month=month, day=day)
    except ValueError:
         st.error(f"Looks like an invalid date: {month}/{day}")
         return None

    hr_group = (hour // 4) * 4
    input_row = pd.DataFrame({'MODZCTA': [zipcode], 'Month': [month], 'Date': [day], 'Hour_group': [hr_group]})
    input_row_dummies = pd.get_dummies(input_row, columns=['Month', 'Date', 'Hour_group'])
    merged_input = pd.merge(input_row_dummies, wavelet_feats, on='MODZCTA', how='left')

    if merged_input.empty or merged_input.isnull().values.any():
         st.error(f"Couldn't find wavelet features for Zip {zipcode}? Check input file.")
         return None

    processed_input = merged_input.drop('MODZCTA', axis=1)

    ref_model_key = f'{os.path.basename(high_risk_model_path)}_features'
    if ref_model_key in st.session_state:
        model_cols = st.session_state[ref_model_key]
        for col in model_cols:
            if col not in processed_input.columns:
                processed_input[col] = 0
        try:
            processed_input = processed_input[model_cols]
        except KeyError as e:
            st.error(f"Column mismatch error: {e}. Model expected different features than generated.")
            return None
    else:
        st.error("Can't find the expected feature list from the loaded model. Cannot proceed.")
        return None

    return processed_input


st.title("🚒 NYC Fire Station Analysis")

st.sidebar.header("Select View")
current_section = st.sidebar.selectbox(
    "Section:",
    ["Welcome", "Explore Data", "Predict Risk", "Optimization", "Visualizations"]
)
st.sidebar.info(f"Last Refresh: {time.strftime('%Y-%m-%d %H:%M:%S')}")


if current_section == "Welcome":
    st.header("Welcome!")
    st.markdown("""
    Hey there! This is an interactive look at our CSE6242 project on NYC fire stations.
    We tried to figure out:
    * Where and when fires happen most often.
    * If we can predict fire risk based on time and location features (using wavelet stuff!).
    * Where good spots for new fire stations might be, using fancy methods like KDE and Genetic Algorithms.

    Use the sidebar to poke around the different parts.
    """)
    st.info("Team 139: Tyler, Vinuka, Harshitha, Rishi, Madeleine, Kevin")

elif current_section == "Explore Data":
    st.header("A Peek at the Data")
    st.write("Quick samples of the data used.")

    st.subheader("Fire Incidents (Snapshot)")
    if fire_inc_df is not None:
        st.dataframe(fire_inc_df.sample(min(5, len(fire_inc_df))))
        st.caption(f"Loaded {len(fire_inc_df)} rows.")
    else:
        st.warning("Couldn't load the main fire incidence data.")

    st.subheader("Wavelet Features (Sample)")
    if wavelet_df is not None:
        st.dataframe(wavelet_df.head())
        st.caption(f"Loaded {len(wavelet_df)} rows from {WAVELET_FEATURES_PATH}.")
    else:
        st.warning(f"Couldn't load wavelet features data from {WAVELET_FEATURES_PATH}.")


elif current_section == "Predict Risk":
    st.header("Predict Fire Incident Probability")
    st.markdown("Pick a time and place to see the predicted probability of a fire incident happening in that 4-hour window, according to our logistic models.")

    if wavelet_features_unique is None:
        st.error("Need the wavelet features data to make predictions!")
    else:
        col1, col2 = st.columns(2)
        col3, col4 = st.columns(2)

        with col1:
            zip_input = st.selectbox("Zip Code (MODZCTA):", available_zips)
        with col2:
            month_input = st.selectbox("Month:", list(range(1, 13)), index=3)
        with col3:
            day_input = st.number_input("Day:", min_value=1, max_value=31, value=13)
        with col4:
            hour_input = st.number_input("Hour (0-23):", min_value=0, max_value=23, value=15)

        predict_button = st.button("Run Prediction")

        if predict_button:
            input_ready = prep_data_for_prediction(zip_input, month_input, day_input, hour_input, wavelet_features_unique)

            st.subheader("Prediction Results")
            if input_ready is not None:
                predictions = {}
                try:
                    if any_risk_model: predictions["Any Risk Prob."] = any_risk_model.predict_proba(input_ready)[0, 1]
                except Exception as e: st.error(f"Any Risk Model Error: {e}")
                try:
                    if high_risk_model: predictions["High Risk Prob."] = high_risk_model.predict_proba(input_ready)[0, 1]
                except Exception as e: st.error(f"High Risk Model Error: {e}")
                try:
                    if med_risk_model: predictions["Medium Risk Prob."] = med_risk_model.predict_proba(input_ready)[0, 1]
                except Exception as e: st.error(f"Medium Risk Model Error: {e}")
                try:
                    if low_risk_model: predictions["Low Risk Prob."] = low_risk_model.predict_proba(input_ready)[0, 1]
                except Exception as e: st.error(f"Low Risk Model Error: {e}")


                if predictions:
                    st.success(f"Predictions for MODZCTA `{zip_input}` on `{month_input}/{day_input}` around hour `{hour_input}`:")
                    pred_results_df = pd.DataFrame([predictions])
                    st.dataframe(pred_results_df.style.format("{:.4f}"))
                else:
                    st.warning("No logistic models loaded or prediction failed.")
            else:
                st.error("Couldn't prepare data for prediction based on input.")

        st.markdown("---")
        st.subheader("DNN Model")
        if dnn_keras_model is None:
            st.info("DNN model (.keras file) wasn't loaded.")
        else:
            st.info("DNN prediction uses different input features (demographic, economic etc.) and requires separate UI and preprocessing logic (not implemented here).")


elif current_section == "Optimization":
    st.header("Finding Better Fire Station Spots")
    st.markdown("Results from offline KDE, K-Medoids, and Genetic Algorithm runs.")

    tab_kmed, tab_ga, tab_kde_viz = st.tabs(["Candidates", "GA Optimal", "KDE Map"])

    with tab_kmed:
        st.subheader("Candidate Potential Locations")
        st.write("These 15 potential spots were shortlisted from high-risk vacant lots.")
        if potential_locs_df is not None:
            st.dataframe(potential_locs_df)
            if 'longitude' in potential_locs_df.columns and 'latitude' in potential_locs_df.columns:
                 map_df = potential_locs_df[['latitude', 'longitude']].dropna().rename(columns={'latitude':'lat', 'longitude':'lon'})
                 if not map_df.empty: st.map(map_df, zoom=10)
        else:
            st.warning(f"Missing the potential locations CSV from `{pot_loc_path}`.")

    with tab_ga:
        st.subheader("Optimal Locations (from Genetic Algorithm)")
        st.write("The GA picked these 2 spots as the best additions based on coverage.")
        if optimal_locs_df is not None:
            st.dataframe(optimal_locs_df)
            if 'longitude' in optimal_locs_df.columns and 'latitude' in optimal_locs_df.columns:
                 map_df = optimal_locs_df[['latitude', 'longitude']].dropna().rename(columns={'latitude':'lat', 'longitude':'lon'})
                 if not map_df.empty: st.map(map_df, zoom=10)
        else:
            st.warning(f"Missing the GA results CSV from `{opt_loc_path}`. Run the script first!")

    with tab_kde_viz:
        st.subheader("KDE Fire Risk Map")
        st.write("Shows calculated fire risk density and the 15 candidates.")
        if os.path.exists(kde_map_path):
            st.image(kde_map_path, caption="KDE Risk + Candidate Locations (Generated Offline)", use_column_width=True)
        else:
             st.warning(f"Map image `{kde_map_path}` not found. Generate it from the KDE notebook.")


elif current_section == "Visualizations":
    st.header("Project Visualizations Dashboard")

    viz_tab1, viz_tab2, viz_tab3, viz_tab4 = st.tabs([
        "Incident Counts", "Location Map", "Wavelet Features", "DNN Performance"
    ])

    with viz_tab1:
        st.subheader("Incident Counts by Zip (Top 30)")
        if fire_inc_df is not None and 'MODZCTA' in fire_inc_df.columns:
            try:
                counts = fire_inc_df['MODZCTA'].value_counts().reset_index().head(30)
                counts.columns = ['MODZCTA', 'Count']
                counts['MODZCTA'] = counts['MODZCTA'].astype(str)
                fig_hist = px.bar(counts, x='MODZCTA', y='Count', title='Top 30 Zip Codes by Incident Count')
                st.plotly_chart(fig_hist, use_container_width=True)
            except Exception as e:
                st.error(f"Couldn't make the histogram: {e}")
        else:
            st.warning("Need fire incidence data with 'MODZCTA' column for this plot.")

    with viz_tab2:
        st.subheader("Map of Candidate vs Optimal Stations")
        if potential_locs_df is not None and optimal_locs_df is not None:
             if ('longitude' in potential_locs_df.columns and 'latitude' in potential_locs_df.columns and
                 'longitude' in optimal_locs_df.columns and 'latitude' in optimal_locs_df.columns):
                try:
                    pdf = potential_locs_df.copy()
                    odf = optimal_locs_df.copy()
                    pdf['Type'] = 'Candidate (KDE/KMedoids)'
                    odf['Type'] = 'Optimal (GA)'
                    combined_df = pd.concat([
                        pdf[['latitude', 'longitude', 'Type']],
                        odf[['latitude', 'longitude', 'Type']]
                    ], ignore_index=True).dropna()

                    fig_map = px.scatter_mapbox(combined_df,
                                                lat="latitude", lon="longitude", color="Type",
                                                title="Candidate vs Optimal Locations",
                                                mapbox_style="carto-positron", zoom=9.5,
                                                center={"lat": 40.7128, "lon": -74.0060},
                                                opacity=0.8, size_max=10)
                    fig_map.update_layout(margin={"r":0,"t":50,"l":0,"b":0}, legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
                    st.plotly_chart(fig_map, use_container_width=True)
                except Exception as e:
                    st.error(f"Couldn't make the location map: {e}")
             else:
                 st.warning("Missing lat/lon data for mapping.")
        else:
            st.warning("Need both potential and optimal location data loaded for this map.")

    with viz_tab3:
        st.subheader("Wavelet Feature Distributions")
        if wavelet_df is not None:
            try:
                wavelet_cols = [col for col in wavelet_df.columns if col not in ['MODZCTA']]
                if wavelet_cols:
                    fig_wavelet, axes = plt.subplots(3, 3, figsize=(12, 10))
                    axes = axes.ravel()
                    cols_to_plot = wavelet_cols[:len(axes)]
                    for i, col in enumerate(cols_to_plot):
                        axes[i].hist(wavelet_df[col].dropna(), bins=20)
                        axes[i].set_title(col, fontsize=10)
                        axes[i].tick_params(axis='x', rotation=30, labelsize=8)
                        axes[i].tick_params(axis='y', labelsize=8)

                    for i in range(len(cols_to_plot), len(axes)):
                        axes[i].set_visible(False)

                    plt.tight_layout(pad=2.0)
                    st.pyplot(fig_wavelet)
                    plt.close(fig_wavelet) # Close figure to free memory
                else:
                    st.warning("No wavelet feature columns found to plot.")

            except Exception as e:
                st.error(f"Could not generate wavelet histograms: {e}")
        else:
            st.warning("Wavelet features data not loaded.")

    with viz_tab4:
        st.subheader("DNN Model Performance (Actual vs. Predicted)")
        st.write("This plot shows performance on the test set for the DNN model predicting weighted incident counts (generated offline).")
        if os.path.exists(dnn_perf_plot_path):
             st.image(dnn_perf_plot_path, caption="DNN Actual vs. Predicted Weighted Incidents", use_column_width=True)
        else:
             st.warning(f"DNN performance plot not found at `{dnn_perf_plot_path}`. Generate and save it from the model development notebook.")

        st.info("Note: The DNN uses different features (demographic, economic, etc.) than the logistic models shown in the 'Predict Risk' tab.")
        