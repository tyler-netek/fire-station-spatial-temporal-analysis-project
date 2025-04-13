# NYC Fire Station Analysis - CSE6242 Project
- --
**Team:** Team 139
**Members:** Tyler, Vinuka, Harshitha, Rishi, Madeleine, Kevin

## What's This About?

This project is for GT's Data and Visual Analytics class (Spring 2025). We looked at NYC fire incidents, tried predicting risk using time features (wavelets!), and explored where new fire stations could go using optimization (KDE, Genetic Algorithms). This repo has our code and this Streamlit app.

## Live App Demo

Check out the interactive dashboard deployed on Render!

[https://fire-station-spatial-temporal-analysis.onrender.com](https://fire-station-spatial-temporal-analysis.onrender.com)

(The free app might take a minute to wake up if it hasn't been used recently.)

## App Features

* **Explore Data:** Look at fire incident and wavelet feature data samples.
* **Predict Risk:** Get fire incident probability for a specific zip/time using our logistic models.
* **Optimization:** See candidate and optimal new station locations, plus the KDE risk map.
* **Visualizations:** Extra plots showing incident counts and station locations.

## Tech We Used

Python 3.11, Streamlit, Pandas, Scikit-learn, TensorFlow/Keras, Geopandas, DEAP, Pipenv, Render, and others.

## Getting it Running Locally

**1. Get Ready:**
   * Need Git and Python 3.11 (added to PATH).
   * Install Pipenv: `pip install pipenv`.
   * Might need system libraries for `geopandas`/`cartopy` if install fails (GDAL, PROJ, GEOS - check their docs for your OS).

**2. Clone the Code:**
   ```bash
   git clone [https://github.com/tyler-netek/fire-station-spatial-temporal-analysis-project.git](https://github.com/tyler-netek/fire-station-spatial-temporal-analysis-project.git)
   cd fire-station-spatial-temporal-analysis-project
   ```

**3. Install Packages:**
   ```bash
   pipenv install
   ```
   (Uses `Pipfile.lock`. Can take a while.)

**4. Data Files:**
   * Most data/models should be in the repo. Make sure `Prediction/wavelet_features.csv` is present.
   * **PLUTO Data:** The KDE notebook needs `pluto_25v1.csv`. Download it from NYC Planning ([Link Needed!]) and put it in `Optimization/Data/`.

**5. Run Offline Stuff (Required!):**
   The app needs results *from* these. Run them first inside the environment (`pipenv shell`).
   * **Generate GA Results:**
      ```bash
      python Optimization/Genetic_algo_final.py
      ```
      (This needs to save output to `Optimization/Data/optimal_ga_locations.csv`. Add the `.to_csv()` line to the python script if needed.)
   * **Generate KDE Map Image:**
      * Run the `Optimization/KDE+ New Fire Station Locations(K Medoids).ipynb` notebook.
      * Make sure it saves the KDE map plot as `Optimization/kde_potential_stations_map.png`.
   * Type `exit` when done.

**6. Run the App:**
   ```bash
   pipenv shell
   streamlit run app.py
   ```
   Check `http://localhost:8501` in your browser.

## Deployment Notes (Render)

Deployed on Render's free tier.
* Build: `pip install pipenv && pipenv install --system --deploy --ignore-pipfile`
* Start: `streamlit run app.py --server.port $PORT --server.headless true`

---