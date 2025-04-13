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
