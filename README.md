# NYC Fire Station Analysis - CSE6242 Project

**Team:** Team 139
**Members:** Tyler, Vinuka, Harshitha, Rishi, Madeleine, Kevin

## What's This About?

This project is for Georgia Tech's Data and Visual Analytics class (Spring 2025). We looked at fire incidents across NYC, tried to predict when/where they might happen using some time-series (wavelet) features, and explored where new fire stations could potentially go using optimization methods like KDE and a Genetic Algorithm. This repo has the code, notebooks, and the interactive Streamlit app we built.

## Live App Demo

Check out the interactive dashboard deployed on Render! (Easiest way to see the results)

- URL

Heads up: The free version might take a minute to wake up if it's been sleeping.

## App Features

* **Explore Data:** Get a quick look at the fire incident data and the wavelet features used for prediction.
* **Predict Risk:** Pick a zip code, date, and time, and see the predicted probability of a fire incident based on our logistic models.
* **Optimization:** See the 15 candidate locations we shortlisted and the 2 locations picked by the genetic algorithm as potentially optimal spots. Also includes the KDE risk map visual.
* **Visualizations:** Some extra plots showing incident counts and comparing the station locations on a map.

## Tech We Used

Python 3.9, Streamlit, Pandas, NumPy, Scikit-learn, Joblib, TensorFlow/Keras, Geopandas, Shapely, DEAP (for GA), Scikit-learn-extra (for KMedoids), Plotly, Matplotlib, Cartopy, Pipenv for environment management, and Render for deployment. Whew!

## Getting it Running Locally

Want to run this on your own machine? Here's how:

**1. Stuff You Need First:**
   * Python (version 3.9 worked for us) - Make sure it's in your PATH.
   * Pipenv (`pip install pipenv` or `pip3 install pipenv`)
   * Maybe some system stuff for the map libraries (`geopandas`, `cartopy`). If `pipenv install` complains about GDAL, PROJ, or GEOS:
      * Ubuntu/Debian: Try `sudo apt-get update && sudo apt-get install -y libgdal-dev gdal-bin libproj-dev proj-bin libgeos-dev`
      * Mac (Homebrew): Try `brew install gdal proj geos`
      * Windows: Good luck! Check the GeoPandas/Cartopy docs, might need specific installers or wheels.

**2. Grab the Code:**
   ```bash
    git clone [https://github.com/tyler-netek/fire-station-spatial-temporal-analysis-project.git](https://github.com/tyler-netek/fire-station-spatial-temporal-analysis-project.git)
    cd fire-station-spatial-temporal-analysis-project
   ```

**3. Install Packages:**
    This uses the `Pipfile.lock` to get the exact versions we used. Might take a bit.
   ```bash
    pipenv install
   ```

**4. Data Files & Models:**
   * Most `.csv`, `.pkl`, `.keras` files needed by `app.py` should be in the repo. Make sure `Prediction/Data/wavelet_features.csv` is present.
   * **Big Files (PLUTO):** You *need* the NYC PLUTO dataset (`pluto_25v1.csv` seems to be the version used) to run the `Optimization/KDE+...ipynb` notebook fully (which generates `Potential_location.csv` and the KDE map). It's too big for Git. You'll have to download it from the NYC Planning website ([find the link!]) and put it in `Optimization/Data/`.

**5. Run the Offline Stuff (Important!):**
   The app shows results *from* these scripts/notebooks. Run them first.
   * Activate the environment: `pipenv shell`
   * Generate the Genetic Algorithm optimal locations:
   `
      python Optimization/Genetic_algo_final.py`
      (Check that this *actually saves* the results to `Optimization/Data/optimal_ga_locations.csv`. You might need to add a line like `result_df.to_csv("Optimization/Data/optimal_ga_locations.csv", index=False)` to the end of the script if it only prints.)
   * Generate the KDE map visual:
      * Run the `Optimization/KDE+ New Fire Station Locations(K Medoids).ipynb` notebook (e.g., `jupyter lab` inside the `pipenv shell`).
      * Make sure the part that shows the plot also *saves* it as `Optimization/kde_potential_stations_map.png`.
   * You can type `exit` to leave the shell when done.

**6. Run the App:**
   * Start the virtual environment again:
      ```bash
      pipenv shell
      ```
   * Launch Streamlit:
      ```bash
      streamlit run app.py
      ```
   * Should open in your browser (usually `http://localhost:8501`).

## Deployment Notes (Render)

We set this up to deploy on Render's free Python tier.
* Uses `Pipfile` / `Pipfile.lock`.
* Render Build Command: `pip install pipenv && pipenv install --system --deploy --ignore-pipfile`
* Render Start Command: `streamlit run app.py --server.port $PORT --server.headless true`

---