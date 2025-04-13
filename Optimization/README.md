FireIncidenceKDE.ipynb- Cleans up the fire incidence data for use in Prediction models, but does not include extracting temporal features.

FireIncidenceTemporal.ipynb- Cleans up the fire incidence data and extracts temporal features for use in Prediction models/Morelet Wavelet Analysis.

Genetic_algo_final.py- Runs the Genetic Algorithm off of three datasets in the Data file. Gives the locations of the new optimal new fire stations in the form of a dataframe with their longitude and latitude to be transfered to the visualization. 
  Datasets needed: 
  1)  Merged_Risk_PopDensity_SquareKilometers.csv
  2)  Potential_location.csv
  3)  all_zip_full_buffer_coverage_with_overlap.csv
optimization_circle_areas.py was used to generate all_zip_full_buffer_coverage_with_overlap.csv for basline coverage values that were improved on. 
