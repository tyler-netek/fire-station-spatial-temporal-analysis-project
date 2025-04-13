FireIncidenceKDE.ipynb- Cleans up the fire incidence data for use in Prediction models, but does not include extracting temporal features.

FireIncidenceTemporal.ipynb- Cleans up the fire incidence data and extracts temporal features for use in Prediction models/Morelet Wavelet Analysis.

Genetic_algo_final.py- Runs the Genetic Algorithm off of three datasets in the Data file. Gives the locations of the new optimal new fire stations in the form of a dataframe with their longitude and latitude to be transfered to the visualization. 
  Datasets needed: 
  1)  Merged_Risk_PopDensity_SquareKilometers.csv
  2)  Potential_location.csv
  3)  all_zip_full_buffer_coverage_with_overlap.csv

optimization_circle_areas.py- was used to generate all_zip_full_buffer_coverage_with_overlap.csv for basline coverage values that were improved on. 

Optimization/KDE+ New Fire Station Locations(K Medoids).ipynb- Contains KDE analysis for fire risk(quantified as noted in the report), generates Potential Station.csv, which contains a list of 15 stations shortlisted as noted in the report and visualizations showing where exactly these fire stations are located on the KDE map.
Datasets needed:
1)building_fire_risk.csv(in the data folder)
2)pluto_25v1.csv(need to upload)
3)Final_Fire_Incidence_Data_with_PopDensity.csv(in the data folder),
