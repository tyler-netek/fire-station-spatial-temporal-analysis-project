# NYC Fire Incident Analysis and Prediction

This project uses open data from New York City to analyze fire incidents and predict risk. The data is combined from multiple sources and processed to extract features using wavelet analysis. These features help improve the understanding of fire dynamics over time and can later be used in predictive models.

## Data Sources

- **Fire Incident Data (Temporal):**  
  - HighRiskData_Temporal.csv  
  - MediumRiskData_Temporal_1.csv  
  - MediumRiskData_Temporal_2.csv  
  - LowRiskData_Temporal_1.csv  
  - LowRiskData_Temporal_2.csv  

- **Building Data:** Information on building fire risk per ZIP code.  
- **Demographic Data:** Basic demographic information for each ZIP code.  
- **Economic Data:** Economic indicators such as household income per ZIP code.

## Process

1. **Data Cleaning and Merging:**  
   Each dataset is cleaned (e.g., removing unnecessary columns, correcting data types) and merged on common fields (like ZIP codes).

2. **Wavelet Feature Extraction:**  
   - For each ZIP code, daily fire incident counts are obtained from the temporal fire incident files.  
   - A Continuous Wavelet Transform (using the Morlet wavelet) is applied to these time series.  
   - Three features are extracted for each risk level (high, medium, low):  
     - **Mean Energy:** The average intensity of the incident patterns.  
     - **Max Energy:** The peak intensity observed.  
     - **Dominant Scale:** The time window scale at which the maximum energy is reached.
   - The extracted features are combined into a CSV file (`wavelet_features.csv`) for later use in predictive models.

3. **Visualization:**  
   Histograms and scalograms are generated to visualize the distribution of the wavelet features across ZIP codes. This step helps in understanding how these features vary across different areas.

4. **Modeling:**  
   The wavelet features will later be merged with other datasets (building, demographic, and economic data) to train machine learning models that predict fire incident risk and support decisions like optimal fire station placements.

## Interpreting the Features

- **high_mean_energy / med_mean_energy / low_mean_energy:**  
  Represents the overall average “energy” or intensity of fire incidents over time for that ZIP code.

- **high_max_energy / med_max_energy / low_max_energy:**  
  Indicates the highest burst of fire incident intensity observed. A higher value means a stronger incident spike.

- **high_dominant_scale / med_dominant_scale / low_dominant_scale:**  
  The scale (i.e., the time window) at which the maximum energy occurs. For high-risk fires, a lower dominant scale might be worse because it indicates very rapid, intense fire activity. In contrast, for medium and low risk, a higher dominant scale might indicate prolonged fire events which could also be damaging.
