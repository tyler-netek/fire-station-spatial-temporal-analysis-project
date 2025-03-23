# NYC Demographic Data Fetcher

This script gets demographic data for NYC from the ACS (American Community Survey) API. It filters to only NYC ZIP codes, cleans and renames stuff, changes data types, sorts by ZIP, and then saves it as a CSV.

`https://api.census.gov/data/key_signup.html`

Requirements:
- Python 3.x
- You need requests and pandas (dask is optional if you merge large datasets)

Setup:
Put your API key in an environment variable called API_KEY. Install dependencies using pip.

Usage:
Just run the script with: python script.py
