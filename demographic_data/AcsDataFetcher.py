import os
import requests
import pandas as pd

class AcsDataFetcher:
    def __init__(self, api_key=None):
        if api_key is None:
            self.api_key = os.getenv("API_KEY")
        else:
            self.api_key = api_key
        if not self.api_key:
            raise Exception("API key missing")
        self.year = "2020"
        self.survey = "acs/acs5"
        self.geo = "zip code tabulation area:*"
        self.base_url = f"https://api.census.gov/data/{self.year}/{self.survey}"
        self.var_list = [
            "NAME", "B01003_001E", "B01002_001E", "B01001_002E", "B01001_026E",
            "B02001_002E", "B02001_003E", "B02001_004E", "B02001_005E",
            "B02001_006E", "B02001_007E", "B02001_008E", "B19013_001E",
            "B17001_001E", "B17001_002E", "B15003_001E", "B15003_017E",
            "B15003_022E", "B25077_001E", "B25064_001E", "B25001_001E",
            "B25003_002E", "B25003_003E", "B23025_001E", "B23025_002E",
            "B23025_003E", "B23025_005E", "B08303_001E", "B18101_001E",
            "B27010_012E", "B25044_001E", "B16001_001E", "B05002_013E",
            "B20004_001E", "B28002_004E", "B25035_001E"
        ]
        self.nyc_zips = [
            "10001", "10002", "10003", "10004", "10005", "10006", "10007", "10009",
            "10010", "10011", "10012", "10013", "10014", "10016", "10017", "10018",
            "10019", "10021", "10022", "10023", "10024", "10025", "10026",
            "10027", "10028", "10029", "10030", "10031", "10032", "10033", "10034",
            "10035", "10036", "10037", "10038", "10039", "10040", "10044",
            "10104", "10105", "10106", "10107", "10113",
            "10114", "10116", "10117", "10118", "10120", "10121",
            "10122", "10123", "10124", "10125", "10126", "10128", "10280", "10282",
            "10451", "10452", "10453", "10454", "10455", "10456", "10457", "10458",
            "10459", "10460", "10461", "10462", "10463", "10464", "10465", "10466",
            "10467", "10468", "10469", "10470", "10471", "10472", "10473", "10474",
            "10475",
            "11201", "11203", "11204", "11205", "11206", "11207", "11208", "11209",
            "11210", "11211", "11212", "11213", "11214", "11215", "11216", "11217",
            "11218", "11219", "11220", "11221", "11222", "11223", "11224", "11225",
            "11226", "11228", "11229", "11230", "11231", "11232", "11233", "11234",
            "11235", "11236", "11237", "11238", "11239", "11240", "11241", "11242",
            "11243", "11244", "11249",
            "11101", "11102", "11103", "11104", "11105", "11106", "11109",
            "11352", "11354", "11355", "11356", "11357", "11358", "11360",
            "11361", "11362", "11363", "11364", "11365", "11366", "11367", "11368",
            "11369", "11370", "11372", "11373", "11374", "11375", "11377",
            "11378", "11379", "11385", "11411", "11412", "11413", "11414", "11415",
            "11416", "11691", "11692", "11693", "11694", "11695", "11697",
            "10301", "10302", "10303", "10304", "10305", "10306", "10307", "10308",
            "10309", "10310", "10312", "10314"
        ]
        self.rename_map = {
            "B01003_001E": "Total_Population",
            "B01002_001E": "Median_Age",
            "B01001_002E": "Male_Population",
            "B01001_026E": "Female_Population",
            "B02001_002E": "White",
            "B02001_003E": "Black",
            "B02001_004E": "American_Indian_Alaska_Native",
            "B02001_005E": "Asian",
            "B02001_006E": "Native_Hawaiian_Pacific_Islander",
            "B02001_007E": "Other_Race",
            "B02001_008E": "Two_or_More_Races",
            "B19013_001E": "Median_Household_Income",
            "B17001_001E": "Poverty_Total",
            "B17001_002E": "Below_Poverty",
            "B15003_001E": "Total_Educ_25plus",
            "B15003_017E": "Bachelor_Degree",
            "B15003_022E": "Advanced_Degree",
            "B25077_001E": "Median_Home_Value",
            "B25064_001E": "Median_Gross_Rent",
            "B25001_001E": "Total_Housing_Units",
            "B25003_002E": "Owner_Occupied",
            "B25003_003E": "Renter_Occupied",
            "B23025_001E": "Total_Labor_Force",
            "B23025_002E": "Not_in_Labor_Force",
            "B23025_003E": "Employed",
            "B23025_005E": "Unemployed",
            "B08303_001E": "Median_Travel_Time",
            "B18101_001E": "Total_With_Disability",
            "B27010_012E": "No_Health_Insurance",
            "B25044_001E": "Vehicles_Available",
            "B16001_001E": "Total_Language_Population",
            "B05002_013E": "Foreign_Born",
            "B20004_001E": "Median_Earnings",
            "B28002_004E": "Households_with_Broadband",
            "B25035_001E": "Median_Year_Built"
        }
        self.params = {"get": ",".join(self.var_list), "for": self.geo, "key": self.api_key}
        self.df = None

    def FetchData(self):
        r = requests.get(self.base_url, params=self.params, timeout=30)
        if r.status_code != 200:
            raise Exception("API request failed with status code " + str(r.status_code))
        data = r.json()
        self.df = pd.DataFrame(data[1:], columns=data[0])
        return self.df

    def FilterNYC(self):
        self.df["ZIP"] = self.df["zip code tabulation area"].str.replace("ZCTA5 ", "").str.strip()
        self.df = self.df[self.df["ZIP"].isin(self.nyc_zips)]
        return self.df

    def RenameAndClean(self):
        self.df.rename(columns=self.rename_map, inplace=True)
        for col in self.rename_map.values():
            self.df[col] = pd.to_numeric(self.df[col], errors="coerce")
        self.df.drop(columns=["zip code tabulation area", "NAME"], inplace=True)
        cols = ["ZIP"] + [c for c in self.df.columns if c != "ZIP"]
        self.df = self.df[cols]
        return self.df

    def ConvertDataTypes(self):
        dtype_map = {
            "ZIP": str,
            "Total_Population": "Int64",
            "Median_Age": float,
            "Male_Population": "Int64",
            "Female_Population": "Int64",
            "White": "Int64",
            "Black": "Int64",
            "American_Indian_Alaska_Native": "Int64",
            "Asian": "Int64",
            "Native_Hawaiian_Pacific_Islander": "Int64",
            "Other_Race": "Int64",
            "Two_or_More_Races": "Int64",
            "Median_Household_Income": "Int64",
            "Poverty_Total": "Int64",
            "Below_Poverty": "Int64",
            "Total_Educ_25plus": "Int64",
            "Bachelor_Degree": "Int64",
            "Advanced_Degree": "Int64",
            "Median_Home_Value": "Int64",
            "Median_Gross_Rent": "Int64",
            "Total_Housing_Units": "Int64",
            "Owner_Occupied": "Int64",
            "Renter_Occupied": "Int64",
            "Total_Labor_Force": "Int64",
            "Not_in_Labor_Force": "Int64",
            "Employed": "Int64",
            "Unemployed": "Int64",
            "Median_Travel_Time": "Int64",
            "Total_With_Disability": "Int64",
            "No_Health_Insurance": "Int64",
            "Vehicles_Available": "Int64",
            "Total_Language_Population": "Int64",
            "Foreign_Born": "Int64",
            "Median_Earnings": "Int64",
            "Households_with_Broadband": "Int64",
            "Median_Year_Built": "Int64"
        }
        for col, dtype in dtype_map.items():
            if col in self.df.columns:
                self.df[col] = self.df[col].astype(dtype)
        return self.df

    def SortByZip(self):
        self.df = self.df.sort_values("ZIP", ascending=True)
        return self.df

    def SaveToCSV(self, filename):
        self.df.to_csv(filename, index=False)
        print("Rows:", len(self.df), "saved to", filename)

    @staticmethod
    def MergeDataFrames(acs_df, external, ext_zip_key, threshold=100000):
        if isinstance(external, str):
            ext_df = pd.read_csv(external)
        else:
            ext_df = external.copy()
        ext_df.rename(columns={ext_zip_key: "ZIP"}, inplace=True)
        if len(ext_df) > threshold:
            try:
                import dask.dataframe as dd
                ddf = dd.from_pandas(acs_df, npartitions=4)
                ext_ddf = dd.from_pandas(ext_df, npartitions=4)
                merged = dd.merge(ddf, ext_ddf, on="ZIP", how="left").compute()
                print("Using Dask merge, merged data has", len(merged), "rows")
                return merged
            except ImportError:
                print("Dask not installed, using pandas merge")
        merged = pd.merge(acs_df, ext_df, on="ZIP", how="left")
        print("Using pandas merge, merged data has", len(merged), "rows")
        return merged

    def MergeWithExternal(self, external, ext_zip_key, threshold=100000):
        return AcsDataFetcher.MergeDataFrames(self.df, external, ext_zip_key, threshold)

    def Run(self):
        self.FetchData()
        self.FilterNYC()
        self.RenameAndClean()
        self.ConvertDataTypes()
        self.SortByZip()
        self.df.replace(-666666666, pd.NA, inplace=True)
        self.SaveToCSV("nyc_demographic_data.csv")
        return self.df

if __name__ == "__main__":
    fetcher = AcsDataFetcher(api_key=os.getenv("API_KEY"))
    dfResult = fetcher.Run()
    print("Sample data:")
    print(dfResult.head())
