import pandas as pd
import os

# Country Name Mapping (Comparitech -> UNODC)
# Adjust these based on errors found during execution
COUNTRY_MAPPING = {
    "United States": "United States of America",
    "United Kingdom": "United Kingdom of Great Britain and Northern Ireland",
    "Russia": "Russian Federation",
    "South Korea": "Republic of Korea",
    "Vietnam": "Viet Nam",
    "Turkey": "Türkiye",
    "Iran": "Iran (Islamic Republic of)",
    "Tanzania": "United Republic of Tanzania",
    "Ivory Coast": "Côte d'Ivoire",
    "Syria": "Syrian Arab Republic",
    "Bolivia": "Bolivia (Plurinational State of)",
    "Venezuela": "Venezuela (Bolivarian Republic of)",
    "Moldova": "Republic of Moldova",
    "Czech Republic": "Czechia"
}

def merge_data():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data', 'processed')
    
    comparitech_path = os.path.join(data_dir, 'diff_comparitech.csv')
    unodc_path = os.path.join(data_dir, 'diff_unodc.csv')
    output_path = os.path.join(data_dir, 'merged_dataset.csv')
    
    print("Loading datasets...")
    df_cam = pd.read_csv(comparitech_path)
    df_hom = pd.read_csv(unodc_path)
    
    print(f"Cameras loaded: {len(df_cam)} cities")
    print(f"Homicides loaded: {len(df_hom)} countries")
    
    # Normalize Country names in Cameras dataset
    df_cam['Country_Normalized'] = df_cam['Country'].replace(COUNTRY_MAPPING)
    
    # Merge
    # Left join to keep all cities
    merged = pd.merge(df_cam, df_hom, left_on='Country_Normalized', right_on='Country', how='left', suffixes=('', '_unodc'))
    
    # Check for unmatched countries
    unmatched = merged[merged['Homicide_Rate'].isna()]['Country'].unique()
    if len(unmatched) > 0:
        print(f"WARNING: The following countries in Comparitech data could not be matched to UNODC data:\n{unmatched}")
    
    # Save
    merged.to_csv(output_path, index=False)
    print(f"Saved merged dataset to {output_path}")
    print(f"Total merged rows: {len(merged)}")
    print(f"Rows with valid Homicide Rate: {merged['Homicide_Rate'].notna().sum()}")

if __name__ == "__main__":
    merge_data()
