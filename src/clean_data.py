import pandas as pd
import os

def clean_comparitech(input_path, output_path):
    print(f"Loading Comparitech data from {input_path}")
    df = pd.read_csv(input_path)
    
    # Rename columns for easier access
    df.rename(columns={
        '# of cameras per 1,000 people': 'Cameras_per_1000',
        '# of CCTV Cameras': 'Camera_Count',
        'Population (2025)': 'Population'
    }, inplace=True)
    
    # Clean numeric columns (handle N/A)
    # Some N/A might be strings "N/A"
    cols_to_clean = ['Cameras_per_1000', 'Camera_Count']
    for col in cols_to_clean:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop rows where critical data is missing
    df.dropna(subset=['Cameras_per_1000'], inplace=True)
    
    # Standardize Country names (basic stripping)
    df['Country'] = df['Country'].str.strip()
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved cleaned Comparitech data to {output_path}")

def clean_unodc(input_path, output_path):
    print(f"Loading UNODC data from {input_path}")
    # Header is on row 3 (index 2)
    df = pd.read_excel(input_path, sheet_name='data_cts_intentional_homicide', header=2)
    
    print("Available columns:", df.columns.tolist())
    
    # Filter for Homicide Rate
    # Indicator: Victims of intentional homicide
    # Unit: Rate per 100,000 population
    
    mask = (df['Indicator'] == 'Victims of intentional homicide') & \
           (df['Unit of measurement'] == 'Rate per 100,000 population')
    
    df_rate = df[mask].copy()
    
    if df_rate.empty:
        print("WARNING: No data found for 'Rate per 100,000 population'. Checking 'Counts'...")
        # Fallback logic if needed, but usually UNODC has rates.
        # Let's check unique units if empty
        print("Unique units in filtered Indicator:", df[df['Indicator']=='Victims of intentional homicide']['Unit of measurement'].unique())
        return

    # Save full time series for trend analysis
    df_rate.sort_values(by=['Country', 'Year'], inplace=True)
    ts_output_path = output_path.replace('diff_unodc.csv', 'unodc_timeseries.csv')
    df_rate[['Country', 'Year', 'VALUE']].to_csv(ts_output_path, index=False)
    print(f"Saved UNODC time series data to {ts_output_path}")

    # Select closest year to 2021 for each country
    target_year = 2021
    
    def get_closest_year(group):
        # Calculate absolute difference from target year
        group['year_diff'] = abs(group['Year'] - target_year)
        # Sort by difference (asc) and then by Year (desc) to prefer later years in tie
        return group.sort_values(by=['year_diff', 'Year'], ascending=[True, False]).iloc[0]

    df_latest = df_rate.groupby('Country', group_keys=False).apply(get_closest_year)
    
    # Optional: Filter out if data is too old (e.g., > 10 years gap)?
    # For now, keeping all to maximize matches, but let's check statistics later.
    
    df_latest = df_latest[['Country', 'Year', 'VALUE', 'Region', 'Subregion']]
    df_latest.rename(columns={'VALUE': 'Homicide_Rate', 'Year': 'Data_Year'}, inplace=True)
    
    # Standardize Country names
    df_latest['Country'] = df_latest['Country'].str.strip()
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_latest.to_csv(output_path, index=False)
    print(f"Saved cleaned UNODC data to {output_path}")

def clean_all():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data', 'processed')
    
    comp_input = os.path.join(base_dir, 'comparitech-cctv-per-1000.csv')
    comp_output = os.path.join(data_dir, 'diff_comparitech.csv')
    
    unodc_input = os.path.join(base_dir, 'unodc-homicide-data.xlsx')
    unodc_output = os.path.join(data_dir, 'diff_unodc.csv')
    
    clean_comparitech(comp_input, comp_output)
    clean_unodc(unodc_input, unodc_output)

if __name__ == "__main__":
    clean_all()
