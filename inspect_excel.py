import pandas as pd

file_path = 'unodc-homicide-data.xlsx'
xl = pd.ExcelFile(file_path)
print(f"Sheet names: {xl.sheet_names}")

for sheet in xl.sheet_names:
    print(f"\n--- Sheet: {sheet} ---")
    df = pd.read_excel(file_path, sheet_name=sheet, nrows=5)
    print(df.columns.tolist())
    print(df.head())
