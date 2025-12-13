# Surveillance vs Homicide Analysis

This project analyzes the correlation between the density of surveillance cameras (Comparitech data) and intentional homicide rates (UNODC data).

## Project Structure

- `src/`: Python scripts for data processing and analysis.
- `data/`: Raw and processed data.
  - `comparitech-cctv-per-1000.csv`: Raw camera data.
  - `unodc-homicide-data.xlsx`: Raw homicide data.
  - `processed/`: Cleaned CSVs.
- `results/`: Output figures and statistics.
- `pixi.toml`: Configuration for the `pixi` environment and tasks.

## Prerequisites

- [Pixi](https://prefix.dev/) package manager.

## Reproducibility

To reproduce the analysis, run the configured Pixi tasks.

### 1. Install Dependencies

Pixi will automatically handle the environment creation.

```bash
pixi install
```

### 2. Run Analysis

You can run the entire pipeline (clean -> merge -> analyze) with a single command:

```bash
pixi run all
```

This command will:

1.  **Clean**: Process raw data from `comparitech-cctv-per-1000.csv` and `unodc-homicide-data.xlsx`.
2.  **Merge**: Combine city-level camera data with country-level homicide rates.
3.  **Analyze**: Perform outlier exclusion, calculate correlations, and generate plots.

### 3. View Results

- **Interactive Graph**: `results/figures/city_cameras_vs_homicide.html` (Open in browser).
- **Statistics**: `results/stats.txt` (Correlation coefficients and sample size).

## Statistical Methods

This analysis follows a strict chronological pipeline to ensure reproducibility:

1.  **Data Preprocessing & Filtering**:

    - **Surveillance Data**: Ingested city-level CCTV density (Cameras/1,000 people) from Comparitech (Bischoff, 2019).
    - **Homicide Data**: Ingested national intentional homicide rates (per 100k) from UNODC (2022).
    - **Temporal Consistency**: UNODC data filtered strictly for the year **2021 only**. Countries without reported data for 2021 are silently excluded to ensure temporal consistency.
    - **Merging**: City data mapped to corresponding national homicide rates.

2.  **Outlier Removal**:

    - Bivariate outlier detection applied using Z-scores.
    - Exclusion criterion: $|Z| > 3$ for either Camera Density or Homicide Rate.

3.  **Analysis & Visualization**:
    - **Correlation**: Calculated Pearson ($r$) and Spearman ($\rho$) coefficients on the cleaned, filtered dataset.
    - **Regression**: Ordinary Least Squares (OLS) linear trendline fitted to visualize the relationship.
    - **Visualization**: Interactive scatter plot generated via Plotly.
