import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import os
import numpy as np

def analyze_and_plot():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data', 'processed', 'merged_dataset.csv')
    ts_path = os.path.join(base_dir, 'data', 'processed', 'unodc_timeseries.csv')
    results_dir = os.path.join(base_dir, 'results')
    figures_dir = os.path.join(results_dir, 'figures')
    
    os.makedirs(figures_dir, exist_ok=True)
    
    df = pd.read_csv(data_path)
    
    # Filter out rows with missing homicide data
    df_clean = df.dropna(subset=['Homicide_Rate', 'Cameras_per_1000']).copy()
    
    # --- China Outlier Check ---
    china_data = df_clean[df_clean['Country_Normalized'] == 'China']
    if not china_data.empty:
        china_cam = china_data['Cameras_per_1000'].values[0]
        china_hom = china_data['Homicide_Rate'].values[0]
        
        # Calculate stats for the WHOLE dataset (including China) to see where it stands
        mean_cam = df_clean['Cameras_per_1000'].mean()
        std_cam = df_clean['Cameras_per_1000'].std()
        
        z_china_cam = (china_cam - mean_cam) / std_cam
        
        print(f"--- China Outlier Check ---")
        print(f"China Cameras/1000: {china_cam}")
        print(f"Global Mean Cameras: {mean_cam:.2f} (Std: {std_cam:.2f})")
        print(f"China Camera Z-Score: {z_china_cam:.2f}")
        if abs(z_china_cam) > 3:
            print("CONCLUSION: China is a statistical outlier (>3 Std Devs) in camera density.")
            print("Excluding China from main analysis...")
            df_clean = df_clean[df_clean['Country_Normalized'] != 'China']
        else:
            print("CONCLUSION: China is NOT a statistical outlier in camera density.")
            print("Keeping China in the analysis as requested.")
    else:
        print("China not found in dataset or already excluded.")

    # --- Outlier Detection (Rest of World) ---
    # Using Z-score > 3
    df_clean['z_cam'] = np.abs(stats.zscore(df_clean['Cameras_per_1000']))
    df_clean['z_hom'] = np.abs(stats.zscore(df_clean['Homicide_Rate']))
    
    outliers = df_clean[(df_clean['z_cam'] > 3) | (df_clean['z_hom'] > 3)]
    df_no_outliers = df_clean[(df_clean['z_cam'] <= 3) & (df_clean['z_hom'] <= 3)]
    
    with open(os.path.join(results_dir, 'stats.txt'), 'w') as f:
        f.write("--- Analysis Statistics (Excluding China) ---\n")
        if not china_data.empty:
             f.write(f"China Camera Density Z-Score: {z_china_cam:.2f}\n")
        f.write(f"Original Cities (No China): {len(df_clean)}\n")
        f.write(f"Outliers excluded (Z>3): {len(outliers)}\n")
        f.write(f"Cities remaining: {len(df_no_outliers)}\n\n")
        
        # --- Correlations ---
        # 1. City Level
        corr_pearson = df_no_outliers['Cameras_per_1000'].corr(df_no_outliers['Homicide_Rate'], method='pearson')
        corr_spearman = df_no_outliers['Cameras_per_1000'].corr(df_no_outliers['Homicide_Rate'], method='spearman')
        
        f.write("--- City Level Correlation ---\n")
        f.write(f"Pearson r: {corr_pearson:.3f}\n")
        f.write(f"Spearman rho: {corr_spearman:.3f}\n\n")
        
        # 2. Country Level
        # Group by Country
        country_grp = df_no_outliers.groupby('Country_Normalized').agg({
            'Cameras_per_1000': 'mean',
            'Homicide_Rate': 'mean' # Should be same for all cities in country, but mean is safe
        }).reset_index()
        
        corr_country_p = country_grp['Cameras_per_1000'].corr(country_grp['Homicide_Rate'], method='pearson')
        corr_country_s = country_grp['Cameras_per_1000'].corr(country_grp['Homicide_Rate'], method='spearman')
        
        f.write("--- Country Level Correlation ---\n")
        f.write(f"Number of Countries: {len(country_grp)}\n")
        f.write(f"Pearson r: {corr_country_p:.3f}\n")
        f.write(f"Spearman rho: {corr_country_s:.3f}\n")

    print("Statistics saved to results/stats.txt")
    
    # --- Plotting ---
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    from scipy import stats as scipy_stats
    
    # 1. City Level Scatter (Interactive) - Clean Minimal Design
    
    # Calculate trendline manually for styling control
    x = df_no_outliers['Cameras_per_1000'].values
    y = df_no_outliers['Homicide_Rate'].values
    slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(x, y)
    line_x = np.array([x.min(), x.max()])
    line_y = slope * line_x + intercept
    
    # Create figure with custom styling
    fig = go.Figure()
    
    # Add scatter points with minimal, clean styling - larger for mobile
    fig.add_trace(go.Scatter(
        x=df_no_outliers['Cameras_per_1000'],
        y=df_no_outliers['Homicide_Rate'],
        mode='markers',
        marker=dict(
            size=10,  # Increased from 8 for better mobile touch targets
            color='rgba(100, 100, 100, 0.6)',  # Subtle gray
            line=dict(width=0.5, color='rgba(255, 255, 255, 0.8)'),
        ),
        text=df_no_outliers['City'],
        customdata=df_no_outliers['Country_Normalized'],
        hovertemplate='<b>%{text}</b><br>' +
                      '%{customdata}<br>' +
                      'Cameras: %{x:.1f}<br>' +
                      'Homicide Rate: %{y:.1f}<br>' +
                      '<extra></extra>',
        name='Cities',
        showlegend=False
    ))
    
    # Add trendline with minimal styling
    fig.add_trace(go.Scatter(
        x=line_x,
        y=line_y,
        mode='lines',
        line=dict(color='rgba(0, 0, 0, 0.3)', width=1.5, dash='dot'),
        name='Trend',
        showlegend=False,
        hoverinfo='skip'
    ))
    
    # Clean, minimal layout with mobile-friendly font sizes
    fig.update_layout(
        title=dict(
            text='City Surveillance vs. Homicide',
            font=dict(
                family='Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
                size=28,  # Increased from 24 for mobile readability
                color='#1a1a1a'
            ),
            x=0.5,
            xanchor='center',
            y=0.95,
            yanchor='top'
        ),
        xaxis=dict(
            title='Cameras per 1,000 People (2019)',
            titlefont=dict(
                family='Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
                size=16,  # Increased from 13 for mobile readability
                color='#4a4a4a'
            ),
            showgrid=True,
            gridcolor='rgba(0, 0, 0, 0.05)',
            gridwidth=1,
            zeroline=False,
            showline=True,
            linewidth=1,
            linecolor='rgba(0, 0, 0, 0.1)',
            tickfont=dict(
                family='Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
                size=14,  # Increased from 11 for mobile readability
                color='#6a6a6a'
            )
        ),
        yaxis=dict(
            title='Homicide Rate per 100K people (2021 or nearest)',
            titlefont=dict(
                family='Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
                size=16,  # Increased from 13 for mobile readability
                color='#4a4a4a'
            ),
            showgrid=True,
            gridcolor='rgba(0, 0, 0, 0.05)',
            gridwidth=1,
            zeroline=False,
            showline=True,
            linewidth=1,
            linecolor='rgba(0, 0, 0, 0.1)',
            tickfont=dict(
                family='Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
                size=14,  # Increased from 11 for mobile readability
                color='#6a6a6a'
            )
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        hovermode='closest',
        hoverlabel=dict(
            bgcolor='white',
            font_size=14,  # Increased from 12 for mobile readability
            font_family='Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
            font_color='#1a1a1a',
            bordercolor='rgba(0, 0, 0, 0.1)'
        ),
        margin=dict(l=90, r=50, t=110, b=160),  # Increased margins for mobile spacing
        autosize=True  # Enable responsive sizing
    )
    
    # Add trendline annotation with larger font
    fig.add_annotation(
        text=f'r = {r_value:.3f}',
        xref='paper', yref='paper',
        x=0.98, y=0.02,
        showarrow=False,
        font=dict(
            family='Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
            size=13,  # Increased from 11 for mobile readability
            color='#8a8a8a'
        ),
        xanchor='right',
        yanchor='bottom'
    )

    # Add Data Source Citations with enhanced styling and larger font
    fig.add_annotation(
        text="<b>Data Sources</b><br>" + 
             "<span style='font-size: 11px;'>Bischoff, P. (2019) 'Surveillance Camera Statistics: Which City has the Most CCTV?', <i>Comparitech</i>.<br>" +
             "UNODC (2022) 'UNODC Research - Data Portal – Intentional Homicide', <i>United Nations Office on Drugs and Crime</i>.</span>",
        xref='paper', yref='paper',
        x=0.5, y=-0.15,
        showarrow=False,
        font=dict(
            family='Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
            size=13,  # Increased from 10 for mobile readability
            color='#5a5a5a'
        ),
        align='center',
        xanchor='center',
        yanchor='top',
        bgcolor='rgba(250, 250, 250, 0.8)',
        bordercolor='rgba(0, 0, 0, 0.05)',
        borderwidth=1,
        borderpad=10  # Increased padding for better mobile spacing
    )
    
    # Write HTML with responsive configuration
    config = {
        'responsive': True,  # Enable responsive resizing
        'displayModeBar': True,
        'displaylogo': False
    }
    fig.write_html(
        os.path.join(figures_dir, 'city_cameras_vs_homicide.html'),
        config=config
    )
    
    print("Interactive plot saved to results/figures/city_cameras_vs_homicide.html")

if __name__ == "__main__":
    analyze_and_plot()
