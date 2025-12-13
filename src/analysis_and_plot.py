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
    
    # Filter strictly for 2021 data (this silently excludes non-2021 data including some China entries if they aren't 2021)
    df_clean = df_clean[df_clean['Data_Year'] == 2021]

    # --- Outlier Detection (Rest of World) ---
    # Using Z-score > 3
    df_clean['z_cam'] = np.abs(stats.zscore(df_clean['Cameras_per_1000']))
    df_clean['z_hom'] = np.abs(stats.zscore(df_clean['Homicide_Rate']))
    
    outliers = df_clean[(df_clean['z_cam'] > 3) | (df_clean['z_hom'] > 3)]
    df_no_outliers = df_clean[(df_clean['z_cam'] <= 3) & (df_clean['z_hom'] <= 3)]
    
    with open(os.path.join(results_dir, 'stats.txt'), 'w') as f:
        f.write("--- Analysis Statistics (2021 Data Only) ---\n")
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
            size=16,  # Significantly increased for mobile visibility
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
                size=40,  # Significantly increased for mobile readability
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
                size=24,  # Significantly increased for mobile readability
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
                size=20,  # Significantly increased for mobile readability
                color='#6a6a6a'
            )
        ),
        yaxis=dict(
            title='Homicide Rate per 100K people (2021)',
            titlefont=dict(
                family='Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
                size=24,  # Significantly increased for mobile readability
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
                size=20,  # Significantly increased for mobile readability
                color='#6a6a6a'
            )
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        hovermode='closest',
        hoverlabel=dict(
            bgcolor='white',
            font_size=20,  # Increased from 18 for value readability
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
            size=24,  # Significantly increased from 18 to 24 for mobile readability
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
            size=16,  # Significantly increased for mobile readability
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
