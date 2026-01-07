import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

def plot_analysis_results(res_df, spans):
    """Create Interactive Diagrams using Plotly"""
    if res_df.empty:
        return go.Figure()
        
    cum_dist = [0] + list(np.cumsum(spans))
    
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=("Shear Force Diagram (SFD)", "Bending Moment Diagram (BMD)", "Deflection"),
        row_heights=[0.33, 0.33, 0.33]
    )
    
    # 1. Shear Force (SFD) - Green Filled
    fig.add_trace(go.Scatter(
        x=res_df['x'], y=res_df['shear']/1000,
        fill='tozeroy', line=dict(color='#27AE60', width=2),
        name="Shear (kN)", hovertemplate="x: %{x:.2f}m<br>V: %{y:.2f} kN"
    ), row=1, col=1)
    
    # 2. Bending Moment (BMD) - Red Line (Inverted Y for Engineers)
    fig.add_trace(go.Scatter(
        x=res_df['x'], y=res_df['moment']/1000,
        fill='tozeroy', line=dict(color='#C0392B', width=2),
        name="Moment (kNm)", hovertemplate="x: %{x:.2f}m<br>M: %{y:.2f} kNm"
    ), row=2, col=1)
    
    # Invert Y axis for Moment (Thai/US Engineering standard)
    fig.update_yaxes(autorange="reversed", title_text="Moment (kNm)", row=2, col=1)
    
    # 3. Deflection - Blue Line
    fig.add_trace(go.Scatter(
        x=res_df['x'], y=res_df['deflection']*1000, # to mm
        line=dict(color='#2980B9', width=2),
        name="Deflection (mm)", hovertemplate="x: %{x:.2f}m<br>Delta: %{y:.2f} mm"
    ), row=3, col=1)
    
    # Add Support Lines
    for x_sup in cum_dist:
        fig.add_vline(x=x_sup, line_width=1, line_dash="dash", line_color="gray")

    # Styling
    fig.update_layout(
        height=700,
        showlegend=False,
        hovermode="x unified",
        margin=dict(l=50, r=20, t=50, b=50),
        plot_bgcolor="white"
    )
    fig.update_xaxes(showgrid=True, gridcolor='#eee', title_text="Length (m)", row=3, col=1)
    fig.update_yaxes(showgrid=True, gridcolor='#eee', title_text="Shear (kN)", row=1, col=1)
    fig.update_yaxes(showgrid=True, gridcolor='#eee', title_text="Deflection (mm)", row=3, col=1)
    
    return fig
