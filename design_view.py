import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

def plot_analysis_results(res, spans, sup_input, loads, reactions):
    # ป้องกัน error กรณี spans เป็น None หรือ empty
    if not spans:
        spans = [0]
    
    cum_dist = [0] + list(np.cumsum(spans))
    total_len = cum_dist[-1]
    
    # Handle Supports Input
    if isinstance(sup_input, pd.DataFrame):
        supports = sup_input.to_dict('records')
    elif isinstance(sup_input, list):
        supports = sup_input
    else:
        supports = []

    # Prepare Data
    y_shear = res['shear'].values / 1000.0
    y_moment = res['moment'].values / 1000.0
    y_def = res['deflection'].values

    fig = make_subplots(
        rows=4, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.08,
        subplot_titles=("1. FBD & Loads", "2. Shear Force (kN)", "3. Bending Moment (kNm)", "4. Deflection (mm)"),
        row_heights=[0.3, 0.22, 0.22, 0.26]
    )

    # --- 1. FBD ---
    # Beam Line
    fig.add_trace(go.Scatter(x=[0, total_len], y=[0, 0], mode='lines', 
                             line=dict(color='black', width=4), hoverinfo='skip'), row=1, col=1)

    # Supports
    for s in supports:
        idx = int(s.get('id', 0))
        # Check index bounds
        x_s = cum_dist[idx] if idx < len(cum_dist) else 0
        fig.add_trace(go.Scatter(
            x=[x_s], y=[-0.05],
            mode='markers',
            marker=dict(symbol='triangle-up', size=15, color='#34495e'),
            showlegend=False, hoverinfo='text', text=s.get('type')
        ), row=1, col=1)

    # Loads
    max_load_mag = 1.0 # Scale factor logic
    for l in loads:
        x_pos = l['x']
        mag = l['mag'] / 1000.0 # kN or kN/m
        dist = l.get('dist', 0)
        
        if abs(mag) > max_load_mag: max_load_mag = abs(mag)

        if l['type'] == 'P':
            fig.add_annotation(x=x_pos, y=0, ax=0, ay=-40, text=f"{mag:.1f}kN", 
                               showarrow=True, arrowhead=2, arrowcolor='red', row=1, col=1)
        elif l['type'] == 'U':
            x_end = x_pos + dist
            # Draw line representing UDL
            fig.add_trace(go.Scatter(x=[x_pos, x_end], y=[0.15, 0.15], mode='lines', 
                                     line=dict(color='orange', width=2), showlegend=False), row=1, col=1)
            fig.add_annotation(x=(x_pos+x_end)/2, y=0.15, text=f"{mag:.1f} kN/m", 
                               showarrow=False, yshift=10, font=dict(color='orange'), row=1, col=1)
            # Add arrows
            for xa in np.linspace(x_pos, x_end, num=int(dist)+2):
                fig.add_annotation(x=xa, y=0, ax=0, ay=-30, showarrow=True, arrowhead=1, 
                                   arrowwidth=1, arrowcolor='orange', row=1, col=1)

    # --- 2. SFD ---
    fig.add_trace(go.Scatter(x=res['x'], y=y_shear, mode='lines', fill='tozeroy', 
                             line=dict(color='#e74c3c'), name='Shear'), row=2, col=1)

    # --- 3. BMD ---
    fig.add_trace(go.Scatter(x=res['x'], y=y_moment, mode='lines', fill='tozeroy', 
                             line=dict(color='#2980b9'), name='Moment'), row=3, col=1)
    fig.update_yaxes(autorange="reversed", row=3, col=1) # Moment Convention

    # --- 4. Deflection ---
    fig.add_trace(go.Scatter(x=res['x'], y=y_def, mode='lines', 
                             line=dict(color='#8e44ad'), name='Deflection'), row=4, col=1)

    # Formatting
    fig.update_layout(height=1000, showlegend=False, margin=dict(l=50, r=20, t=40, b=40))
    fig.update_yaxes(range=[-0.5, 0.5], row=1, col=1, showticklabels=False) # Fix FBD range
    
    return fig
