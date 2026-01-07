import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

def plot_analysis_results(res, spans, sup_df, loads):
    """
    Plot Analysis Results with Max/Min Annotations.
    """
    cum_dist = [0] + list(np.cumsum(spans))
    total_len = cum_dist[-1]
    
    # Create Subplots
    fig = make_subplots(rows=3, cols=1, 
                        shared_xaxes=True, 
                        vertical_spacing=0.08,
                        subplot_titles=("Shear Force Diagram (SFD)", "Bending Moment Diagram (BMD)", "Deflection"),
                        row_heights=[0.3, 0.35, 0.35])

    # --- 1. Shear Force (SFD) ---
    y_shear = res['shear'] / 1000.0 # kN
    max_v = y_shear.max()
    min_v = y_shear.min()
    
    fig.add_trace(go.Scatter(x=res['x'], y=y_shear, mode='lines', fill='tozeroy', 
                             line=dict(color='#e74c3c', width=2), name='Shear (kN)'), row=1, col=1)
    
    # Annotate Max/Min Shear
    if not np.isnan(max_v):
        fig.add_annotation(x=res.loc[y_shear.idxmax(), 'x'], y=max_v, text=f"{max_v:.2f}", showarrow=True, row=1, col=1)
    if not np.isnan(min_v):
        fig.add_annotation(x=res.loc[y_shear.idxmin(), 'x'], y=min_v, text=f"{min_v:.2f}", showarrow=True, row=1, col=1)

    # --- 2. Bending Moment (BMD) ---
    y_moment = res['moment'] / 1000.0 # kNm
    max_m = y_moment.max()
    min_m = y_moment.min() 

    fig.add_trace(go.Scatter(x=res['x'], y=y_moment, mode='lines', fill='tozeroy', 
                             line=dict(color='#3498db', width=2), name='Moment (kNm)'), row=2, col=1)
    
    if not np.isnan(max_m):
        fig.add_annotation(x=res.loc[y_moment.idxmax(), 'x'], y=max_m, text=f"{max_m:.2f}", showarrow=True, row=2, col=1, ay=-20)
    if not np.isnan(min_m):
        fig.add_annotation(x=res.loc[y_moment.idxmin(), 'x'], y=min_m, text=f"{min_m:.2f}", showarrow=True, row=2, col=1, ay=20)

    # --- 3. Deflection ---
    y_def = res['deflection'] # mm
    if len(y_def) > 0:
        idx_def = y_def.abs().idxmax()
        val_def = y_def.iloc[idx_def]

        fig.add_trace(go.Scatter(x=res['x'], y=y_def, mode='lines', 
                                 line=dict(color='#2ecc71', width=2), name='Deflection (mm)'), row=3, col=1)
        
        fig.add_annotation(x=res.loc[idx_def, 'x'], y=val_def, text=f"{val_def:.2f} mm", showarrow=True, row=3, col=1)

    # --- Supports Lines ---
    for x_s in cum_dist:
        fig.add_vline(x=x_s, line_width=1, line_dash="dash", line_color="gray")

    # --- Layout (FIXED bg_color -> paper_bgcolor) ---
    fig.update_layout(
        height=700, 
        showlegend=False, 
        margin=dict(l=50, r=20, t=40, b=40), 
        paper_bgcolor='white',  # แก้ไขตรงนี้
        plot_bgcolor='white'    # และตรงนี้
    )
    
    fig.update_xaxes(title_text="Distance (m)", row=3, col=1, showgrid=True, gridcolor='#eee')
    fig.update_yaxes(title_text="V (kN)", row=1, col=1, showgrid=True, gridcolor='#eee')
    fig.update_yaxes(title_text="M (kNm)", row=2, col=1, showgrid=True, gridcolor='#eee')
    fig.update_yaxes(title_text="Def (mm)", row=3, col=1, showgrid=True, gridcolor='#eee')

    return fig
