import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

def plot_analysis_results(res_df, spans, supports, loads, reactions):
    """
    Creates a Plotly figure with 3 subplots: Load Model, Shear, Moment.
    """
    fig = make_subplots(rows=3, cols=1, 
                        shared_xaxes=True, 
                        vertical_spacing=0.1,
                        subplot_titles=("Load Model & Structure", "Shear Force Diagram (SFD)", "Bending Moment Diagram (BMD)"),
                        row_heights=[0.3, 0.35, 0.35])

    # --- 1. Load Model ---
    # Beam Line
    total_L = sum(spans)
    fig.add_trace(go.Scatter(x=[0, total_L], y=[0, 0], mode='lines', line=dict(color='black', width=4)), row=1, col=1)
    
    # Supports
    for idx, row in supports.iterrows():
        sym = "triangle-up" if row['type'] != 'Fixed' else "square"
        fig.add_trace(go.Scatter(x=[row['x']], y=[-0.2], mode='markers', marker=dict(symbol=sym, size=15, color='black'), showlegend=False), row=1, col=1)
        
    # Loads (Simplified Visualization)
    for l in loads:
        x_start = 0 
        # Calculate absolute x position based on span index
        current_x = 0
        for i in range(l['span_index']): current_x += spans[i]
        
        if l['type'] == 'P':
            x_load = current_x + l['dist']
            fig.add_trace(go.Scatter(x=[x_load, x_load], y=[0.5, 0], mode='lines+markers', line=dict(color='red', width=2), marker=dict(symbol='arrow-down', size=10)), row=1, col=1)
            fig.add_annotation(x=x_load, y=0.6, text=f"{l['mag']/1000:.1f}kN", showarrow=False, row=1, col=1)
        elif l['type'] == 'U':
            x_s = current_x
            x_e = current_x + l['dist']
            fig.add_trace(go.Scatter(x=[x_s, x_e], y=[0.3, 0.3], mode='lines', line=dict(color='blue'), fill='tozeroy', fillcolor='rgba(0,0,255,0.1)'), row=1, col=1)
            fig.add_annotation(x=(x_s+x_e)/2, y=0.4, text=f"{l['mag']/1000:.1f}kN/m", showarrow=False, row=1, col=1)

    # --- 2. SFD ---
    fig.add_trace(go.Scatter(x=res_df['x'], y=res_df['shear']/1000, mode='lines', name='Shear (kN)', line=dict(color='#e74c3c'), fill='tozeroy'), row=2, col=1)
    
    # --- 3. BMD ---
    # Invert Moment for civil engineering sign convention (Positive Moment Down)
    fig.add_trace(go.Scatter(x=res_df['x'], y=-res_df['moment']/1000, mode='lines', name='Moment (kNm)', line=dict(color='#2ecc71'), fill='tozeroy'), row=3, col=1)
    fig.update_yaxes(autorange="reversed", row=3, col=1) # Flip Y axis for Moment

    fig.update_layout(height=800, showlegend=False, template="plotly_white")
    return fig
