import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

def plot_analysis_results(res, spans, sup_df, loads, reactions):
    """
    Revised Analysis Plots:
    - Clear Grid & Dashed Lines
    - Reactions separated from Supports
    - Civil Engineering Standard (BMD Inverted)
    """
    cum_dist = [0] + list(np.cumsum(spans))
    total_len = cum_dist[-1]
    
    # Setup Subplots (Fixed Heights for clarity)
    fig = make_subplots(
        rows=4, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.05,
        subplot_titles=(
            "1. Free Body Diagram (Load Model)", 
            "2. Shear Force Diagram (SFD)", 
            "3. Bending Moment Diagram (BMD)", 
            "4. Deflection"
        ),
        row_heights=[0.25, 0.25, 0.25, 0.25]
    )

    # --- ROW 1: FBD (Model) ---
    # 1.1 Beam Line
    fig.add_trace(go.Scatter(x=[0, total_len], y=[0, 0], mode='lines', 
                             line=dict(color='black', width=5), hoverinfo='skip'), row=1, col=1)

    # 1.2 Supports (Draw ABOVE beam to avoid reaction conflict)
    for s in sup_df.to_dict('records'):
        x_s = cum_dist[s['id']] if s['id'] < len(cum_dist) else 0
        # Draw Triangle/Pin
        fig.add_trace(go.Scatter(
            x=[x_s], y=[0.15], # Offset UP
            mode='markers+text',
            marker=dict(symbol='triangle-down', size=15, color='#2c3e50'),
            text=[f"{s.get('type','Sup')}"], textposition="top center",
            hoverinfo='none'
        ), row=1, col=1)

    # 1.3 Reactions (Draw BELOW beam)
    for r in reactions:
        x_r = cum_dist[r['node_id']] if r['node_id'] < len(cum_dist) else 0
        val = r.get('fy', 0) / 1000.0
        # Arrow Vector
        fig.add_annotation(
            x=x_r, y=-0.1, ax=0, ay=40, # Arrow points UP to the beam
            text=f"R={val:.2f}", 
            showarrow=True, arrowhead=2, arrowsize=1.5, arrowwidth=2, arrowcolor='green',
            row=1, col=1
        )

    # 1.4 Loads
    for l in loads:
        x_pos = l['x']
        mag = l['mag'] / 1000.0
        if l['type'] == 'P':
            fig.add_annotation(x=x_pos, y=0.1, ax=0, ay=-40, text=f"P={mag:.2f}", 
                               showarrow=True, arrowhead=2, arrowcolor='red', row=1, col=1)
        elif l['type'] == 'U':
            x_end = x_pos + l['dist']
            # Draw distributed load line
            fig.add_trace(go.Scatter(x=[x_pos, x_end], y=[0.5, 0.5], mode='lines', 
                                     line=dict(color='orange', width=2), hoverinfo='skip'), row=1, col=1)
            fig.add_annotation(x=(x_pos+x_end)/2, y=0.5, ax=0, ay=-20, text=f"w={mag:.2f}", 
                               showarrow=True, arrowcolor='orange', row=1, col=1)

    # --- ROW 2: SHEAR (SFD) ---
    y_shear = res['shear'] / 1000.0
    fig.add_trace(go.Scatter(x=res['x'], y=y_shear, mode='lines', fill='tozeroy', 
                             line=dict(color='#e74c3c', width=2), name='Shear'), row=2, col=1)
    
    # Label Max/Min Shear
    v_max, v_min = y_shear.max(), y_shear.min()
    fig.add_annotation(x=res.loc[y_shear.idxmax(), 'x'], y=v_max, text=f"{v_max:.2f}", showarrow=False, yshift=10, row=2, col=1)
    fig.add_annotation(x=res.loc[y_shear.idxmin(), 'x'], y=v_min, text=f"{v_min:.2f}", showarrow=False, yshift=-10, row=2, col=1)

    # --- ROW 3: MOMENT (BMD) - INVERTED ---
    y_moment = res['moment'] / 1000.0
    fig.add_trace(go.Scatter(x=res['x'], y=y_moment, mode='lines', fill='tozeroy', 
                             line=dict(color='#2980b9', width=2), name='Moment'), row=3, col=1)
    
    # Label Max/Min Moment
    m_max, m_min = y_moment.max(), y_moment.min()
    # Annotate global peaks
    fig.add_annotation(x=res.loc[y_moment.idxmax(), 'x'], y=m_max, text=f"{m_max:.2f}", showarrow=False, yshift=10, row=3, col=1)
    fig.add_annotation(x=res.loc[y_moment.idxmin(), 'x'], y=m_min, text=f"{m_min:.2f}", showarrow=False, yshift=-10, row=3, col=1)
    
    # Invert Y Axis for BMD (Civil Standard)
    fig.update_yaxes(autorange="reversed", title_text="Moment (kNm)", row=3, col=1)

    # --- ROW 4: DEFLECTION ---
    y_def = res['deflection']
    fig.add_trace(go.Scatter(x=res['x'], y=y_def, mode='lines', 
                             line=dict(color='#27ae60', width=2), name='Deflection'), row=4, col=1)
    d_max = y_def.abs().max()
    d_idx = y_def.abs().idxmax()
    fig.add_annotation(x=res.loc[d_idx, 'x'], y=y_def[d_idx], text=f"{y_def[d_idx]:.2f} mm", showarrow=True, row=4, col=1)

    # --- GLOBAL STYLING ---
    # Vertical Dashed Lines (Grid) at Supports
    for x_line in cum_dist:
        fig.add_vline(x=x_line, line_width=1, line_dash="dash", line_color="gray", opacity=0.5)

    fig.update_layout(
        height=1000, 
        showlegend=False, 
        plot_bgcolor='white',
        margin=dict(l=60, r=20, t=40, b=40)
    )
    
    # Update axes with proper grid
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#eee', zeroline=True, zerolinecolor='#333', row=4, col=1, title_text="Length (m)")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#eee', zeroline=True, zerolinecolor='#333')
    
    # Hide Y axis ticks for FBD
    fig.update_yaxes(showticklabels=False, row=1, col=1)

    return fig
