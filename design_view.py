import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

def plot_analysis_results(res, spans, sup_df, loads, reactions):
    """
    Revised Analysis Plots:
    - Supports moved TOP (Triangle Down)
    - Reactions moved BOTTOM (Arrow Up)
    - Loads: Better visuals
    - Annotations: Restore arrows
    """
    cum_dist = [0] + list(np.cumsum(spans))
    total_len = cum_dist[-1]
    
    # Define Subplots
    fig = make_subplots(
        rows=4, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.06,
        subplot_titles=(
            "1. Free Body Diagram (FBD)", 
            "2. Shear Force Diagram (SFD)", 
            "3. Bending Moment Diagram (BMD)", 
            "4. Deflection"
        ),
        row_heights=[0.3, 0.23, 0.23, 0.24]
    )

    # --- ROW 1: FBD ---
    # 1.1 Beam Line
    fig.add_trace(go.Scatter(x=[0, total_len], y=[0, 0], mode='lines', 
                             line=dict(color='black', width=4), hoverinfo='skip'), row=1, col=1)

    # 1.2 Supports (Moved to TOP, Pointing Down)
    for s in sup_df.to_dict('records'):
        x_s = cum_dist[s['id']] if s['id'] < len(cum_dist) else 0
        fig.add_trace(go.Scatter(
            x=[x_s], y=[0.15], # Offset ABOVE beam
            mode='markers+text',
            marker=dict(symbol='triangle-down', size=14, color='#2c3e50'),
            text=[f"{s.get('type','Sup')}"], textposition="top center",
            hoverinfo='none', showlegend=False
        ), row=1, col=1)

    # 1.3 Reactions (Moved to BOTTOM, Pointing Up)
    for r in reactions:
        x_r = cum_dist[r['node_id']] if r['node_id'] < len(cum_dist) else 0
        val = r.get('fy', 0) / 1000.0
        # Arrow text label
        fig.add_annotation(
            x=x_r, y=-0.1, ax=0, ay=30, # Tail starts lower
            text=f"R={val:.2f}", 
            showarrow=True, arrowhead=2, arrowsize=1.2, arrowwidth=2, arrowcolor='green',
            row=1, col=1
        )

    # 1.4 Loads (Improved Visuals)
    for l in loads:
        x_pos = l['x']
        mag = l['mag'] / 1000.0
        if l['type'] == 'P':
            # Point Load Arrow
            fig.add_annotation(x=x_pos, y=0.1, ax=0, ay=-40, text=f"P={mag:.2f}", 
                               showarrow=True, arrowhead=2, arrowcolor='red', row=1, col=1)
        elif l['type'] == 'U':
            # Distributed Load (Rectangular Block)
            x_end = x_pos + l['dist']
            # Draw a filled area for UDL
            fig.add_shape(type="rect",
                x0=x_pos, y0=0, x1=x_end, y1=0.2, # height of load block
                line=dict(color="orange", width=0),
                fillcolor="rgba(255, 165, 0, 0.3)",
                row=1, col=1
            )
            # Add label
            fig.add_annotation(x=(x_pos+x_end)/2, y=0.25, ax=0, ay=0, text=f"w={mag:.2f}", 
                               showarrow=False, font=dict(color='orange'), row=1, col=1)

    # --- ROW 2: SHEAR (SFD) ---
    y_shear = res['shear'] / 1000.0
    fig.add_trace(go.Scatter(x=res['x'], y=y_shear, mode='lines', fill='tozeroy', 
                             line=dict(color='#e74c3c', width=2), name='Shear'), row=2, col=1)
    
    # Restore Arrow Annotations for Max/Min
    v_max = y_shear.max()
    v_min = y_shear.min()
    x_vmax = res.loc[y_shear.idxmax(), 'x']
    x_vmin = res.loc[y_shear.idxmin(), 'x']
    
    fig.add_annotation(x=x_vmax, y=v_max, text=f"{v_max:.2f}", showarrow=True, arrowhead=1, ax=20, ay=-20, row=2, col=1)
    fig.add_annotation(x=x_vmin, y=v_min, text=f"{v_min:.2f}", showarrow=True, arrowhead=1, ax=20, ay=20, row=2, col=1)

    # --- ROW 3: MOMENT (BMD) - Inverted ---
    y_moment = res['moment'] / 1000.0
    fig.add_trace(go.Scatter(x=res['x'], y=y_moment, mode='lines', fill='tozeroy', 
                             line=dict(color='#2980b9', width=2), name='Moment'), row=3, col=1)
    
    m_max = y_moment.max()
    m_min = y_moment.min()
    x_mmax = res.loc[y_moment.idxmax(), 'x']
    x_mmin = res.loc[y_moment.idxmin(), 'x']

    fig.add_annotation(x=x_mmax, y=m_max, text=f"{m_max:.2f}", showarrow=True, arrowhead=1, ax=30, ay=-30, row=3, col=1)
    fig.add_annotation(x=x_mmin, y=m_min, text=f"{m_min:.2f}", showarrow=True, arrowhead=1, ax=30, ay=30, row=3, col=1)

    # Invert Y for BMD
    fig.update_yaxes(autorange="reversed", title_text="Moment (kNm)", row=3, col=1)

    # --- ROW 4: DEFLECTION ---
    y_def = res['deflection']
    fig.add_trace(go.Scatter(x=res['x'], y=y_def, mode='lines', 
                             line=dict(color='#27ae60', width=2), name='Deflection'), row=4, col=1)
    d_max = y_def.abs().max()
    x_dmax = res.loc[y_def.abs().idxmax(), 'x']
    # Show value at max deflection
    fig.add_annotation(x=x_dmax, y=y_def[y_def.abs().idxmax()], text=f"{d_max:.2f} mm", 
                       showarrow=True, arrowhead=2, ax=0, ay=40, row=4, col=1)

    # --- LAYOUT ---
    fig.update_layout(height=900, showlegend=False, plot_bgcolor='white', margin=dict(l=50, r=20, t=30, b=30))
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#f0f0f0', row=4, col=1, title_text="Length (m)")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#f0f0f0')
    fig.update_yaxes(showticklabels=False, row=1, col=1) # Hide Y axis on FBD

    # Grid lines for supports
    for x_line in cum_dist:
        fig.add_vline(x=x_line, line_width=1, line_dash="dash", line_color="gray", opacity=0.5)

    return fig
