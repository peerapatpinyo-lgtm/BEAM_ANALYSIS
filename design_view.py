import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

def plot_analysis_results(res, spans, sup_df, loads, reactions):
    """
    Professional Engineering Diagrams
    - FBD: Loads with transparency & arrows
    - SFD/BMD: Clear Zero Lines & Grids
    - Civil Convention (BMD Inverted)
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
    # 1.1 Beam Line (Main Element)
    fig.add_trace(go.Scatter(x=[0, total_len], y=[0, 0], mode='lines', 
                             line=dict(color='black', width=5), hoverinfo='skip'), row=1, col=1)

    # 1.2 Supports (Top Side)
    for s in sup_df.to_dict('records'):
        x_s = cum_dist[s['id']] if s['id'] < len(cum_dist) else 0
        fig.add_trace(go.Scatter(
            x=[x_s], y=[0.12], # Offset above beam
            mode='markers+text',
            marker=dict(symbol='triangle-down', size=14, color='#2c3e50'),
            text=[f"{s.get('type','Sup')}"], textposition="top center",
            hoverinfo='none', showlegend=False
        ), row=1, col=1)

    # 1.3 Reactions (Bottom Side)
    for r in reactions:
        x_r = cum_dist[r['node_id']] if r['node_id'] < len(cum_dist) else 0
        val = r.get('fy', 0) / 1000.0
        fig.add_annotation(
            x=x_r, y=-0.12, ax=0, ay=35, # Arrow pointing UP
            text=f"R={val:.2f}", 
            showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor='green',
            row=1, col=1
        )

    # 1.4 Loads (Improved Visualization)
    for l in loads:
        x_pos = l['x']
        mag = l['mag'] / 1000.0
        
        if l['type'] == 'P':
            # Point Load
            fig.add_annotation(x=x_pos, y=0.1, ax=0, ay=-40, text=f"P={mag:.2f}", 
                               showarrow=True, arrowhead=2, arrowcolor='red', row=1, col=1)
        
        elif l['type'] == 'U':
            # UDL: Use Transparent Block + Center Arrow
            x_end = x_pos + l['dist']
            
            # Transparent Block (To show beam underneath)
            fig.add_shape(type="rect",
                x0=x_pos, y0=0, x1=x_end, y1=0.25,
                line=dict(width=0),
                fillcolor="rgba(255, 165, 0, 0.2)", # Opacity 0.2
                row=1, col=1
            )
            
            # Top Line for UDL
            fig.add_trace(go.Scatter(
                x=[x_pos, x_end], y=[0.25, 0.25], 
                mode='lines', line=dict(color='orange', width=2), hoverinfo='skip'
            ), row=1, col=1)
            
            # Center Arrow & Label
            x_mid = (x_pos + x_end) / 2
            fig.add_annotation(
                x=x_mid, y=0.125, ax=0, ay=-20, # Small arrow pointing down inside the block
                text=f"w={mag:.2f}",
                showarrow=True, arrowhead=1, arrowcolor='orange', font=dict(color='orange'),
                row=1, col=1
            )

    # --- ROW 2: SHEAR (SFD) ---
    y_shear = res['shear'] / 1000.0
    fig.add_trace(go.Scatter(x=res['x'], y=y_shear, mode='lines', fill='tozeroy', 
                             line=dict(color='#e74c3c', width=2), name='Shear'), row=2, col=1)
    
    # Annotations
    v_max, v_min = y_shear.max(), y_shear.min()
    fig.add_annotation(x=res.loc[y_shear.idxmax(), 'x'], y=v_max, text=f"{v_max:.2f}", showarrow=True, arrowhead=1, ax=0, ay=-20, row=2, col=1)
    fig.add_annotation(x=res.loc[y_shear.idxmin(), 'x'], y=v_min, text=f"{v_min:.2f}", showarrow=True, arrowhead=1, ax=0, ay=20, row=2, col=1)

    # --- ROW 3: MOMENT (BMD) - Inverted ---
    y_moment = res['moment'] / 1000.0
    fig.add_trace(go.Scatter(x=res['x'], y=y_moment, mode='lines', fill='tozeroy', 
                             line=dict(color='#2980b9', width=2), name='Moment'), row=3, col=1)
    
    m_max, m_min = y_moment.max(), y_moment.min()
    fig.add_annotation(x=res.loc[y_moment.idxmax(), 'x'], y=m_max, text=f"{m_max:.2f}", showarrow=True, arrowhead=1, ax=0, ay=-30, row=3, col=1)
    fig.add_annotation(x=res.loc[y_moment.idxmin(), 'x'], y=m_min, text=f"{m_min:.2f}", showarrow=True, arrowhead=1, ax=0, ay=30, row=3, col=1)
    
    # Invert Y Axis (Civil Standard)
    fig.update_yaxes(autorange="reversed", title_text="Moment (kNm)", row=3, col=1)

    # --- ROW 4: DEFLECTION ---
    y_def = res['deflection']
    fig.add_trace(go.Scatter(x=res['x'], y=y_def, mode='lines', 
                             line=dict(color='#27ae60', width=2), name='Deflection'), row=4, col=1)
    d_max = y_def.abs().max()
    fig.add_annotation(x=res.loc[y_def.abs().idxmax(), 'x'], y=y_def[y_def.abs().idxmax()], 
                       text=f"{d_max:.2f} mm", showarrow=True, arrowhead=2, ax=0, ay=40, row=4, col=1)

    # --- GLOBAL STYLING ---
    fig.update_layout(
        height=1000, 
        showlegend=False, 
        plot_bgcolor='white', 
        paper_bgcolor='white',
        margin=dict(l=60, r=20, t=30, b=30)
    )
    
    # Add Supports Vertical Lines
    for x_line in cum_dist:
        fig.add_vline(x=x_line, line_width=1, line_dash="dash", line_color="gray", opacity=0.5)

    # Enhanced Grid & Zero Lines
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#f0f0f0', title_text="Length (m)", row=4, col=1)
    
    # Update all Y Axes to have a strong Zero Line
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#f0f0f0', 
                     zeroline=True, zerolinewidth=2, zerolinecolor='black')
    
    # Hide Y-ticks for FBD
    fig.update_yaxes(showticklabels=False, zeroline=False, row=1, col=1)

    return fig
