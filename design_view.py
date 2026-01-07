import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

def plot_analysis_results(res, spans, sup_df, loads, reactions):
    """
    World-Class Engineering Visualization
    - Correct Vector directions for Loads (Pointing Down to Beam)
    - Full Axis Labeling (SI Units)
    - Clean Aesthetic matching Technical Publications
    """
    cum_dist = [0] + list(np.cumsum(spans))
    total_len = cum_dist[-1]
    
    # Pre-process for exact visualization
    y_shear = res['shear'].values / 1000.0
    y_moment = res['moment'].values / 1000.0
    y_def = res['deflection'].values

    # Define Subplots
    fig = make_subplots(
        rows=4, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.08,
        subplot_titles=(
            "1. Free Body Diagram (FBD)", 
            "2. Shear Force Diagram (SFD)", 
            "3. Bending Moment Diagram (BMD)", 
            "4. Deflection Diagram"
        ),
        row_heights=[0.3, 0.22, 0.22, 0.26]
    )

    # --- ROW 1: FREE BODY DIAGRAM (FBD) ---
    # 1.1 The Beam (Neutral Axis)
    fig.add_trace(go.Scatter(x=[0, total_len], y=[0, 0], mode='lines', 
                             line=dict(color='black', width=5), hoverinfo='skip'), row=1, col=1)

    # 1.2 Supports (Pinned/Roller)
    for s in sup_df.to_dict('records'):
        x_s = cum_dist[s['id']] if s['id'] < len(cum_dist) else 0
        fig.add_trace(go.Scatter(
            x=[x_s], y=[-0.02], # Touching bottom of beam
            mode='markers+text',
            marker=dict(symbol='triangle-up', size=16, color='#2c3e50', line=dict(width=1, color='black')),
            text=[f"{s.get('type','Sup')}"], textposition="bottom center",
            hoverinfo='none', showlegend=False
        ), row=1, col=1)

    # 1.3 Reactions (Green Arrows UP)
    for r in reactions:
        x_r = cum_dist[r['node_id']] if r['node_id'] < len(cum_dist) else 0
        val = r.get('fy', 0) / 1000.0
        # Annotation: Head at (x, 0), Tail below
        fig.add_annotation(
            x=x_r, y=-0.15, ax=0, ay=40, # ay positive moves tail DOWN relative to head? No, text relative to point.
            # Let's use strict coordinates for Engineering precision
            # Workaround: Pointing UP to the support
            text=f"R={val:.2f} kN", 
            showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor='#27ae60',
            row=1, col=1
        )

    # 1.4 Loads (CORRECTED: Pointing DOWN to Beam)
    for l in loads:
        x_pos = l['x']
        mag = l['mag'] / 1000.0
        
        if l['type'] == 'P':
            # Point Load: Head at (x, 0), Tail above
            fig.add_annotation(
                x=x_pos, y=0, ax=0, ay=-50, # ay negative = tail is ABOVE head
                text=f"P={mag:.2f} kN", 
                showarrow=True, arrowhead=2, arrowcolor='#c0392b', 
                row=1, col=1
            )
        
        elif l['type'] == 'U':
            # Uniform Load: 
            # 1. Draw a "Load Line" floating above
            load_h = 0.25 # Height of load representation
            x_end = x_pos + l['dist']
            
            # The horizontal bar of the UDL
            fig.add_trace(go.Scatter(
                x=[x_pos, x_end], y=[load_h, load_h], 
                mode='lines', line=dict(color='#e67e22', width=2), hoverinfo='skip'
            ), row=1, col=1)
            
            # The vertical arrows (Comb pattern)
            n_arrows = max(3, int(l['dist'] * 3))
            x_arrows = np.linspace(x_pos, x_end, n_arrows)
            for xa in x_arrows:
                fig.add_annotation(
                    x=xa, y=0, # Head at beam
                    ax=0, ay=-40, # Tail above beam (approx at load_h)
                    showarrow=True, arrowhead=1, arrowwidth=1.5, arrowcolor='#e67e22', arrowsize=0.8,
                    row=1, col=1
                )
            
            # Label
            fig.add_annotation(
                x=(x_pos+x_end)/2, y=load_h, ax=0, ay=-20,
                text=f"w={mag:.2f} kN/m", font=dict(color='#e67e22'), showarrow=False,
                row=1, col=1
            )

    # --- ROW 2: SHEAR (SFD) ---
    fig.add_trace(go.Scatter(x=res['x'], y=y_shear, mode='lines', fill='tozeroy', 
                             line=dict(color='#e74c3c', width=2), name='Shear'), row=2, col=1)
    
    # Peak Labels
    v_max, v_min = y_shear.max(), y_shear.min()
    fig.add_annotation(x=res.iloc[y_shear.argmax()]['x'], y=v_max, text=f"{v_max:.2f}", showarrow=True, ax=0, ay=-20, row=2, col=1)
    fig.add_annotation(x=res.iloc[y_shear.argmin()]['x'], y=v_min, text=f"{v_min:.2f}", showarrow=True, ax=0, ay=20, row=2, col=1)

    # --- ROW 3: MOMENT (BMD) ---
    fig.add_trace(go.Scatter(x=res['x'], y=y_moment, mode='lines', fill='tozeroy', 
                             line=dict(color='#2980b9', width=2), name='Moment'), row=3, col=1)
    
    # Peak Label
    idx_m = np.argmax(np.abs(y_moment))
    m_val = y_moment[idx_m]
    fig.add_annotation(x=res.iloc[idx_m]['x'], y=m_val, text=f"{m_val:.2f}", showarrow=True, ax=0, ay=30 if m_val > 0 else -30, row=3, col=1)
    
    # Invert Y (Engineering Convention)
    fig.update_yaxes(autorange="reversed", row=3, col=1)

    # --- ROW 4: DEFLECTION ---
    fig.add_trace(go.Scatter(x=res['x'], y=y_def, mode='lines', 
                             line=dict(color='#8e44ad', width=2), name='Deflection'), row=4, col=1)
    d_max = np.min(y_def)
    fig.add_annotation(x=res.iloc[np.argmin(y_def)]['x'], y=d_max, text=f"{d_max:.2f} mm", showarrow=True, ax=0, ay=40, row=4, col=1)

    # --- LAYOUT & AXIS LABELS ---
    fig.update_layout(height=1100, showlegend=False, plot_bgcolor='white', margin=dict(t=40, b=40, l=80, r=20))
    
    # 1. FBD Axes (Hidden Y, Show X)
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, row=1, col=1)
    
    # 2. SFD Axes
    fig.update_yaxes(title_text="Shear Force (kN)", showgrid=True, gridcolor='#eee', zeroline=True, zerolinewidth=2, zerolinecolor='black', row=2, col=1)
    
    # 3. BMD Axes
    fig.update_yaxes(title_text="Moment (kNm)", showgrid=True, gridcolor='#eee', zeroline=True, zerolinewidth=2, zerolinecolor='black', row=3, col=1)
    
    # 4. Deflection Axes
    fig.update_yaxes(title_text="Deflection (mm)", showgrid=True, gridcolor='#eee', zeroline=True, zerolinewidth=1, zerolinecolor='black', row=4, col=1)
    fig.update_xaxes(title_text="Distance along Beam (m)", showgrid=True, row=4, col=1)

    # Add Support Lines
    for x_line in cum_dist:
        fig.add_vline(x=x_line, line_width=1, line_dash="dash", line_color="gray", opacity=0.5)

    return fig
