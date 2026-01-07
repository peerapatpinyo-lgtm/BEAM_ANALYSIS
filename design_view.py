design_view.py
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

def plot_analysis_results(res, spans, sup_df, loads, reactions):
    """
    Plots 4 Diagrams:
    1. Load & Reaction Model (FBD)
    2. Shear Force (SFD)
    3. Bending Moment (BMD)
    4. Deflection
    """
    cum_dist = [0] + list(np.cumsum(spans))
    total_len = cum_dist[-1]
    
    # Create 4 Subplots
    fig = make_subplots(
        rows=4, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.05,
        subplot_titles=(
            "1. Load & Reaction Diagram (FBD)", 
            "2. Shear Force Diagram (SFD)", 
            "3. Bending Moment Diagram (BMD)", 
            "4. Deflection"
        ),
        row_heights=[0.25, 0.25, 0.25, 0.25]
    )

    # --- ROW 1: LOAD & REACTION DIAGRAM ---
    # 1.1 Draw Beam Line
    fig.add_trace(go.Scatter(x=[0, total_len], y=[0, 0], mode='lines', 
                             line=dict(color='black', width=5), hoverinfo='skip'), row=1, col=1)
    
    # 1.2 Draw Loads (Arrows)
    max_load_mag = 0
    if loads:
        max_load_mag = max([l['mag'] for l in loads]) if loads else 1000
    
    for l in loads:
        x_pos = l['x']
        mag_kN = l['mag'] / 1000.0
        
        if l['type'] == 'P':
            # Point Load Arrow (Down)
            fig.add_annotation(
                x=x_pos, y=0, ax=0, ay=-40,
                text=f"P={mag_kN:.2f}", showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor='red',
                row=1, col=1
            )
        elif l['type'] == 'U':
            # UDL (Representation line + label)
            x_end = x_pos + l['dist']
            x_mid = (x_pos + x_end) / 2
            # Draw a block or line to represent UDL
            fig.add_trace(go.Scatter(
                x=[x_pos, x_end], y=[0.5, 0.5], mode='lines', 
                line=dict(color='orange', width=2, dash='dot'), hoverinfo='skip'
            ), row=1, col=1)
            fig.add_annotation(
                x=x_mid, y=0, ax=0, ay=-30,
                text=f"w={mag_kN:.2f}", showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor='orange',
                row=1, col=1
            )

    # 1.3 Draw Reactions (Arrows Up)
    # Convert reactions to list of dicts if DataFrame
    if isinstance(reactions, pd.DataFrame):
        reac_list = reactions.to_dict('records')
    elif isinstance(reactions, list):
        reac_list = reactions
    else:
        reac_list = []

    # Map node index to x position
    node_x_map = {i: x for i, x in enumerate(cum_dist)}
    
    for r in reac_list:
        # Check structure of r
        if 'node_id' in r: nid = r['node_id']
        elif 'id' in r: nid = r['id']
        else: continue # Skip if unknown format
            
        # Get reaction value
        val = r.get('fy', 0) / 1000.0
        x_r = node_x_map.get(nid, 0)
        
        # Arrow Up
        fig.add_annotation(
            x=x_r, y=0, ax=0, ay=40,
            text=f"R={val:.2f}", showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor='green',
            row=1, col=1
        )

    # Set Y-range for Load Diagram to make it look spacious
    fig.update_yaxes(range=[-1, 2], showticklabels=False, row=1, col=1)


    # --- ROW 2: SHEAR (SFD) ---
    y_shear = res['shear'] / 1000.0
    fig.add_trace(go.Scatter(x=res['x'], y=y_shear, mode='lines', fill='tozeroy', 
                             line=dict(color='#e74c3c', width=2), name='Shear (kN)'), row=2, col=1)
    # Annotations
    max_v, min_v = y_shear.max(), y_shear.min()
    if not pd.isna(max_v): fig.add_annotation(x=res.loc[y_shear.idxmax(), 'x'], y=max_v, text=f"{max_v:.2f}", showarrow=False, yshift=10, row=2, col=1)
    if not pd.isna(min_v): fig.add_annotation(x=res.loc[y_shear.idxmin(), 'x'], y=min_v, text=f"{min_v:.2f}", showarrow=False, yshift=-10, row=2, col=1)


    # --- ROW 3: MOMENT (BMD) ---
    y_moment = res['moment'] / 1000.0
    fig.add_trace(go.Scatter(x=res['x'], y=y_moment, mode='lines', fill='tozeroy', 
                             line=dict(color='#3498db', width=2), name='Moment (kNm)'), row=3, col=1)
    # Annotations
    max_m, min_m = y_moment.max(), y_moment.min()
    if not pd.isna(max_m): fig.add_annotation(x=res.loc[y_moment.idxmax(), 'x'], y=max_m, text=f"{max_m:.2f}", showarrow=False, yshift=10, row=3, col=1)
    if not pd.isna(min_m): fig.add_annotation(x=res.loc[y_moment.idxmin(), 'x'], y=min_m, text=f"{min_m:.2f}", showarrow=False, yshift=-10, row=3, col=1)


    # --- ROW 4: DEFLECTION ---
    y_def = res['deflection']
    fig.add_trace(go.Scatter(x=res['x'], y=y_def, mode='lines', 
                             line=dict(color='#2ecc71', width=2), name='Def (mm)'), row=4, col=1)
    # Annotation
    if len(y_def) > 0:
        idx_def = y_def.abs().idxmax()
        val_def = y_def.iloc[idx_def]
        fig.add_annotation(x=res.loc[idx_def, 'x'], y=val_def, text=f"{val_def:.2f}", showarrow=True, row=4, col=1)


    # --- GLOBAL FORMATTING ---
    # Add vertical dashed lines at supports
    for x_s in cum_dist:
        fig.add_vline(x=x_s, line_width=1, line_dash="dash", line_color="gray")

    fig.update_layout(
        height=900, # Taller for 4 rows
        showlegend=False, 
        margin=dict(l=50, r=20, t=40, b=40), 
        paper_bgcolor='white',
        plot_bgcolor='white'
    )
    
    # Axes Titles
    fig.update_yaxes(title_text="F (kN)", row=1, col=1)
    fig.update_yaxes(title_text="V (kN)", row=2, col=1)
    fig.update_yaxes(title_text="M (kNm)", row=3, col=1)
    fig.update_yaxes(title_text="Def (mm)", row=4, col=1)
    fig.update_xaxes(title_text="Distance (m)", row=4, col=1)

    return fig
