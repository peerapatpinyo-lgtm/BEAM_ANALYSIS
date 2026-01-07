import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

def plot_analysis_results(res, spans, sup_df, loads, reactions):
    """
    Plots 4 Diagrams with Textbook-style Supports and detailed styling.
    """
    cum_dist = [0] + list(np.cumsum(spans))
    total_len = cum_dist[-1]
    
    # Create 4 Subplots
    fig = make_subplots(
        rows=4, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.08,
        subplot_titles=(
            "1. Load & Reaction Model (FBD)", 
            "2. Shear Force Diagram (SFD)", 
            "3. Bending Moment Diagram (BMD)", 
            "4. Deflection (mm)"
        ),
        row_heights=[0.3, 0.25, 0.25, 0.2]
    )

    # --- ROW 1: FBD (Textbook Style) ---
    # 1.1 Beam Line
    fig.add_trace(go.Scatter(x=[0, total_len], y=[0, 0], mode='lines', 
                             line=dict(color='black', width=4), hoverinfo='skip'), row=1, col=1)
    
    # 1.2 Supports (Custom Shapes)
    # Map node index to x position
    node_x_map = {i: x for i, x in enumerate(cum_dist)}
    
    if isinstance(reactions, pd.DataFrame): reac_dict = reactions.set_index('node_id')['fy'].to_dict() # Fallback
    elif isinstance(reactions, dict): reac_dict = reactions
    else: reac_dict = {}

    sup_list = sup_df.to_dict('records')
    
    for s in sup_list:
        nid = s.get('id', s.get('node_id'))
        x_pos = node_x_map[nid]
        stype = s['type']
        
        # Draw Support Shapes
        if stype == 'Pin':
            # Triangle
            fig.add_trace(go.Scatter(
                x=[x_pos-0.15, x_pos, x_pos+0.15, x_pos-0.15], 
                y=[-0.3, 0, -0.3, -0.3], 
                fill='toself', line=dict(color='black', width=2), fillcolor='white',
                mode='lines', showlegend=False, hoverinfo='skip'
            ), row=1, col=1)
            # Hatch lines for ground
            for k in np.linspace(-0.15, 0.15, 5):
                fig.add_trace(go.Scatter(x=[x_pos+k, x_pos+k-0.05], y=[-0.3, -0.4], mode='lines', line=dict(color='black', width=1), showlegend=False), row=1, col=1)

        elif stype == 'Roller':
            # Circle
            fig.add_shape(type="circle", x0=x_pos-0.15, y0=-0.3, x1=x_pos+0.15, y1=0, line=dict(color="black", width=2), fillcolor="white", row=1, col=1)
            # Line below
            fig.add_trace(go.Scatter(x=[x_pos-0.2, x_pos+0.2], y=[-0.3, -0.3], mode='lines', line=dict(color='black', width=2), showlegend=False), row=1, col=1)
            # Ground Hatches
            for k in np.linspace(-0.2, 0.2, 6):
                fig.add_trace(go.Scatter(x=[x_pos+k, x_pos+k-0.05], y=[-0.3, -0.4], mode='lines', line=dict(color='black', width=1), showlegend=False), row=1, col=1)

        elif stype == 'Fixed':
            # Vertical Line
            fig.add_trace(go.Scatter(x=[x_pos, x_pos], y=[-0.3, 0.3], mode='lines', line=dict(color='black', width=4), showlegend=False), row=1, col=1)
            # Hatches (Left or Right depending on pos)
            direction = -1 if x_pos == 0 else 1
            for k in np.linspace(-0.3, 0.3, 7):
                fig.add_trace(go.Scatter(x=[x_pos, x_pos + direction*0.15], y=[k, k-0.05], mode='lines', line=dict(color='black', width=1), showlegend=False), row=1, col=1)

    # 1.3 Loads
    max_load_scale = 1.0 # Scale factor for arrows
    for l in loads:
        x_pos = l['x'] + (cum_dist[l['span_index']] if 'span_index' in l else 0)
        mag_val = l['mag'] / 1000.0
        
        if l['type'] == 'P':
            fig.add_annotation(
                x=x_pos, y=0, ax=0, ay=-50,
                text=f"<b>{mag_val:.1f}kN</b>", showarrow=True, arrowhead=2, arrowsize=1.5, arrowwidth=2, arrowcolor='red',
                row=1, col=1
            )
        elif l['type'] == 'U':
            x_end = x_pos + l['dist']
            # Draw distributed arrows
            for x_arr in np.linspace(x_pos, x_end, 5):
                fig.add_annotation(x=x_arr, y=0, ax=0, ay=-30, arrowhead=2, arrowsize=1, arrowwidth=1, arrowcolor='orange', row=1, col=1, text="")
            # Label
            fig.add_annotation(x=(x_pos+x_end)/2, y=0.5, text=f"<b>w={mag_val:.1f} kN/m</b>", showarrow=False, font=dict(color='orange'), row=1, col=1)
            # Line block
            fig.add_trace(go.Scatter(x=[x_pos, x_end], y=[0.5, 0.5], mode='lines', line=dict(color='orange', width=1), showlegend=False), row=1, col=1)
            fig.add_trace(go.Scatter(x=[x_pos, x_pos], y=[0, 0.5], mode='lines', line=dict(color='orange', width=1), showlegend=False), row=1, col=1)
            fig.add_trace(go.Scatter(x=[x_end, x_end], y=[0, 0.5], mode='lines', line=dict(color='orange', width=1), showlegend=False), row=1, col=1)

    # 1.4 Reactions Labels
    for nid, val in reac_dict.items():
        x_r = node_x_map.get(nid, 0)
        r_kn = val / 1000.0
        if abs(r_kn) > 0.01:
            fig.add_annotation(
                x=x_r, y=-0.5, ax=0, ay=30,
                text=f"<b>R={r_kn:.2f}</b>", showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor='green',
                row=1, col=1
            )

    fig.update_yaxes(range=[-1.5, 1.5], visible=False, row=1, col=1)


    # --- ROW 2: SHEAR (SFD) ---
    y_shear = res['shear'] / 1000.0
    fig.add_trace(go.Scatter(x=res['x'], y=y_shear, mode='lines', fill='tozeroy', 
                             line=dict(color='#e74c3c', width=2), name='Shear (kN)'), row=2, col=1)
    # Annotate Max/Min Shear
    if len(y_shear) > 0:
        idx_max, idx_min = y_shear.idxmax(), y_shear.idxmin()
        fig.add_annotation(x=res.loc[idx_max, 'x'], y=y_shear[idx_max], text=f"{y_shear[idx_max]:.2f}", showarrow=False, yshift=10, row=2, col=1)
        fig.add_annotation(x=res.loc[idx_min, 'x'], y=y_shear[idx_min], text=f"{y_shear[idx_min]:.2f}", showarrow=False, yshift=-10, row=2, col=1)


    # --- ROW 3: MOMENT (BMD) + TENSION SIDE ---
    y_moment = res['moment'] / 1000.0
    # Standard Math: Positive Up. 
    # Civil Eng: Positive Moment (Sagging) -> Tension Bottom. Negative Moment (Hogging) -> Tension Top.
    # We plot standard (+ Up) but colour code and label the tension side.
    
    # Split Positive (Sagging) and Negative (Hogging) for coloring
    pos_m = y_moment.clip(lower=0)
    neg_m = y_moment.clip(upper=0)

    fig.add_trace(go.Scatter(x=res['x'], y=pos_m, mode='lines', fill='tozeroy', 
                             line=dict(color='#3498db', width=0), fillcolor='rgba(52, 152, 219, 0.3)',
                             name='Sagging (+)', showlegend=False), row=3, col=1)
    
    fig.add_trace(go.Scatter(x=res['x'], y=neg_m, mode='lines', fill='tozeroy', 
                             line=dict(color='#e67e22', width=0), fillcolor='rgba(230, 126, 34, 0.3)',
                             name='Hogging (-)', showlegend=False), row=3, col=1)

    fig.add_trace(go.Scatter(x=res['x'], y=y_moment, mode='lines', 
                             line=dict(color='blue', width=2), name='Moment (kNm)'), row=3, col=1)

    # Label Tension Zones
    max_m = y_moment.max()
    min_m = y_moment.min()
    
    # Place text indicating where to put steel
    if max_m > 0.1:
        fig.add_annotation(x=res.loc[y_moment.idxmax(), 'x'], y=max_m, text=f"Max +{max_m:.2f}<br>(Bot Steel)", showarrow=True, arrowhead=1, row=3, col=1)
    if min_m < -0.1:
        fig.add_annotation(x=res.loc[y_moment.idxmin(), 'x'], y=min_m, text=f"Max {min_m:.2f}<br>(Top Steel)", showarrow=True, arrowhead=1, ax=20, ay=20, row=3, col=1)


    # --- ROW 4: DEFLECTION ---
    # Solver outputs deflection in mm directly now.
    y_def = res['deflection']
    fig.add_trace(go.Scatter(x=res['x'], y=y_def, mode='lines', 
                             line=dict(color='#2ecc71', width=2), name='Def (mm)'), row=4, col=1)
    
    # Max Deflection Label
    if len(y_def) > 0:
        idx_def = y_def.abs().idxmax()
        val_def = y_def.iloc[idx_def]
        fig.add_annotation(x=res.loc[idx_def, 'x'], y=val_def, text=f"<b>Max: {val_def:.2f} mm</b>", showarrow=True, row=4, col=1)


    # --- GLOBAL LAYOUT ---
    for x_s in cum_dist:
        fig.add_vline(x=x_s, line_width=1, line_dash="dash", line_color="lightgray")

    fig.update_layout(
        height=1000, 
        showlegend=False, 
        margin=dict(l=60, r=20, t=50, b=50), 
        paper_bgcolor='white',
        plot_bgcolor='white',
        hovermode="x unified"
    )
    
    fig.update_yaxes(title_text="Load (kN)", row=1, col=1)
    fig.update_yaxes(title_text="V (kN)", row=2, col=1)
    fig.update_yaxes(title_text="M (kNm)", row=3, col=1)
    fig.update_yaxes(title_text="Def (mm)", row=4, col=1)
    fig.update_xaxes(title_text="Distance (m)", row=4, col=1)

    return fig
