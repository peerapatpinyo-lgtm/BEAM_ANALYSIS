import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

def plot_analysis_results(res, spans, sup_input, loads, reactions):
    cum_dist = [0] + list(np.cumsum(spans))
    total_len = cum_dist[-1]
    
    # Handle Reactions Input
    reac_map = {}
    if isinstance(reactions, list):
        for r in reactions: reac_map[int(r['node_id'])] = r['fy']
    elif isinstance(reactions, dict):
        reac_map = {int(k): v for k, v in reactions.items()}

    # Handle Supports Input
    supports = sup_input if isinstance(sup_input, list) else []
    if isinstance(sup_input, pd.DataFrame): supports = sup_input.to_dict('records')

    fig = make_subplots(
        rows=3, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.1,
        subplot_titles=("1. FBD & Loads", "2. Shear Force (kN)", "3. Bending Moment (kNm)"),
        row_heights=[0.3, 0.35, 0.35]
    )

    # --- ROW 1: FBD ---
    fig.add_trace(go.Scatter(x=[0, total_len], y=[0, 0], mode='lines', line=dict(color='black', width=4), hoverinfo='skip'), row=1, col=1)

    # Supports & Reactions
    for s in supports:
        nid = int(s.get('id', 0))
        if nid < len(cum_dist):
            x_s = cum_dist[nid]
            # Support Triangle
            fig.add_trace(go.Scatter(
                x=[x_s], y=[-0.1], mode='markers+text',
                marker=dict(symbol='triangle-up', size=18, color='#34495e'),
                text=[s.get('type','')], textposition="bottom center", showlegend=False
            ), row=1, col=1)
            
            # Reaction Arrow & Label
            val = reac_map.get(nid, 0)
            if abs(val) > 1: # Threshold to show
                fig.add_annotation(
                    x=x_s, y=-0.4, # Positioned lower
                    text=f"R={val/1000:.2f}",
                    showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor='green',
                    ax=0, ay=35, row=1, col=1
                )

    # Loads
    for l in loads:
        x_pos = l['x']; mag = l['mag']/1000.0
        if l['type'] == 'P':
            fig.add_annotation(x=x_pos, y=0.4, text=f"{mag:.2f}", showarrow=True, arrowhead=2, arrowcolor='red', ax=0, ay=-35, row=1, col=1)
        elif l['type'] == 'U':
            dist = l.get('dist',0); x_end = x_pos + dist
            fig.add_trace(go.Scatter(x=[x_pos, x_end], y=[0.4, 0.4], mode='lines', line=dict(color='orange'), showlegend=False), row=1, col=1)
            fig.add_annotation(x=(x_pos+x_end)/2, y=0.5, text=f"w={mag:.2f}", showarrow=False, font=dict(color='orange'), row=1, col=1)
            for xa in np.linspace(x_pos, x_end, max(3, int(dist)+1)):
                fig.add_annotation(x=xa, y=0, ax=0, ay=-25, arrowhead=1, arrowcolor='orange', row=1, col=1)

    # --- ROW 2: SFD ---
    y_s = res['shear']/1000.0
    fig.add_trace(go.Scatter(x=res['x'], y=y_s, mode='lines', fill='tozeroy', line=dict(color='#e74c3c'), name='Shear'), row=2, col=1)
    if not y_s.empty:
        fig.add_annotation(x=res['x'][y_s.idxmax()], y=y_s.max(), text=f"{y_s.max():.2f}", showarrow=False, yshift=10, row=2, col=1)
        fig.add_annotation(x=res['x'][y_s.idxmin()], y=y_s.min(), text=f"{y_s.min():.2f}", showarrow=False, yshift=-10, row=2, col=1)

    # --- ROW 3: BMD ---
    y_m = res['moment']/1000.0
    fig.add_trace(go.Scatter(x=res['x'], y=y_m, mode='lines', fill='tozeroy', line=dict(color='#2980b9'), name='Moment'), row=3, col=1)
    if not y_m.empty:
        fig.add_annotation(x=res['x'][y_m.idxmax()], y=y_m.max(), text=f"{y_m.max():.2f}", showarrow=True, arrowhead=1, ay=30, row=3, col=1)
        fig.add_annotation(x=res['x'][y_m.idxmin()], y=y_m.min(), text=f"{y_m.min():.2f}", showarrow=True, arrowhead=1, ay=-30, row=3, col=1)
    
    fig.update_yaxes(autorange="reversed", row=3, col=1)

    # --- LAYOUT ---
    fig.update_layout(height=900, showlegend=False, plot_bgcolor='white', margin=dict(l=60, r=20, t=50, b=50))
    # Explicit Axis Labels
    fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False, range=[-1, 1], row=1, col=1)
    fig.update_yaxes(title_text="Shear (kN)", showgrid=True, zeroline=True, zerolinewidth=2, zerolinecolor='black', row=2, col=1)
    fig.update_yaxes(title_text="Moment (kNm)", showgrid=True, zeroline=True, zerolinewidth=2, zerolinecolor='black', row=3, col=1)
    fig.update_xaxes(title_text="Length (m)", showgrid=True, row=3, col=1)

    return fig
