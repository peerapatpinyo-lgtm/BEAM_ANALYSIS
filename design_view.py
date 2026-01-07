import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

def plot_analysis_results(res, spans, sup_df, loads, reactions):
    """
    Plots Analysis with:
    - Real Support Icons (Pin/Fixed)
    - Moment on Tension Side (Inverted Y)
    - Loads & Reactions
    """
    cum_dist = [0] + list(np.cumsum(spans))
    total_len = cum_dist[-1]
    
    # Create 4 Subplots
    fig = make_subplots(
        rows=4, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.08,
        subplot_titles=(
            "1. Free Body Diagram (FBD)", 
            "2. Shear Force (SFD)", 
            "3. Bending Moment (Tension Side)", 
            "4. Deflection"
        ),
        row_heights=[0.2, 0.25, 0.25, 0.2]
    )

    # --- ROW 1: FBD & SUPPORTS ---
    # Beam Line
    fig.add_trace(go.Scatter(x=[0, total_len], y=[0, 0], mode='lines', 
                             line=dict(color='black', width=4), hoverinfo='skip'), row=1, col=1)

    # Draw Supports (Icons)
    sup_data = sup_df.to_dict('records')
    for s in sup_data:
        # Determine X position
        if 'span_index' in s and s['span_index'] < len(spans): 
            # If specified by span index (not implemented fully in simple app, assuming node based)
            pass
            
        # Map node index to x (Assuming linear sequential nodes)
        # Note: In this simple solver, supports are at nodes 0, 1, 2...
        if s['id'] < len(cum_dist):
            x_s = cum_dist[s['id']]
            sType = s.get('type', 'Pin')
            
            if sType == 'Fixed':
                # Draw a vertical block
                fig.add_shape(type="rect",
                    x0=x_s-0.1, y0=-0.5, x1=x_s+0.1, y1=0.5,
                    line=dict(color="black", width=2), fillcolor="gray",
                    row=1, col=1
                )
            else: # Pin or Roller -> Triangle
                fig.add_trace(go.Scatter(
                    x=[x_s], y=[-0.1],
                    mode='markers',
                    marker=dict(symbol='triangle-up', size=15, color='black'),
                    hoverinfo='name', name=f'Support {s["id"]}'
                ), row=1, col=1)

    # Draw Loads
    for l in loads:
        x_pos = l['x']
        mag_kN = l['mag'] / 1000.0
        if l['type'] == 'P':
            fig.add_annotation(x=x_pos, y=0, ax=0, ay=-40, text=f"P={mag_kN:.2f}", 
                               showarrow=True, arrowhead=2, arrowcolor='red', row=1, col=1)
        elif l['type'] == 'U':
            x_end = x_pos + l['dist']
            fig.add_trace(go.Scatter(x=[x_pos, x_end], y=[0.5, 0.5], mode='lines', 
                                     line=dict(color='orange', width=2, dash='dot'), hoverinfo='skip'), row=1, col=1)
            fig.add_annotation(x=(x_pos+x_end)/2, y=0, ax=0, ay=-30, text=f"w={mag_kN:.2f}", 
                               showarrow=True, arrowhead=2, arrowcolor='orange', row=1, col=1)

    # Draw Reactions
    for r in reactions:
        x_r = cum_dist[r['node_id']] if r['node_id'] < len(cum_dist) else 0
        val = r.get('fy', 0) / 1000.0
        fig.add_annotation(x=x_r, y=0, ax=0, ay=40, text=f"R={val:.2f}", 
                           showarrow=True, arrowhead=2, arrowcolor='green', row=1, col=1)
    
    fig.update_yaxes(range=[-1, 2], showticklabels=False, row=1, col=1)

    # --- ROW 2: SHEAR ---
    y_shear = res['shear'] / 1000.0
    fig.add_trace(go.Scatter(x=res['x'], y=y_shear, mode='lines', fill='tozeroy', 
                             line=dict(color='#e74c3c', width=2), name='Shear'), row=2, col=1)
    
    # --- ROW 3: MOMENT (INVERTED Y) ---
    y_moment = res['moment'] / 1000.0
    fig.add_trace(go.Scatter(x=res['x'], y=y_moment, mode='lines', fill='tozeroy', 
                             line=dict(color='#3498db', width=2), name='Moment'), row=3, col=1)
    # Highlight Max/Min
    max_m, min_m = y_moment.max(), y_moment.min()
    fig.add_annotation(x=res.loc[y_moment.idxmax(), 'x'], y=max_m, text=f"{max_m:.2f}", showarrow=False, yshift=10, row=3, col=1)
    fig.add_annotation(x=res.loc[y_moment.idxmin(), 'x'], y=min_m, text=f"{min_m:.2f}", showarrow=False, yshift=-10, row=3, col=1)

    # Invert Y axis for Moment (Civil Convention: Tension Side)
    fig.update_yaxes(autorange="reversed", title_text="M (kNm)", row=3, col=1)

    # --- ROW 4: DEFLECTION ---
    y_def = res['deflection']
    fig.add_trace(go.Scatter(x=res['x'], y=y_def, mode='lines', 
                             line=dict(color='#2ecc71', width=2), name='Deflection'), row=4, col=1)

    # Global
    for x_s in cum_dist:
        fig.add_vline(x=x_s, line_width=1, line_dash="dash", line_color="lightgray")

    fig.update_layout(height=1000, showlegend=False, margin=dict(l=50, r=20, t=40, b=40), paper_bgcolor='white', plot_bgcolor='white')
    fig.update_xaxes(title_text="Length (m)", row=4, col=1)
    
    return fig
