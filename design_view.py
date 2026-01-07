import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

def plot_analysis_results(res, spans, sup_df, loads, reactions):
    """
    Plots FBD, SFD, BMD (Tension Side), Deflection
    """
    cum_dist = [0] + list(np.cumsum(spans))
    total_len = cum_dist[-1]
    
    # Scale helper for supports and arrows
    S_SCALE = max(total_len * 0.04, 0.4) 

    fig = make_subplots(
        rows=4, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.08,
        subplot_titles=(
            "1. Loading Diagram (FBD)", 
            "2. Shear Force Diagram (SFD)", 
            "3. Bending Moment Diagram (BMD)", 
            "4. Deflection (mm)"
        ),
        row_heights=[0.3, 0.25, 0.25, 0.2]
    )

    # --- ROW 1: FBD ---
    # Beam
    fig.add_trace(go.Scatter(x=[0, total_len], y=[0, 0], mode='lines', 
                             line=dict(color='black', width=5), hoverinfo='skip'), row=1, col=1)

    # Supports
    node_x_map = {i: x for i, x in enumerate(cum_dist)}
    sup_list = sup_df.to_dict('records')
    
    for s in sup_list:
        nid = s.get('id', s.get('node_id'))
        x_pos = node_x_map[nid]
        stype = s['type']
        
        w = S_SCALE * 0.6
        h = S_SCALE * 0.8
        
        if stype == 'Pin':
            fig.add_trace(go.Scatter(
                x=[x_pos-w, x_pos, x_pos+w, x_pos-w], 
                y=[-h, 0, -h, -h], 
                fill='toself', line=dict(color='#34495e', width=2), fillcolor='white',
                mode='lines', showlegend=False, hoverinfo='skip'
            ), row=1, col=1)
        elif stype == 'Roller':
            r = w
            fig.add_shape(type="circle", x0=x_pos-r, y0=-2*r, x1=x_pos+r, y1=0, 
                          line=dict(color="#34495e", width=2), fillcolor="white", row=1, col=1)
            fig.add_trace(go.Scatter(x=[x_pos-w*1.5, x_pos+w*1.5], y=[-2*r, -2*r], 
                                     mode='lines', line=dict(color='#34495e', width=2), showlegend=False), row=1, col=1)
        elif stype == 'Fixed':
            fig.add_trace(go.Scatter(x=[x_pos, x_pos], y=[-h, h], 
                                     mode='lines', line=dict(color='#34495e', width=6), showlegend=False), row=1, col=1)

    # Loads (User loads only, excluding Self-weight for visual clarity or add note)
    for l in loads:
        span_idx = l['span_index']
        x_local = l['x']
        x_global_start = cum_dist[span_idx] + x_local
        
        mag_val = l['mag'] / 1000.0 # kN
        
        if l['type'] == 'P':
            fig.add_annotation(
                x=x_global_start, y=0, 
                ax=0, ay=-60, 
                text=f"<b>{mag_val:.1f}kN</b>", 
                showarrow=True, arrowhead=2, arrowwidth=2, arrowcolor='#c0392b',
                row=1, col=1
            )
        elif l['type'] == 'U':
            x_global_end = x_global_start + l['dist']
            # Distributed arrows
            steps = np.linspace(x_global_start, x_global_end, 5)
            for xa in steps:
                fig.add_annotation(x=xa, y=0, ax=0, ay=-40, arrowhead=2, arrowwidth=1, arrowcolor='#d35400', row=1, col=1, text="")
            # Top bar
            fig.add_trace(go.Scatter(x=[x_global_start, x_global_end], y=[S_SCALE, S_SCALE], 
                                     mode='lines', line=dict(color='#d35400', width=1), showlegend=False), row=1, col=1)
            fig.add_annotation(x=(x_global_start+x_global_end)/2, y=S_SCALE*1.3, 
                               text=f"<b>w={mag_val:.1f}</b>", showarrow=False, font=dict(color='#d35400'), row=1, col=1)

    # Reactions Text
    for nid, val in reactions.items():
        x_r = node_x_map.get(nid, 0)
        r_kn = val / 1000.0
        if abs(r_kn) > 0.01:
            fig.add_annotation(
                x=x_r, y=-S_SCALE*1.5, 
                text=f"<b>R={r_kn:.2f}</b>", showarrow=False, font=dict(color='green'),
                row=1, col=1
            )

    fig.update_yaxes(visible=False, range=[-S_SCALE*2.5, S_SCALE*2.5], row=1, col=1)


    # --- ROW 2: SFD ---
    y_shear = res['shear'] / 1000.0
    fig.add_trace(go.Scatter(x=res['x'], y=y_shear, mode='lines', fill='tozeroy', 
                             line=dict(color='#2980b9', width=2), name='Shear'), row=2, col=1)
    if len(y_shear) > 0:
        idx_max = y_shear.abs().idxmax()
        fig.add_annotation(x=res.loc[idx_max, 'x'], y=y_shear[idx_max], text=f"{y_shear[idx_max]:.2f}", showarrow=False, yshift=15, row=2, col=1)


    # --- ROW 3: BMD (Tension Side) ---
    # Concept: Sagging (+M) -> Tension Bottom -> Plot DOWN
    #          Hogging (-M) -> Tension Top    -> Plot UP
    # Implementation: Plot (-Moment)
    
    y_moment = res['moment'] / 1000.0 # kNm
    y_plot = -y_moment 

    # Plot Lines
    fig.add_trace(go.Scatter(x=res['x'], y=y_plot, mode='lines', 
                             line=dict(color='black', width=1), name='Moment'), row=3, col=1)

    # Color Fills
    # 1. Sagging (+M in calculation, -Y in plot) -> Blue Area (Bottom)
    fig.add_trace(go.Scatter(x=res['x'], y=y_plot.clip(upper=0), mode='lines', fill='tozeroy',
                             line=dict(width=0), fillcolor='rgba(52, 152, 219, 0.4)', name='Sagging (+M)'), row=3, col=1)
    
    # 2. Hogging (-M in calculation, +Y in plot) -> Orange Area (Top)
    fig.add_trace(go.Scatter(x=res['x'], y=y_plot.clip(lower=0), mode='lines', fill='tozeroy',
                             line=dict(width=0), fillcolor='rgba(230, 126, 34, 0.4)', name='Hogging (-M)'), row=3, col=1)

    # Max/Min Labels
    m_max = y_moment.max() # Sagging (+value)
    m_min = y_moment.min() # Hogging (-value)
    
    # Label for Sagging (Plot Down)
    if m_max > 0.01:
        idx = y_moment.idxmax()
        fig.add_annotation(x=res.loc[idx, 'x'], y=-m_max, 
                           text=f"<b>M+ {m_max:.2f}</b><br>(Bot Steel)", 
                           showarrow=True, arrowhead=1, ax=0, ay=40, row=3, col=1)
        
    # Label for Hogging (Plot Up)
    if m_min < -0.01:
        idx = y_moment.idxmin()
        fig.add_annotation(x=res.loc[idx, 'x'], y=-m_min, 
                           text=f"<b>M- {abs(m_min):.2f}</b><br>(Top Steel)", 
                           showarrow=True, arrowhead=1, ax=0, ay=-40, row=3, col=1)


    # --- ROW 4: Deflection ---
    y_def = res['deflection']
    fig.add_trace(go.Scatter(x=res['x'], y=y_def, mode='lines', 
                             line=dict(color='#27ae60', width=2), name='Deflection'), row=4, col=1)
    
    if len(y_def) > 0:
        idx_def = y_def.abs().idxmax()
        val_def = y_def.iloc[idx_def]
        fig.add_annotation(x=res.loc[idx_def, 'x'], y=val_def, text=f"Max: {val_def:.2f}mm", showarrow=True, row=4, col=1)


    # --- Layout Settings ---
    fig.update_layout(height=1000, showlegend=False, hovermode="x unified", paper_bgcolor='white', plot_bgcolor='rgba(250,250,250,1)')
    
    # Vertical grid lines
    for x_s in cum_dist:
        fig.add_vline(x=x_s, line_width=1, line_dash="dash", line_color="gray")

    fig.update_yaxes(title_text="V (kN)", row=2, col=1)
    fig.update_yaxes(title_text="M (kNm)", row=3, col=1)
    fig.update_yaxes(title_text="Def (mm)", row=4, col=1)
    fig.update_xaxes(title_text="Distance (m)", row=4, col=1)

    return fig
