import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

def plot_analysis_results(res, spans, sup_df, loads, reactions):
    """
    Plots diagrams with:
    1. Proportional Supports (Scaled to beam length)
    2. Moment on Tension Side (Inverted logic for RC design)
    3. Correct Load positioning
    """
    cum_dist = [0] + list(np.cumsum(spans))
    total_len = cum_dist[-1]
    
    # Scale Factor for Visuals (Supports, Arrows)
    # Makes supports visible regardless of beam length
    S_SCALE = max(total_len * 0.04, 0.4) 

    fig = make_subplots(
        rows=4, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.08,
        subplot_titles=(
            "1. Loading & Supports (FBD)", 
            "2. Shear Force (SFD)", 
            "3. Bending Moment (BMD - Plotted on Tension Side)", 
            "4. Deflection (mm)"
        ),
        row_heights=[0.3, 0.25, 0.25, 0.2]
    )

    # --- ROW 1: FBD ---
    # Beam
    fig.add_trace(go.Scatter(x=[0, total_len], y=[0, 0], mode='lines', 
                             line=dict(color='black', width=5), hoverinfo='skip'), row=1, col=1)

    # Supports (Proportional Geometries)
    node_x_map = {i: x for i, x in enumerate(cum_dist)}
    sup_list = sup_df.to_dict('records')
    
    for s in sup_list:
        nid = s.get('id', s.get('node_id'))
        x_pos = node_x_map[nid]
        stype = s['type']
        
        # Dimensions based on S_SCALE
        w = S_SCALE * 0.6  # Half width
        h = S_SCALE * 0.8  # Height
        
        if stype == 'Pin':
            # Triangle
            fig.add_trace(go.Scatter(
                x=[x_pos-w, x_pos, x_pos+w, x_pos-w], 
                y=[-h, 0, -h, -h], 
                fill='toself', line=dict(color='#2c3e50', width=2), fillcolor='white',
                mode='lines', showlegend=False, hoverinfo='skip'
            ), row=1, col=1)
            
        elif stype == 'Roller':
            # Circle
            r = w
            # Use SVG path for perfect circle approximation or basic shape
            fig.add_shape(type="circle", x0=x_pos-r, y0=-2*r, x1=x_pos+r, y1=0, 
                          line=dict(color="#2c3e50", width=2), fillcolor="white", row=1, col=1)
            # Base line
            fig.add_trace(go.Scatter(x=[x_pos-w*1.5, x_pos+w*1.5], y=[-2*r, -2*r], 
                                     mode='lines', line=dict(color='#2c3e50', width=2), showlegend=False), row=1, col=1)

        elif stype == 'Fixed':
            # Vertical Block
            fig.add_trace(go.Scatter(x=[x_pos, x_pos], y=[-h, h], 
                                     mode='lines', line=dict(color='#2c3e50', width=6), showlegend=False), row=1, col=1)
            # Hatches
            direction = -1 if x_pos == 0 else 1
            for k in np.linspace(-h, h, 6):
                fig.add_trace(go.Scatter(x=[x_pos, x_pos + direction*w], y=[k, k-w/2], 
                                         mode='lines', line=dict(color='#2c3e50', width=1), showlegend=False), row=1, col=1)

    # Loads (Arrows)
    # We display USER loads only (Self-weight is implicit)
    for l in loads:
        span_idx = l['span_index']
        x_local = l['x']
        x_global_start = cum_dist[span_idx] + x_local
        
        mag_val = l['mag'] / 1000.0
        
        arrow_len = S_SCALE * 1.5
        
        if l['type'] == 'P':
            fig.add_annotation(
                x=x_global_start, y=0, 
                ax=0, ay=-60, # Pixel offset for arrow tail
                text=f"<b>{mag_val:.1f}kN</b>", 
                showarrow=True, arrowhead=2, arrowwidth=2, arrowcolor='#e74c3c',
                row=1, col=1
            )
        elif l['type'] == 'U':
            x_global_end = x_global_start + l['dist']
            # Draw multiple small arrows
            steps = np.linspace(x_global_start, x_global_end, 5)
            for xa in steps:
                fig.add_annotation(x=xa, y=0, ax=0, ay=-40, arrowhead=2, arrowwidth=1, arrowcolor='#d35400', row=1, col=1, text="")
            # Bar on top
            fig.add_trace(go.Scatter(x=[x_global_start, x_global_end], y=[S_SCALE, S_SCALE], 
                                     mode='lines', line=dict(color='#d35400', width=1), showlegend=False), row=1, col=1)
            # Label
            fig.add_annotation(x=(x_global_start+x_global_end)/2, y=S_SCALE*1.2, 
                               text=f"<b>w={mag_val:.1f}</b>", showarrow=False, font=dict(color='#d35400'), row=1, col=1)

    # Reactions Text
    for nid, val in reactions.items():
        x_r = node_x_map.get(nid, 0)
        r_kn = val / 1000.0
        if abs(r_kn) > 0.01:
            fig.add_annotation(
                x=x_r, y=-S_SCALE*1.2, 
                text=f"<b>R={r_kn:.2f}</b>", showarrow=False, font=dict(color='green'),
                row=1, col=1
            )
            
    fig.update_yaxes(visible=False, range=[-S_SCALE*2, S_SCALE*2], row=1, col=1)


    # --- ROW 2: SHEAR (SFD) ---
    y_shear = res['shear'] / 1000.0
    fig.add_trace(go.Scatter(x=res['x'], y=y_shear, mode='lines', fill='tozeroy', 
                             line=dict(color='#2980b9', width=2), name='Shear'), row=2, col=1)
    # Max labels
    if len(y_shear) > 0:
        idx_max = y_shear.abs().idxmax()
        fig.add_annotation(x=res.loc[idx_max, 'x'], y=y_shear[idx_max], text=f"{y_shear[idx_max]:.1f}", showarrow=False, yshift=10, row=2, col=1)


    # --- ROW 3: MOMENT (BMD - Tension Side) ---
    # Concept: +Moment (Sagging) -> Tension Bottom -> Plot DOWN
    #          -Moment (Hogging) -> Tension Top    -> Plot UP
    # Implementation: Multiply by -1 for Plotting
    
    y_moment = res['moment'] / 1000.0
    y_plot = -y_moment # Invert for tension-side plotting

    # Separate for coloring
    # If y_moment is Positive (Sagging), y_plot is Negative (Down). Color Blue (Bot Steel).
    # If y_moment is Negative (Hogging), y_plot is Positive (Up). Color Orange (Top Steel).
    
    # We use fill='tozeroy' but need to be careful with crossing points.
    # Plotly handles this well usually.
    
    fig.add_trace(go.Scatter(x=res['x'], y=y_plot, mode='lines', 
                             line=dict(color='black', width=1), name='Moment'), row=3, col=1)
    
    # Add fill areas manually to ensure correct colors
    # Area Above 0 (Mathematical Negative Moment -> Hogging -> Top Steel)
    fig.add_trace(go.Scatter(x=res['x'], y=y_plot.clip(lower=0), mode='lines', fill='tozeroy',
                             line=dict(width=0), fillcolor='rgba(230, 126, 34, 0.4)', name='Top Steel (Hogging)'), row=3, col=1)
    
    # Area Below 0 (Mathematical Positive Moment -> Sagging -> Bot Steel)
    fig.add_trace(go.Scatter(x=res['x'], y=y_plot.clip(upper=0), mode='lines', fill='tozeroy',
                             line=dict(width=0), fillcolor='rgba(52, 152, 219, 0.4)', name='Bot Steel (Sagging)'), row=3, col=1)

    # Labels
    m_max = y_moment.max()
    m_min = y_moment.min()
    
    # Label for Sagging (+M, plotted Down)
    if m_max > 0.1:
        idx = y_moment.idxmax()
        fig.add_annotation(x=res.loc[idx, 'x'], y=-m_max, text=f"<b>M+ {m_max:.1f}</b><br>(Bot Steel)", 
                           showarrow=True, arrowhead=1, ax=0, ay=40, row=3, col=1)
        
    # Label for Hogging (-M, plotted Up)
    if m_min < -0.1:
        idx = y_moment.idxmin()
        fig.add_annotation(x=res.loc[idx, 'x'], y=-m_min, text=f"<b>M- {abs(m_min):.1f}</b><br>(Top Steel)", 
                           showarrow=True, arrowhead=1, ax=0, ay=-40, row=3, col=1)

    # --- ROW 4: DEFLECTION ---
    y_def = res['deflection']
    fig.add_trace(go.Scatter(x=res['x'], y=y_def, mode='lines', 
                             line=dict(color='#27ae60', width=2), name='Deflection'), row=4, col=1)
    
    if len(y_def) > 0:
        idx_def = y_def.abs().idxmax()
        fig.add_annotation(x=res.loc[idx_def, 'x'], y=y_def[idx_def], text=f"Max: {y_def[idx_def]:.2f}mm", row=4, col=1)


    # --- LAYOUT ---
    fig.update_layout(
        height=1000, 
        showlegend=False,
        hovermode="x unified",
        paper_bgcolor='white',
        plot_bgcolor='rgba(240,240,240,0.3)'
    )
    
    # Grid lines at supports
    for x_s in cum_dist:
        fig.add_vline(x=x_s, line_width=1, line_dash="dash", line_color="gray")

    fig.update_yaxes(title_text="FBD", row=1, col=1)
    fig.update_yaxes(title_text="Shear (kN)", row=2, col=1)
    fig.update_yaxes(title_text="Moment (kNm)", row=3, col=1, autorange="reversed") # Reverse Axis for Civil convention? No, we inverted data.
    # Actually, since we inverted data (-M up, +M down), we keep standard axis so +Y is up.
    # Wait, we put -M (Hogging) as +Y. 
    # Let's just Label Y axis clearly.
    fig.update_yaxes(title_text="Moment (Tension Side)", row=3, col=1)
    fig.update_yaxes(title_text="Deflection (mm)", row=4, col=1)
    fig.update_xaxes(title_text="Length (m)", row=4, col=1)

    return fig
