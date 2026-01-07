import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd

def plot_analysis_results(res_df, spans, supports_df, load_list_raw):
    """
    Create Interactive Diagrams (Structure + SFD + BMD + Deflection)
    Structure model loads are visually scaled proportionally to their magnitude.
    """
    if res_df.empty:
        return go.Figure()

    # --- 1. Data Preparation ---
    cum_dist = [0] + list(np.cumsum(spans))
    total_len = cum_dist[-1]

    # Clean and Prepare Loads Data
    clean_loads = []
    max_load_mag = 0.1 # Default small value to avoid div by zero
    if load_list_raw:
        for l in load_list_raw:
            try:
                span_idx = int(l.get('span_index', 0))
                local_x = float(l.get('x', 0))
                # Calculate global X position
                abs_x = cum_dist[span_idx] + local_x
                
                mag = float(l['mag']) / 1000.0 # Convert N to kN for visualization
                abs_mag = abs(mag)
                if abs_mag > max_load_mag: max_load_mag = abs_mag
                
                clean_loads.append({
                    'mag_kn': mag,
                    'abs_mag': abs_mag,
                    'global_x': abs_x,
                    'type': l.get('type','P'),
                    'case': l.get('case','DL'),
                    'dist': float(l.get('dist',0))
                })
            except: continue

    # Define visual scaling factor (Max height of load visualization = 0.6 units)
    VISUAL_HEIGHT_LIMIT = 0.6
    scale_factor = VISUAL_HEIGHT_LIMIT / max_load_mag if max_load_mag > 0 else 1

    # --- 2. Create Subplots (4 Rows) ---
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Structure Model & Loads", "Shear Force (SFD)", "Bending Moment (BMD)", "Deflection"),
        row_heights=[0.25, 0.25, 0.25, 0.25] # Equal height for balance
    )

    # === ROW 1: Structure Model & Proportional Loads ===
    
    # A. Draw Beam Line
    fig.add_trace(go.Scatter(
        x=[0, total_len], y=[0, 0], 
        mode='lines', line=dict(color='black', width=4), 
        hoverinfo='skip'
    ), row=1, col=1)
    
    # B. Draw Supports
    if not supports_df.empty:
        for _, sup in supports_df.iterrows():
            x_pos = cum_dist[int(sup['id'])]
            stype = sup['type']
            if stype == 'None': continue
            
            # Symbol Selection
            sym = 'triangle-up'
            col = '#7F8C8D' # Gray
            if stype == 'Pin': sym, col = 'triangle-up', '#2C3E50'
            elif stype == 'Roller': sym, col = 'circle', '#27AE60'
            elif stype == 'Fixed': sym, col = 'square', '#C0392B'
            
            fig.add_trace(go.Scatter(
                x=[x_pos], y=[-0.15], # Slightly below beam
                mode='markers+text',
                marker=dict(symbol=sym, size=18, color=col, line=dict(width=2, color='black')),
                text=[stype], textposition="bottom center",
                hoverinfo='text'
            ), row=1, col=1)

    # C. Draw Proportional Loads
    for l in clean_loads:
        # Color: Dead Load = Gray, Live Load = Red
        color = "#C0392B" if l['case'] == 'LL' else "#5D6D7E"
        scaled_h = l['abs_mag'] * scale_factor
        
        if l['type'] == 'P': # Point Load (Arrow Line)
            # Draw line arrow stem
            fig.add_trace(go.Scatter(
                x=[l['global_x'], l['global_x']], y=[scaled_h, 0],
                mode='lines', line=dict(color=color, width=3),
                hoverinfo='skip'
            ), row=1, col=1)
            # Draw Arrowhead (marker at bottom)
            fig.add_trace(go.Scatter(
                x=[l['global_x']], y=[0],
                mode='markers', marker=dict(symbol='arrow-down', size=14, color=color),
                hoverinfo='skip'
            ), row=1, col=1)
            # Text Label
            fig.add_trace(go.Scatter(
                x=[l['global_x']], y=[scaled_h + 0.05],
                mode='text', text=[f"{l['mag_kn']:.1f} kN"],
                textposition="top center", textfont=dict(color=color)
            ), row=1, col=1)

        elif l['type'] == 'U': # Uniform Load (Filled Area)
            x_start = l['global_x']
            x_end = x_start + l['dist']
            # Use Scatter with fill to create the load block
            fig.add_trace(go.Scatter(
                x=[x_start, x_start, x_end, x_end], 
                y=[0, scaled_h, scaled_h, 0],
                mode='lines', fill='toself', 
                line=dict(color=color, width=0), fillcolor=color, opacity=0.4,
                hoverinfo='skip'
            ), row=1, col=1)
            # Text Label (Center)
            fig.add_trace(go.Scatter(
                x=[(x_start+x_end)/2], y=[scaled_h + 0.05],
                mode='text', text=[f"{l['mag_kn']:.1f} kN/m"],
                textposition="top center", textfont=dict(color=color)
            ), row=1, col=1)

        elif l['type'] == 'M': # Moment (Curved Arrow symbol)
            # Use a marker to represent moment
            symbol = "arrow-up-down" # Placeholder, Plotly doesn't have good curved arrow marker yet
            fig.add_trace(go.Scatter(
                x=[l['global_x']], y=[0.15],
                mode='markers+text', 
                marker=dict(symbol=symbol, size=20, color=color),
                text=[f"M={l['mag_kn']:.1f}"], textposition="top center",
                hoverinfo='skip'
            ), row=1, col=1)

    # === ROWS 2, 3, 4: Analysis Diagrams (Same as before but adapted) ===
    
    # SFD (Green)
    fig.add_trace(go.Scatter(
        x=res_df['x'], y=res_df['shear']/1000,
        fill='tozeroy', line=dict(color='#27AE60', width=2),
        name="Shear (kN)", hovertemplate="x: %{x:.2f}m<br>V: %{y:.2f} kN"
    ), row=2, col=1)
    
    # BMD (Red - Inverted)
    fig.add_trace(go.Scatter(
        x=res_df['x'], y=res_df['moment']/1000,
        fill='tozeroy', line=dict(color='#C0392B', width=2),
        name="Moment (kNm)", hovertemplate="x: %{x:.2f}m<br>M: %{y:.2f} kNm"
    ), row=3, col=1)
    fig.update_yaxes(autorange="reversed", row=3, col=1) # Invert Y
    
    # Deflection (Blue)
    fig.add_trace(go.Scatter(
        x=res_df['x'], y=res_df['deflection']*1000,
        line=dict(color='#2980B9', width=2),
        name="Deflection (mm)", hovertemplate="x: %{x:.2f}m<br>δ: %{y:.2f} mm"
    ), row=4, col=1)

    # --- Final Layout Adjustments ---
    # Add Vertical Grid Lines at supports for all rows
    for x_sup in cum_dist:
        for r in range(1, 5):
            fig.add_vline(x=x_sup, line_width=1, line_dash="dash", line_color="gray", opacity=0.5, layer="below", row=r, col=1)

    fig.update_layout(
        height=900, # Taller figure for 4 rows
        showlegend=False,
        hovermode="x unified",
        margin=dict(l=60, r=20, t=60, b=40),
        plot_bgcolor="white"
    )
    
    # Y-Axis Setup
    # Row 1: Structure (Fixed range to accommodate visual loads/supports)
    fig.update_yaxes(range=[-0.5, VISUAL_HEIGHT_LIMIT + 0.3], showgrid=False, zeroline=False, showticklabels=False, row=1, col=1)
    
    # Axis Labels
    fig.update_yaxes(title_text="V (kN)", showgrid=True, gridcolor='#eee', row=2, col=1)
    fig.update_yaxes(title_text="M (kNm)", showgrid=True, gridcolor='#eee', row=3, col=1)
    fig.update_yaxes(title_text="δ (mm)", showgrid=True, gridcolor='#eee', row=4, col=1)
    fig.update_xaxes(title_text="Length (m)", showgrid=True, gridcolor='#eee', row=4, col=1)
    
    return fig
