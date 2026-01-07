import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

def plot_analysis_results(res_df, spans, supports, loads, reactions):
    """
    Creates a detailed structural analysis plot (Load, SFD, BMD).
    Improved for textbook-style visualization.
    """
    # Create Subplots: 3 Rows
    fig = make_subplots(
        rows=3, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.08,
        subplot_titles=(
            "<b>1. Free Body Diagram (Load Model)</b>", 
            "<b>2. Shear Force Diagram (SFD)</b>", 
            "<b>3. Bending Moment Diagram (BMD)</b>"
        ),
        row_heights=[0.3, 0.35, 0.35]
    )

    # ==========================================
    # 1. LOAD MODEL (Textbook Style)
    # ==========================================
    total_L = sum(spans)
    cum_dist = [0] + list(np.cumsum(spans))
    
    # Beam Line (Thick Black Line)
    fig.add_trace(go.Scatter(
        x=[0, total_L], y=[0, 0], 
        mode='lines', 
        line=dict(color='black', width=5),
        hoverinfo='skip'
    ), row=1, col=1)
    
    # Supports (Triangles/Squares)
    for idx, row in supports.iterrows():
        # Pin = Triangle, Roller = Circle/Triangle, Fixed = Square
        sym = "triangle-up"
        if row['type'] == 'Fixed': sym = "square"
        elif row['type'] == 'Roller': sym = "circle"
        
        fig.add_trace(go.Scatter(
            x=[row['x']], y=[-0.05], # Slightly below beam
            mode='markers+text',
            marker=dict(symbol=sym, size=18, color='#2c3e50', line=dict(width=2, color='black')),
            text=[row['type']], textposition="bottom center",
            hoverinfo='text',
            name="Support"
        ), row=1, col=1)

    # Loads Processing
    max_load_mag = 1.0
    if loads:
        max_load_mag = max([l['mag'] for l in loads]) if len(loads) > 0 else 1.0
    
    scale_factor = 0.5 / (max_load_mag if max_load_mag > 0 else 1) # Scaling for visual height

    for l in loads:
        # Calculate absolute start X
        start_x = cum_dist[l['span_index']]
        mag_kN = l['mag'] / 1000.0
        
        if l['type'] == 'P':
            # Point Load: ARROW Style
            x_loc = start_x + l['dist']
            # Invisible point to anchor arrow
            fig.add_annotation(
                x=x_loc, y=0,
                ax=x_loc, ay=-60, # Length of arrow tail in pixels (upwards)
                xref="x1", yref="y1",
                axref="x1", ayref="y1", # Not needed if using pixel offset
                showarrow=True,
                arrowhead=2, arrowsize=1.5, arrowwidth=2, arrowcolor="#c0392b",
                text=f"<b>P={mag_kN:.2f} kN</b>",
                yshift=10
            )

        elif l['type'] == 'U':
            # UDL: Shaded Block with Top Line + Center Arrow
            x_s = start_x
            x_e = x_s + l['dist']
            h_vis = 0.3 # Fixed visual height for UDL to look clean
            
            # Shaded Area
            fig.add_trace(go.Scatter(
                x=[x_s, x_e, x_e, x_s], 
                y=[h_vis, h_vis, 0, 0], 
                fill='toself', 
                fillcolor='rgba(52, 152, 219, 0.2)', # Light Blue transparent
                mode='none',
                hoverinfo='skip'
            ), row=1, col=1)
            
            # Top Line
            fig.add_trace(go.Scatter(
                x=[x_s, x_e], y=[h_vis, h_vis],
                mode='lines', line=dict(color='#2980b9', width=2),
                hoverinfo='skip'
            ), row=1, col=1)
            
            # Central Label & Arrow
            mid_x = (x_s + x_e) / 2
            fig.add_annotation(
                x=mid_x, y=h_vis,
                ax=mid_x, ay=0, # Point down to beam
                xref="x1", yref="y1",
                axref="x1", ayref="y1",
                showarrow=True, arrowhead=2, arrowcolor="#2980b9",
                text=f"w={mag_kN:.2f} kN/m",
                yshift=10
            )

    # ==========================================
    # 2. SHEAR FORCE DIAGRAM (SFD)
    # ==========================================
    # Add Zero Line
    fig.add_hline(y=0, line_dash="solid", line_color="black", line_width=1, row=2, col=1)
    
    # Plot Data
    y_shear = res_df['shear'] / 1000.0
    fig.add_trace(go.Scatter(
        x=res_df['x'], y=y_shear, 
        mode='lines', 
        name='Shear (kN)', 
        line=dict(color='#e74c3c', width=2),
        fill='tozeroy',
        fillcolor='rgba(231, 76, 60, 0.1)'
    ), row=2, col=1)

    # Max/Min Annotations
    v_max = y_shear.max()
    v_min = y_shear.min()
    idx_max = y_shear.idxmax()
    idx_min = y_shear.idxmin()
    
    # Label Max Positive
    if abs(v_max) > 0.01:
        fig.add_annotation(x=res_df['x'][idx_max], y=v_max, text=f"{v_max:.2f}", showarrow=False, yshift=10, font=dict(color='red'), row=2, col=1)
    # Label Max Negative
    if abs(v_min) > 0.01:
        fig.add_annotation(x=res_df['x'][idx_min], y=v_min, text=f"{v_min:.2f}", showarrow=False, yshift=-10, font=dict(color='red'), row=2, col=1)

    # ==========================================
    # 3. BENDING MOMENT DIAGRAM (BMD)
    # ==========================================
    # Add Zero Line
    fig.add_hline(y=0, line_dash="solid", line_color="black", line_width=1, row=3, col=1)

    # Plot Data (Inverted Y is handled by layout, data remains actual sign)
    y_moment = res_df['moment'] / 1000.0
    
    fig.add_trace(go.Scatter(
        x=res_df['x'], y=y_moment, 
        mode='lines', 
        name='Moment (kNm)', 
        line=dict(color='#27ae60', width=2), # Green
        fill='tozeroy',
        fillcolor='rgba(39, 174, 96, 0.1)'
    ), row=3, col=1)

    # Max/Min Annotations (BMD)
    m_max = y_moment.max()
    m_min = y_moment.min()
    idx_m_max = y_moment.idxmax()
    idx_m_min = y_moment.idxmin()

    # Label Max Positive (Sagging) - Will be at bottom visually
    if abs(m_max) > 0.01:
        fig.add_annotation(x=res_df['x'][idx_m_max], y=m_max, text=f"Max+ {m_max:.2f}", showarrow=True, arrowhead=1, ay=20, font=dict(color='green'), row=3, col=1)
    # Label Max Negative (Hogging) - Will be at top visually
    if abs(m_min) > 0.01:
        fig.add_annotation(x=res_df['x'][idx_m_min], y=m_min, text=f"Max- {m_min:.2f}", showarrow=True, arrowhead=1, ay=-20, font=dict(color='green'), row=3, col=1)

    # ==========================================
    # LAYOUT & GRIDS
    # ==========================================
    
    # Add Vertical Grid Lines at Supports/Spans
    for x_pos in cum_dist:
        fig.add_vline(x=x_pos, line_width=1, line_dash="dash", line_color="gray", opacity=0.5)

    fig.update_layout(
        height=900, 
        showlegend=False, 
        template="plotly_white",
        margin=dict(t=50, b=50, l=60, r=20),
        hovermode="x unified"
    )

    # Update Axes
    # Row 1: Load (Hidden Y)
    fig.update_yaxes(visible=False, showgrid=False, row=1, col=1)
    
    # Row 2: Shear
    fig.update_yaxes(title_text="Shear (kN)", showgrid=True, gridcolor='lightgray', zeroline=False, row=2, col=1)
    
    # Row 3: Moment (INVERTED)
    fig.update_yaxes(title_text="Moment (kNm)", showgrid=True, gridcolor='lightgray', zeroline=False, autorange="reversed", row=3, col=1)

    return fig
