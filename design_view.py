import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def draw_interactive_diagrams(df, spans, sup_df, loads, unit_force, unit_len):
    # --- 1. LOAD CALCULATION LIST ---
    st.markdown("### 📋 Design Loads")
    
    if loads:
        load_data = []
        for i, l in enumerate(loads):
            if l['type'] == 'P':
                desc, val, pos = "Point Load", f"{l['mag']} {unit_force}", f"@ {l['x']} {unit_len} (Span {l['span_idx']+1})"
            elif l['type'] == 'U':
                desc, val, pos = "Uniform Load", f"{l['mag']} {unit_force}/{unit_len}", f"Span {l['span_idx']+1}"
            elif l['type'] == 'M':
                desc, val, pos = "Moment", f"{l['mag']} {unit_force}-{unit_len}", f"@ {l['x']} {unit_len} (Span {l['span_idx']+1})"
            
            load_data.append([i+1, desc, val, pos])
            
        st.table(pd.DataFrame(load_data, columns=["No.", "Type", "Magnitude", "Location"]))
    else:
        st.info("No loads applied.")

    if df is None or df.empty: return

    st.markdown("---")
    st.markdown("### 📊 Structural Analysis Diagrams")

    # --- 2. PREPARE DATA ---
    total_len = sum(spans)
    cum_spans = [0] + list(np.cumsum(spans))
    
    # Key Points for Drop Lines
    key_points = set()
    for x in cum_spans: key_points.add(round(x, 3))
    for l in loads:
        abs_x = cum_spans[int(l['span_idx'])] + l['x']
        key_points.add(round(abs_x, 3))
    sorted_keys = sorted(list(key_points))

    # Create Subplots
    fig = make_subplots(
        rows=4, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.08,
        subplot_titles=("Structure Model", f"Shear Force ({unit_force})", f"Bending Moment ({unit_force}-{unit_len})", f"Deflection ({unit_len})"),
        row_heights=[0.15, 0.28, 0.28, 0.29]
    )

    # ==========================================
    # ROW 1: STRUCTURE (FBD)
    # ==========================================
    # Beam Line
    fig.add_trace(go.Scatter(x=[0, total_len], y=[0, 0], line=dict(color='black', width=3), hoverinfo='skip'), row=1, col=1)
    
    # Supports & Nodes
    sup_map = {int(r['id']): r['type'] for _, r in sup_df.iterrows()}
    for i, x in enumerate(cum_spans):
        # Node Labels
        fig.add_annotation(
            x=x, y=0.35, text=f"Node {i+1}", showarrow=False,
            font=dict(size=10, color="gray"), row=1, col=1
        )
        
        if i in sup_map:
            stype = sup_map[i]
            if stype == 'Fixed':
                fig.add_shape(type="line", x0=x, y0=-0.25, x1=x, y1=0.25, line=dict(width=4, color='black'), row=1, col=1)
            else:
                sym = 'triangle-up' if stype == 'Pin' else 'circle'
                # Plot EXACTLY at y=0, symbol handles visual offset
                fig.add_trace(go.Scatter(
                    x=[x], y=[0], 
                    mode='markers', 
                    marker=dict(symbol=sym, size=14, color='white', line=dict(color='black', width=2)), 
                    name=stype, hoverinfo='name'
                ), row=1, col=1)

    # Loads
    for l in loads:
        x_abs = cum_spans[int(l['span_idx'])] + l['x']
        if l['type'] == 'P':
            fig.add_annotation(x=x_abs, y=0, ax=0, ay=-40, arrowhead=2, arrowwidth=2, arrowcolor='red', text=f"P={l['mag']}", font=dict(color='red', size=10), row=1, col=1)
        elif l['type'] == 'U':
             xs, xe = cum_spans[int(l['span_idx'])], cum_spans[int(l['span_idx'])+1]
             fig.add_shape(type="rect", x0=xs, x1=xe, y0=0.05, y1=0.15, fillcolor="red", opacity=0.15, line_width=0, row=1, col=1)
             fig.add_annotation(x=(xs+xe)/2, y=0.2, text=f"w={l['mag']}", showarrow=False, font=dict(color="red", size=10), row=1, col=1)
        elif l['type'] == 'M':
            fig.add_annotation(x=x_abs, y=0, text=f"M={l['mag']}", showarrow=True, arrowhead=1, ax=0, ay=-30, arrowcolor='purple', font=dict(size=10), row=1, col=1)

    # ==========================================
    # HELPER: Label with Max/Min Text + Location
    # ==========================================
    def add_labels_with_text(x_data, y_data, row_idx, color_code):
        y_arr = np.array(y_data)
        if len(y_arr) == 0: return

        # 1. Global Max
        max_idx = np.argmax(y_arr)
        max_val = y_arr[max_idx]
        max_x = x_data[max_idx]
        
        # 2. Global Min
        min_idx = np.argmin(y_arr)
        min_val = y_arr[min_idx]
        min_x = x_data[min_idx]

        # Common Style
        style = dict(
            showarrow=False, # NO ARROWS as requested
            font=dict(color=color_code, size=11),
            bgcolor="rgba(255,255,255,0.85)", # Background to read over lines
            bordercolor=color_code, borderwidth=1, borderpad=3
        )

        # Plot Max
        if abs(max_val) > 1e-4:
            fig.add_annotation(
                x=max_x, y=max_val,
                text=f"<b>Max:</b> {max_val:.2f}<br>@ {max_x:.2f}m", # Rich Text
                yshift=20, # Shift UP
                row=row_idx, col=1, **style
            )

        # Plot Min (if different)
        if abs(min_val) > 1e-4 and abs(max_x - min_x) > 0.05:
            fig.add_annotation(
                x=min_x, y=min_val,
                text=f"<b>Min:</b> {min_val:.2f}<br>@ {min_x:.2f}m", # Rich Text
                yshift=-20, # Shift DOWN
                row=row_idx, col=1, **style
            )

    # ==========================================
    # PLOT DATA
    # ==========================================
    # Shear
    c_shear = '#E67E22'
    fig.add_trace(go.Scatter(x=df['x'], y=df['shear'], fill='tozeroy', line=dict(color=c_shear, width=2), name="Shear"), row=2, col=1)
    add_labels_with_text(df['x'], df['shear'], 2, c_shear)

    # Moment
    c_moment = '#2980B9'
    fig.add_trace(go.Scatter(x=df['x'], y=df['moment'], fill='tozeroy', line=dict(color=c_moment, width=2), name="Moment"), row=3, col=1)
    add_labels_with_text(df['x'], df['moment'], 3, c_moment)

    # Deflection
    c_defl = '#27AE60'
    fig.add_trace(go.Scatter(x=df['x'], y=df['deflection'], fill='tozeroy', line=dict(color=c_defl, width=2), name="Deflection"), row=4, col=1)
    
    # Deflection Abs Max
    abs_d_idx = df['deflection'].abs().idxmax()
    d_val = df.loc[abs_d_idx, 'deflection']
    d_x = df.loc[abs_d_idx, 'x']
    if abs(d_val) > 1e-9:
        fig.add_annotation(
            x=d_x, y=d_val,
            text=f"<b>Max:</b> {d_val:.2e}<br>@ {d_x:.2f}m",
            showarrow=False,
            yshift=25 if d_val > 0 else -25,
            font=dict(color=c_defl, size=11),
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor=c_defl, borderwidth=1, borderpad=3,
            row=4, col=1
        )

    # ==========================================
    # STYLING
    # ==========================================
    # Vertical Drop Lines (ALL Key points)
    for kp in sorted_keys:
        fig.add_vline(x=kp, line_width=1, line_dash="dash", line_color="gray", opacity=0.5)

    fig.update_layout(
        height=1000, 
        showlegend=False,
        template="simple_white",
        margin=dict(l=50, r=20, t=40, b=40),
        font=dict(family="Roboto, Arial", size=12)
    )
    
    # Grids
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#EEE', row=4, col=1)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#EEE')
    fig.update_yaxes(visible=False, row=1, col=1) # Hide FBD Y

    st.plotly_chart(fig, use_container_width=True)

def render_result_tables(df, reac, spans):
    st.markdown("### 📍 Support Reactions")
    reac_data = []
    for i in range(len(spans)+1):
        ry = reac[2*i]
        mz = reac[2*i+1]
        reac_data.append({
            "Node": i+1, 
            "Vertical (Ry)": f"{ry:.2f}", 
            "Moment (Mz)": f"{mz:.2f}"
        })
    st.table(pd.DataFrame(reac_data))
