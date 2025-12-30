import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def draw_interactive_diagrams(df, spans, sup_df, loads, unit_force="kg", unit_len="m"):
    # --- 1. INPUT SUMMARY (Clean & Compact) ---
    st.markdown("### 🏗️ Design Loads & Configuration")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if loads:
            load_data = []
            for i, l in enumerate(loads):
                span_txt = f"Span {l['span_idx']+1}"
                if l['type'] == 'P':
                    desc, val, loc = "Point Load (P)", f"{l['mag']:,.2f} {unit_force}", f"@ {l['x']:.2f} {unit_len} ({span_txt})"
                elif l['type'] == 'U':
                    desc, val, loc = "Uniform Load (w)", f"{l['mag']:,.2f} {unit_force}/{unit_len}", f"Full {span_txt}"
                elif l['type'] == 'M':
                    desc, val, loc = "Moment (M)", f"{l['mag']:,.2f} {unit_force}-{unit_len}", f"@ {l['x']:.2f} {unit_len} ({span_txt})"
                
                load_data.append([desc, val, loc])
            
            df_loads = pd.DataFrame(load_data, columns=["Type", "Magnitude", "Location"])
            st.dataframe(df_loads, use_container_width=True, hide_index=True)
        else:
            st.info("ℹ️ No loads applied yet.")

    with col2:
        st.caption("Total Length")
        st.markdown(f"**{sum(spans):.2f} {unit_len}**")
        st.caption("Supports")
        st.markdown(f"**{len(sup_df)} Nodes**")

    if df is None or df.empty: return

    st.markdown("---")
    
    # --- 2. PREPARE PLOTTING DATA ---
    total_len = sum(spans)
    cum_spans = [0] + list(np.cumsum(spans))
    
    # Identify Critical Key Points (Nodes + Load Locations) for Vertical Grid Lines
    key_points = set()
    for x in cum_spans: key_points.add(round(x, 3))
    for l in loads:
        abs_x = cum_spans[int(l['span_idx'])] + l['x']
        key_points.add(round(abs_x, 3))
    sorted_keys = sorted(list(key_points))

    # Setup Subplots (Standard Engineering Layout)
    fig = make_subplots(
        rows=4, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.06,
        subplot_titles=(
            "<b>Structure Model (FBD)</b>", 
            f"<b>Shear Force Diagram (SFD)</b> [{unit_force}]", 
            f"<b>Bending Moment Diagram (BMD)</b> [{unit_force}-{unit_len}]", 
            f"<b>Deflection (δ)</b> [{unit_len}]"
        ),
        row_heights=[0.18, 0.28, 0.28, 0.26]
    )

    # ==========================================
    # ROW 1: FREE BODY DIAGRAM (FBD)
    # ==========================================
    # 1. Main Beam Line
    fig.add_trace(go.Scatter(x=[0, total_len], y=[0, 0], line=dict(color='black', width=4), hoverinfo='skip'), row=1, col=1)
    
    # 2. Supports & Nodes
    sup_map = {int(r['id']): r['type'] for _, r in sup_df.iterrows()}
    for i, x in enumerate(cum_spans):
        # Node Bubble
        fig.add_annotation(
            x=x, y=0.35, text=f"N{i+1}", showarrow=False,
            font=dict(size=10, color="#555"),
            bgcolor="#eee", bordercolor="#ccc", borderwidth=1, borderpad=2,
            row=1, col=1
        )
        
        # Support Symbol
        if i in sup_map:
            stype = sup_map[i]
            if stype == 'Fixed':
                fig.add_shape(type="line", x0=x, y0=-0.25, x1=x, y1=0.25, line=dict(width=5, color='#333'), row=1, col=1)
            else:
                sym = 'triangle-up' if stype == 'Pin' else 'circle'
                # Draw marker exactly at y=0
                fig.add_trace(go.Scatter(
                    x=[x], y=[0], 
                    mode='markers', 
                    marker=dict(symbol=sym, size=16, color='white', line=dict(color='#333', width=2)), 
                    name=stype, hoverinfo='name'
                ), row=1, col=1)
                # For Roller, add line below
                if stype == 'Roller':
                    fig.add_shape(type="line", x0=x-0.1, y0=-0.08, x1=x+0.1, y1=-0.08, line=dict(width=2, color='#333'), row=1, col=1)

    # 3. Loads Visualization
    for l in loads:
        x_abs = cum_spans[int(l['span_idx'])] + l['x']
        if l['type'] == 'P':
            # Arrow pointing down
            fig.add_annotation(
                x=x_abs, y=0, ax=0, ay=-50, 
                arrowhead=2, arrowwidth=2, arrowcolor='#D32F2F', 
                text=f"<b>P={l['mag']}</b>", font=dict(color='#D32F2F', size=11),
                row=1, col=1
            )
        elif l['type'] == 'U':
             xs, xe = cum_spans[int(l['span_idx'])], cum_spans[int(l['span_idx'])+1]
             # Rectangular Block
             fig.add_shape(type="rect", x0=xs, x1=xe, y0=0.05, y1=0.20, fillcolor="#D32F2F", opacity=0.15, line_width=0, row=1, col=1)
             fig.add_annotation(x=(xs+xe)/2, y=0.25, text=f"<b>w={l['mag']}</b>", showarrow=False, font=dict(color="#D32F2F", size=11), row=1, col=1)
        elif l['type'] == 'M':
            fig.add_annotation(x=x_abs, y=0, text=f"<b>M={l['mag']}</b>", showarrow=True, arrowhead=1, ax=0, ay=-35, arrowcolor='#7B1FA2', font=dict(color='#7B1FA2', size=11), row=1, col=1)

    # ==========================================
    # HELPER: SMART LABELS (No Arrows, Clean Boxes)
    # ==========================================
    def add_smart_labels(x_series, y_series, row_idx, color_hex):
        y_arr = np.array(y_series)
        x_arr = np.array(x_series)
        if len(y_arr) == 0: return

        # Find Indices
        max_idx = np.argmax(y_arr)
        min_idx = np.argmin(y_arr)
        
        # Helper to create label
        def create_tag(val, x_pos, is_top):
            return dict(
                x=x_pos, y=val,
                text=f"<b>{val:.2f}</b><br><span style='font-size:9px'>@ {x_pos:.2f}m</span>",
                showarrow=False,
                yshift=15 if is_top else -15,
                font=dict(color=color_hex, size=11),
                bgcolor="rgba(255,255,255,0.9)", # High opacity background to read over grid
                bordercolor=color_hex, borderwidth=1, borderpad=3,
                align="center"
            )

        # 1. Max Label
        if abs(y_arr[max_idx]) > 1e-4:
            fig.add_annotation(row=row_idx, col=1, **create_tag(y_arr[max_idx], x_arr[max_idx], True))

        # 2. Min Label (if distinct)
        if abs(y_arr[min_idx]) > 1e-4 and abs(x_arr[max_idx] - x_arr[min_idx]) > 0.05:
            fig.add_annotation(row=row_idx, col=1, **create_tag(y_arr[min_idx], x_arr[min_idx], False))

    # ==========================================
    # ROW 2: SHEAR (SFD) - Engineer Orange
    # ==========================================
    c_shear = '#E67E22' # Pumpkin Orange
    fig.add_trace(go.Scatter(x=df['x'], y=df['shear'], fill='tozeroy', mode='lines', line=dict(color=c_shear, width=2), name="Shear"), row=2, col=1)
    add_smart_labels(df['x'], df['shear'], 2, c_shear)

    # ==========================================
    # ROW 3: MOMENT (BMD) - Engineer Blue
    # ==========================================
    c_moment = '#2980B9' # Strong Blue
    fig.add_trace(go.Scatter(x=df['x'], y=df['moment'], fill='tozeroy', mode='lines', line=dict(color=c_moment, width=2), name="Moment"), row=3, col=1)
    add_smart_labels(df['x'], df['moment'], 3, c_moment)

    # ==========================================
    # ROW 4: DEFLECTION - Engineer Green
    # ==========================================
    c_defl = '#27AE60' # Nephritis Green
    fig.add_trace(go.Scatter(x=df['x'], y=df['deflection'], fill='tozeroy', mode='lines', line=dict(color=c_defl, width=2), name="Deflection"), row=4, col=1)
    
    # Only label Absolute Max Deflection
    abs_d_idx = df['deflection'].abs().idxmax()
    d_val = df.loc[abs_d_idx, 'deflection']
    d_x = df.loc[abs_d_idx, 'x']
    
    if abs(d_val) > 1e-9:
        fig.add_annotation(
            x=d_x, y=d_val,
            text=f"<b>Max: {d_val:.2e}</b><br><span style='font-size:9px'>@ {d_x:.2f}m</span>",
            showarrow=False, yshift=20 if d_val > 0 else -20,
            font=dict(color=c_defl, size=11),
            bgcolor="rgba(255,255,255,0.9)", bordercolor=c_defl, borderwidth=1, borderpad=3,
            row=4, col=1
        )

    # ==========================================
    # GLOBAL STYLING & GRID
    # ==========================================
    # Draw Vertical Drop Lines at ALL Key Points
    for kp in sorted_keys:
        fig.add_vline(x=kp, line_width=1, line_dash="dot", line_color="#bbb")

    fig.update_layout(
        height=1100, # Tall enough for clarity
        showlegend=False,
        template="plotly_white", # Best for engineering (subtle grids)
        margin=dict(l=60, r=40, t=40, b=40),
        hovermode="x unified"
    )
    
    # Hide Y-axis for FBD only
    fig.update_yaxes(visible=False, row=1, col=1)
    
    # Ensure zero lines are distinct
    fig.update_yaxes(zeroline=True, zerolinewidth=1.5, zerolinecolor='#333')

    st.plotly_chart(fig, use_container_width=True)

    # --- 3. RESULT TABLE (Reactions) ---
    st.subheader("📍 Support Reactions")
    
    reac_data = []
    for i in range(len(spans)+1):
        ry = reac[2*i]
        mz = reac[2*i+1]
        
        # Only show nodes with supports (checking Ry or Mz != 0 is a proxy, or use support map)
        if abs(ry) > 1e-5 or abs(mz) > 1e-5:
            reac_data.append({
                "Node": f"{i+1}",
                "Vertical Reaction (Ry)": f"{ry:.2f} {unit_force}",
                "Moment Reaction (Mz)": f"{mz:.2f} {unit_force}-{unit_len}"
            })
            
    if reac_data:
        st.table(pd.DataFrame(reac_data).set_index("Node"))
    else:
        st.write("No significant reactions.")
