import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def draw_interactive_diagrams(df, spans, sup_df, loads, unit_force, unit_len):
    # --- 1. LOAD COMBINATION TABLE ---
    st.markdown("### 📋 Load Combination & Report")
    
    if loads:
        load_data = []
        for i, l in enumerate(loads):
            if l['type'] == 'P':
                desc, val = "Point Load", f"{l['mag']} {unit_force}"
                pos = f"@ x = {l['x']} {unit_len} (Span {l['span_idx']+1})"
            elif l['type'] == 'U':
                desc, val = "Uniform Load", f"{l['mag']} {unit_force}/{unit_len}"
                pos = f"Full Span {l['span_idx']+1}"
            elif l['type'] == 'M':
                desc, val = "Moment Load", f"{l['mag']} {unit_force}-{unit_len}"
                pos = f"@ x = {l['x']} {unit_len} (Span {l['span_idx']+1})"
            
            load_data.append([i+1, desc, val, pos])
            
        df_list = pd.DataFrame(load_data, columns=["No.", "Type", "Magnitude", "Position"])
        st.table(df_list)
    else:
        st.info("No loads applied yet.")

    if df is None or df.empty: return

    # --- 2. PLOTTING DIAGRAMS ---
    st.markdown("### 📊 Diagrams")
    total_len = sum(spans)
    cum_spans = [0] + list(np.cumsum(spans))
    
    # คำนวณจุดสำคัญ (Supports + Loads) เพื่อวาดเส้นแนวดิ่ง (Drop Lines)
    key_points = set(cum_spans)
    for l in loads:
        key_points.add(cum_spans[int(l['span_idx'])] + l['x'])
    sorted_keys = sorted(list(key_points))

    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=("Structure (FBD)", "Shear Force (V)", "Bending Moment (M)", "Deflection (δ)"),
                        row_heights=[0.15, 0.25, 0.3, 0.3])

    # --- ROW 1: Structure (FBD) ---
    fig.add_trace(go.Scatter(x=[0, total_len], y=[0, 0], line=dict(color='black', width=4), hoverinfo='skip'), row=1, col=1)
    
    # Plot Supports
    sup_map = {int(r['id']): r['type'] for _, r in sup_df.iterrows()}
    for i, x in enumerate(cum_spans):
        if i in sup_map:
            stype = sup_map[i]
            sym = 'triangle-up' if stype == 'Pin' else ('circle' if stype == 'Roller' else 'square')
            fig.add_trace(go.Scatter(x=[x], y=[-0.1], mode='markers', marker=dict(symbol=sym, size=15, color='white', line=dict(color='black', width=2)), name=stype, hovertemplate=f"{stype} Support"), row=1, col=1)
            if stype == 'Fixed':
                fig.add_shape(type="line", x0=x, y0=-0.2, x1=x, y1=0.2, line=dict(width=5, color='black'), row=1, col=1)

    # Plot Loads
    for l in loads:
        x_abs = cum_spans[int(l['span_idx'])] + l['x']
        if l['type'] == 'P':
            fig.add_annotation(x=x_abs, y=0, ax=0, ay=-40, arrowhead=2, text=f"P={l['mag']}", row=1, col=1)
        elif l['type'] == 'M':
            fig.add_annotation(x=x_abs, y=0, text=f"↺ M={l['mag']}", showarrow=True, arrowhead=1, ax=0, ay=-30, arrowwidth=2, arrowcolor='purple', row=1, col=1)
            fig.add_trace(go.Scatter(x=[x_abs], y=[0], mode='markers', marker=dict(symbol='star', size=10, color='purple'), showlegend=False, hoverinfo='skip'), row=1, col=1)
        elif l['type'] == 'U':
             xs = cum_spans[int(l['span_idx'])]
             xe = cum_spans[int(l['span_idx'])+1]
             fig.add_shape(type="rect", x0=xs, x1=xe, y0=0.05, y1=0.15, fillcolor="red", opacity=0.2, line_width=0, row=1, col=1)
             fig.add_annotation(x=(xs+xe)/2, y=0.2, text=f"w={l['mag']}", showarrow=False, row=1, col=1)

    # --- ROW 2: Shear Force (V) ---
    fig.add_trace(go.Scatter(x=df['x'], y=df['shear'], fill='tozeroy', line=dict(color='#D97706'), fillcolor='rgba(217, 119, 6, 0.1)', name="Shear"), row=2, col=1)
    # Max/Min Labels for Shear
    v_max = df['shear'].max()
    v_min = df['shear'].min()
    v_max_x = df.loc[df['shear'].idxmax()]['x']
    v_min_x = df.loc[df['shear'].idxmin()]['x']
    
    if abs(v_max) > 1e-3:
        fig.add_annotation(x=v_max_x, y=v_max, text=f"{v_max:.2f}", showarrow=False, yshift=10, font=dict(color='red', size=10), row=2, col=1)
    if abs(v_min) > 1e-3:
        fig.add_annotation(x=v_min_x, y=v_min, text=f"{v_min:.2f}", showarrow=False, yshift=-10, font=dict(color='red', size=10), row=2, col=1)

    # --- ROW 3: Bending Moment (M) ---
    fig.add_trace(go.Scatter(x=df['x'], y=df['moment'], fill='tozeroy', line=dict(color='#2563EB'), fillcolor='rgba(37, 99, 235, 0.1)', name="Moment"), row=3, col=1)
    # Max/Min Labels for Moment
    m_max = df['moment'].max()
    m_min = df['moment'].min()
    m_max_x = df.loc[df['moment'].idxmax()]['x']
    m_min_x = df.loc[df['moment'].idxmin()]['x']
    
    if abs(m_max) > 1e-3:
        fig.add_annotation(x=m_max_x, y=m_max, text=f"{m_max:.2f}", showarrow=False, yshift=10, font=dict(color='red', size=10), row=3, col=1)
    if abs(m_min) > 1e-3:
        fig.add_annotation(x=m_min_x, y=m_min, text=f"{m_min:.2f}", showarrow=False, yshift=-10, font=dict(color='red', size=10), row=3, col=1)

    # --- ROW 4: Deflection ---
    fig.add_trace(go.Scatter(x=df['x'], y=df['deflection'], fill='tozeroy', line=dict(color='#10B981'), fillcolor='rgba(16, 185, 129, 0.1)', name="Deflection"), row=4, col=1)
    # Max Abs Deflection Label
    d_abs_max_idx = df['deflection'].abs().idxmax()
    d_val = df.loc[d_abs_max_idx]['deflection']
    d_x = df.loc[d_abs_max_idx]['x']
    if abs(d_val) > 1e-9:
        fig.add_annotation(x=d_x, y=d_val, text=f"{d_val:.2e}", showarrow=False, yshift=10, font=dict(color='red', size=10), row=4, col=1)

    # --- ADD VERTICAL DROP LINES (เส้นประแนวดิ่ง) ---
    # ลากเส้นผ่านทุก Subplot ตรงตำแหน่ง Key Points
    for kp in sorted_keys:
        fig.add_vline(x=kp, line_width=1, line_dash="dash", line_color="gray", opacity=0.5)

    # Layout Updates
    fig.update_layout(height=1000, showlegend=False, title_text="Analysis Results")
    fig.update_yaxes(title_text=f"V ({unit_force})", row=2, col=1)
    fig.update_yaxes(title_text=f"M ({unit_force}-{unit_len})", row=3, col=1)
    fig.update_yaxes(title_text="Deflection", row=4, col=1)
    
    st.plotly_chart(fig, use_container_width=True)

def render_result_tables(df, reac, spans):
    st.subheader("📍 Support Reactions")
    reac_data = []
    for i in range(len(spans)+1):
        ry = reac[2*i]
        mz = reac[2*i+1]
        reac_data.append({"Node": i+1, "Vertical (Ry)": f"{ry:.2f}", "Moment (Mz)": f"{mz:.2f}"})
    st.table(pd.DataFrame(reac_data))
