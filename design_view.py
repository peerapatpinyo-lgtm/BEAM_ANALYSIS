import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def draw_interactive_diagrams(df, spans, sup_df, loads, unit_force, unit_len):
    # --- LOAD COMBINATION REPORT ---
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

    # --- PLOTTING ---
    st.markdown("### 📊 Diagrams")
    total_len = sum(spans)
    cum_spans = [0] + list(np.cumsum(spans))
    
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=("Structure (FBD)", "Shear Force (V)", "Bending Moment (M)", "Deflection (δ)"),
                        row_heights=[0.15, 0.25, 0.3, 0.3])

    # 1. Structure
    fig.add_trace(go.Scatter(x=[0, total_len], y=[0, 0], line=dict(color='black', width=4)), row=1, col=1)
    
    sup_map = {int(r['id']): r['type'] for _, r in sup_df.iterrows()}
    for i, x in enumerate(cum_spans):
        if i in sup_map:
            stype = sup_map[i]
            sym = 'triangle-up' if stype == 'Pin' else ('circle' if stype == 'Roller' else 'square')
            fig.add_trace(go.Scatter(x=[x], y=[-0.1], mode='markers', marker=dict(symbol=sym, size=15, color='white', line=dict(color='black', width=2)), name=stype), row=1, col=1)
            if stype == 'Fixed':
                fig.add_shape(type="line", x0=x, y0=-0.2, x1=x, y1=0.2, line=dict(width=5, color='black'), row=1, col=1)

    for l in loads:
        x_abs = cum_spans[int(l['span_idx'])] + l['x']
        if l['type'] == 'P':
            fig.add_annotation(x=x_abs, y=0, ax=0, ay=-40, arrowhead=2, text=f"P={l['mag']}", row=1, col=1)
        elif l['type'] == 'M':
            fig.add_annotation(x=x_abs, y=0, text=f"↺ M={l['mag']}", showarrow=True, arrowhead=1, ax=0, ay=-30, arrowwidth=2, arrowcolor='purple', row=1, col=1)
            fig.add_trace(go.Scatter(x=[x_abs], y=[0], mode='markers', marker=dict(symbol='star', size=10, color='purple'), showlegend=False), row=1, col=1)
        elif l['type'] == 'U':
             xs = cum_spans[int(l['span_idx'])]
             xe = cum_spans[int(l['span_idx'])+1]
             fig.add_shape(type="rect", x0=xs, x1=xe, y0=0.05, y1=0.15, fillcolor="red", opacity=0.2, line_width=0, row=1, col=1)
             fig.add_annotation(x=(xs+xe)/2, y=0.2, text=f"w={l['mag']}", showarrow=False, row=1, col=1)

    # 2. Shear
    fig.add_trace(go.Scatter(x=df['x'], y=df['shear'], fill='tozeroy', line=dict(color='#D97706'), fillcolor='rgba(217, 119, 6, 0.1)', name="Shear"), row=2, col=1)
    
    # 3. Moment
    fig.add_trace(go.Scatter(x=df['x'], y=df['moment'], fill='tozeroy', line=dict(color='#2563EB'), fillcolor='rgba(37, 99, 235, 0.1)', name="Moment"), row=3, col=1)
    
    # 4. Deflection
    fig.add_trace(go.Scatter(x=df['x'], y=df['deflection'], fill='tozeroy', line=dict(color='#10B981'), fillcolor='rgba(16, 185, 129, 0.1)', name="Deflection"), row=4, col=1)

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
