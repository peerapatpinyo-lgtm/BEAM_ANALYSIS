import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- COLORS & STYLES ---
C_BEAM = 'black'
C_PIN_ROLLER = 'white' 
C_OUTLINE = 'black'
C_SHEAR_LINE = '#D97706'    
C_SHEAR_FILL = 'rgba(217, 119, 6, 0.1)' 
C_MOMENT_LINE = '#2563EB'   
C_MOMENT_FILL = 'rgba(37, 99, 235, 0.1)'
C_DEFLECT_LINE = '#10B981'  
C_DEFLECT_FILL = 'rgba(16, 185, 129, 0.1)'
C_TEXT_BG = "rgba(255,255,255,0.9)"

def draw_interactive_diagrams(df, spans, sup_df, loads, unit_force, unit_len):
    st.markdown("### 📊 Analysis Diagrams")

    # --- LOAD TABLE DISPLAY (Validation Re-check) ---
    if loads:
        with st.expander("📋 Loads Summary Check", expanded=False):
            data_sum = []
            for i, l in enumerate(loads):
                if l['type'] == 'P':
                    desc = f"Point Load {l['mag']} {unit_force}"
                    pos = f"@ x = {l['x']} {unit_len} (Span {l['span_idx']+1})"
                elif l['type'] == 'U':
                    desc = f"UDL {l['mag']} {unit_force}/{unit_len}"
                    pos = f"Full Span {l['span_idx']+1}"
                elif l['type'] == 'M':
                    desc = f"Moment {l['mag']} {unit_force}-{unit_len}"
                    pos = f"@ x = {l['x']} {unit_len} (Span {l['span_idx']+1})"
                data_sum.append({"Load": desc, "Position": pos})
            st.table(pd.DataFrame(data_sum))

    if df is None or df.empty:
        return

    total_len = sum(spans)
    cum_spans = [0] + list(np.cumsum(spans))
    
    # Create Subplots
    fig = make_subplots(
        rows=4, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.08,
        row_heights=[0.15, 0.25, 0.30, 0.30],
        subplot_titles=("Structure (FBD)", "Shear Force (V)", "Bending Moment (M)", "Deflection (δ)")
    )

    # 1. Structure
    fig.add_trace(go.Scatter(x=[0, total_len], y=[0, 0], mode='lines', line=dict(color='black', width=4), hoverinfo='none'), row=1, col=1)
    
    # Supports Visuals
    sup_map = {int(r['id']): r['type'] for _, r in sup_df.iterrows()}
    for i, x in enumerate(cum_spans):
        if i in sup_map:
            stype = sup_map[i]
            if stype == 'Pin':
                fig.add_trace(go.Scatter(x=[x], y=[-0.1], mode='markers', marker=dict(symbol='triangle-up', size=15, color='white', line=dict(width=2, color='black')), name="Pin"), row=1, col=1)
            elif stype == 'Roller':
                fig.add_trace(go.Scatter(x=[x], y=[-0.1], mode='markers', marker=dict(symbol='circle', size=15, color='white', line=dict(width=2, color='black')), name="Roller"), row=1, col=1)
            elif stype == 'Fixed':
                fig.add_shape(type="line", x0=x, y0=-0.2, x1=x, y1=0.2, line=dict(width=5, color='black'), row=1, col=1)

    # Load Visuals
    for l in loads:
        x_abs = cum_spans[int(l['span_idx'])] + l['x']
        if l['type'] == 'P':
            fig.add_annotation(x=x_abs, y=0, ax=0, ay=-40, arrowhead=2, text=f"P={l['mag']}", row=1, col=1)
        elif l['type'] == 'M':
            fig.add_annotation(x=x_abs, y=0, text=f"↺ M={l['mag']}", showarrow=True, arrowhead=1, ax=0, ay=-30, row=1, col=1)
        elif l['type'] == 'U':
             x_start = cum_spans[int(l['span_idx'])]
             x_end = cum_spans[int(l['span_idx']) + 1]
             fig.add_shape(type="rect", x0=x_start, x1=x_end, y0=0.05, y1=0.15, fillcolor="red", opacity=0.2, line_width=0, row=1, col=1)
             fig.add_annotation(x=(x_start+x_end)/2, y=0.2, text=f"w={l['mag']}", showarrow=False, row=1, col=1)

    # 2. Shear
    fig.add_trace(go.Scatter(x=df['x'], y=df['shear'], fill='tozeroy', line=dict(color=C_SHEAR_LINE), fillcolor=C_SHEAR_FILL, name="Shear"), row=2, col=1)
    
    # 3. Moment (With Zero Check)
    fig.add_trace(go.Scatter(x=df['x'], y=df['moment'], fill='tozeroy', line=dict(color=C_MOMENT_LINE), fillcolor=C_MOMENT_FILL, name="Moment"), row=3, col=1)
    
    # Annotate Max/Min Moment
    m_max = df['moment'].max()
    m_min = df['moment'].min()
    # Only annotate significant values
    if abs(m_max) > 0.01:
        xm = df.loc[df['moment'].idxmax(), 'x']
        fig.add_annotation(x=xm, y=m_max, text=f"{m_max:.2f}", showarrow=False, yshift=10, row=3, col=1)
    if abs(m_min) > 0.01:
        xm = df.loc[df['moment'].idxmin(), 'x']
        fig.add_annotation(x=xm, y=m_min, text=f"{m_min:.2f}", showarrow=False, yshift=-10, row=3, col=1)

    # 4. Deflection
    fig.add_trace(go.Scatter(x=df['x'], y=df['deflection'], fill='tozeroy', line=dict(color=C_DEFLECT_LINE), fillcolor=C_DEFLECT_FILL, name="Deflection"), row=4, col=1)

    # Layout Formatting
    fig.update_layout(height=1000, showlegend=False, hovermode="x unified", title_text="Beam Analysis Results")
    fig.update_xaxes(showgrid=True, gridcolor='#eee')
    fig.update_yaxes(showgrid=True, gridcolor='#eee')
    
    # Axis Titles
    fig.update_yaxes(title_text=f"V ({unit_force})", row=2, col=1)
    fig.update_yaxes(title_text=f"M ({unit_force}-{unit_len})", row=3, col=1)
    fig.update_yaxes(title_text=f"δ (m)", row=4, col=1)
    fig.update_xaxes(title_text=f"Distance x ({unit_len})", row=4, col=1)

    st.plotly_chart(fig, use_container_width=True)

def render_result_tables(df, reac, spans):
    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    c1.metric("Max Shear", f"{df['shear'].abs().max():.2f}")
    c2.metric("Max Moment", f"{df['moment'].abs().max():.2f}")
    c3.metric("Max Deflection", f"{df['deflection'].abs().max():.4e}")
    
    st.subheader("📍 Support Reactions")
    reac_data = []
    for i in range(len(spans)+1):
        ry = reac[2*i]
        mz = reac[2*i+1]
        # Clean small numbers
        if abs(ry) < 1e-5: ry = 0.0
        if abs(mz) < 1e-5: mz = 0.0
        
        reac_data.append({
            "Node": i+1, 
            "Vertical (Ry)": f"{ry:.2f}", 
            "Moment (Mz)": f"{mz:.2f}"
        })
    st.table(pd.DataFrame(reac_data))
