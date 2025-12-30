import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- ENGINEERING COLORS ---
C_SHEAR_LINE = '#D97706'  # เส้นสีส้มเข้ม
C_SHEAR_FILL = 'rgba(245, 158, 11, 0.2)' 
C_MOMENT_LINE = '#2563EB' # เส้นสีน้ำเงินเข้ม
C_MOMENT_FILL = 'rgba(37, 99, 235, 0.2)'
C_ZERO_LINE = 'black'     # เส้นศูนย์สีดำ
C_BEAM = 'black'
C_SUP = '#4B5563'

def draw_interactive_diagrams(df, spans, sup_df, loads, unit_force, unit_len):
    total_len = sum(spans)
    cum_spans = [0] + list(np.cumsum(spans))
    
    # สร้าง 3 กราฟเรียงกัน
    fig = make_subplots(
        rows=3, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.08,
        row_heights=[0.2, 0.4, 0.4],
        subplot_titles=("<b>Structural Model</b>", f"<b>Shear Force Diagram (SFD)</b>", f"<b>Bending Moment Diagram (BMD)</b>")
    )

    # ==========================
    # 1. STRUCTURAL MODEL
    # ==========================
    # เส้นคาน (Main Beam)
    fig.add_trace(go.Scatter(
        x=[0, total_len], y=[0, 0],
        mode='lines', line=dict(color=C_BEAM, width=4),
        hoverinfo='skip'
    ), row=1, col=1)

    # จุดรองรับ (Supports) - วาดให้แตะเส้นคานพอดี
    sup_x = [cum_spans[int(row['id'])] for _, row in sup_df.iterrows()]
    sup_txt = [row['type'] for _, row in sup_df.iterrows()]
    
    # ใช้ Marker สามเหลี่ยม
    fig.add_trace(go.Scatter(
        x=sup_x, y=[0]*len(sup_x), # วางที่ 0 เป๊ะ
        mode='markers',
        marker=dict(symbol='triangle-up', size=20, color=C_SUP, line=dict(width=2, color='black')),
        text=sup_txt, hoverinfo='text', name='Support'
    ), row=1, col=1)

    # Loads (Visual Arrows)
    for l in loads:
        x_s = cum_spans[int(l['span_idx'])]
        if l['type'] == 'P':
            # Arrow
            fig.add_annotation(
                x=x_s + l['x'], y=0, ax=0, ay=-60,
                arrowhead=2, arrowsize=1.2, arrowwidth=2, arrowcolor='#DC2626',
                text=f"P={l['P']}", font=dict(color='#DC2626'),
                row=1, col=1
            )
        elif l['type'] == 'U':
            x_e = cum_spans[int(l['span_idx'])+1]
            # UDL Block
            fig.add_shape(
                type="rect", x0=x_s, y0=0.05, x1=x_e, y1=0.25,
                line=dict(width=0), fillcolor='#DC2626', opacity=0.15,
                row=1, col=1
            )
            fig.add_annotation(
                x=(x_s+x_e)/2, y=0.3, showarrow=False,
                text=f"w={l['w']}", font=dict(color='#DC2626'),
                row=1, col=1
            )

    # ==========================
    # 2. SHEAR FORCE (SFD)
    # ==========================
    # เส้น Zero Line
    fig.add_shape(type="line", x0=0, x1=total_len, y0=0, y1=0, line=dict(color=C_ZERO_LINE, width=2), row=2, col=1)
    
    # กราฟ Shear
    fig.add_trace(go.Scatter(
        x=df['x'], y=df['shear'],
        mode='lines', 
        line=dict(color=C_SHEAR_LINE, width=2, shape='linear'), # shape='linear' เพราะเรามีจุดเยอะละเอียดแล้ว
        fill='tozeroy', # ถมสีลงหาเส้นศูนย์เสมอ
        fillcolor=C_SHEAR_FILL,
        name='Shear'
    ), row=2, col=1)

    # Label Max Shear
    v_max = df['shear'].abs().max()
    if v_max > 0:
        row_v = df.loc[df['shear'].abs() == v_max].iloc[0]
        fig.add_annotation(
            x=row_v['x'], y=row_v['shear'],
            text=f"<b>Vmax: {row_v['shear']:.2f}</b>",
            showarrow=True, arrowhead=1, ax=40, ay=-30 if row_v['shear'] > 0 else 30,
            font=dict(color=C_SHEAR_LINE), row=2, col=1
        )

    # ==========================
    # 3. BENDING MOMENT (BMD)
    # ==========================
    # เส้น Zero Line
    fig.add_shape(type="line", x0=0, x1=total_len, y0=0, y1=0, line=dict(color=C_ZERO_LINE, width=2), row=3, col=1)

    # กราฟ Moment
    fig.add_trace(go.Scatter(
        x=df['x'], y=df['moment'],
        mode='lines', 
        line=dict(color=C_MOMENT_LINE, width=2, shape='spline', smoothing=0), # spline for curvature
        fill='tozeroy', 
        fillcolor=C_MOMENT_FILL,
        name='Moment'
    ), row=3, col=1)

    # Label Moments
    m_max = df['moment'].max()
    m_min = df['moment'].min()
    
    if m_max > 1:
        xm = df.loc[df['moment'] == m_max, 'x'].iloc[0]
        fig.add_annotation(x=xm, y=m_max, text=f"<b>M(+): {m_max:.2f}</b>", showarrow=True, arrowhead=1, ax=0, ay=-30, font=dict(color=C_MOMENT_LINE), row=3, col=1)
    
    if m_min < -1:
        xm = df.loc[df['moment'] == m_min, 'x'].iloc[0]
        fig.add_annotation(x=xm, y=m_min, text=f"<b>M(-): {m_min:.2f}</b>", showarrow=True, arrowhead=1, ax=0, ay=30, font=dict(color=C_MOMENT_LINE), row=3, col=1)

    # ==========================
    # LAYOUT SETTINGS
    # ==========================
    fig.update_layout(
        height=700,
        margin=dict(l=50, r=20, t=40, b=40),
        paper_bgcolor='white', plot_bgcolor='white',
        hovermode="x unified", showlegend=False,
        font=dict(family="sans-serif", size=12)
    )
    
    # Engineering Grid Style
    axis_style = dict(
        showline=True, linewidth=1, linecolor='black', # เส้นกรอบดำ
        showgrid=True, gridcolor='#F3F4F6', gridwidth=1, # Grid จางๆ
        zeroline=False, # เราวาด Zeroline เองแล้ว ปิด Auto
        showticklabels=True
    )
    fig.update_xaxes(**axis_style)
    fig.update_yaxes(**axis_style)
    
    # Fix Model Y-Axis to center beam
    fig.update_yaxes(range=[-0.5, 0.5], visible=False, row=1, col=1)

    st.plotly_chart(fig, use_container_width=True)

def render_result_tables(df, reac, spans):
    # (ใช้ Code เดิมส่วนนี้ได้เลยครับ มันโอเคแล้ว)
    st.markdown("##### 4. Analysis Results")
    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Max Shear", f"{df['shear'].abs().max():.2f}")
    with c2: st.metric("Max Moment (+)", f"{df['moment'].max():.2f}")
    with c3: st.metric("Max Moment (-)", f"{df['moment'].min():.2f}")

    tab1, tab2 = st.tabs(["📍 Support Reactions", "📄 Shear/Moment Values"])
    with tab1:
        r_data = []
        for i in range(len(spans)+1):
            r_data.append({"Node": i+1, "Ry": f"{reac[2*i]:.2f}", "Mz": f"{reac[2*i+1]:.2f}"})
        st.dataframe(pd.DataFrame(r_data), hide_index=True, use_container_width=True)
    with tab2:
        df_show = df.copy()
        df_show.columns = ["x", "V", "M"]
        st.dataframe(df_show.style.format("{:.2f}"), use_container_width=True, height=300)
