import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- ENGINEERING COLORS ---
C_SHEAR_LINE = '#D97706'  # ส้มเข้ม
C_SHEAR_FILL = 'rgba(245, 158, 11, 0.2)' 
C_MOMENT_LINE = '#2563EB' # น้ำเงินเข้ม
C_MOMENT_FILL = 'rgba(37, 99, 235, 0.2)'
C_ZERO_LINE = 'black'
C_BEAM = '#111827'
C_SUP = '#4B5563'

def draw_interactive_diagrams(df, spans, sup_df, loads, unit_force, unit_len):
    total_len = sum(spans)
    cum_spans = [0] + list(np.cumsum(spans))
    
    fig = make_subplots(
        rows=3, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.08,
        row_heights=[0.2, 0.4, 0.4],
        subplot_titles=("<b>Structural Model</b>", "<b>Shear Force Diagram (SFD)</b>", "<b>Bending Moment Diagram (BMD)</b>")
    )

    # === 1. MODEL ===
    # Beam Line
    fig.add_trace(go.Scatter(
        x=[0, total_len], y=[0, 0],
        mode='lines', line=dict(color=C_BEAM, width=4),
        hoverinfo='skip'
    ), row=1, col=1)

    # Supports
    sup_x = [cum_spans[int(row['id'])] for _, row in sup_df.iterrows()]
    sup_txt = [row['type'] for _, row in sup_df.iterrows()]
    fig.add_trace(go.Scatter(
        x=sup_x, y=[0]*len(sup_x),
        mode='markers',
        marker=dict(symbol='triangle-up', size=18, color=C_SUP, line=dict(width=2, color='black')),
        text=sup_txt, hoverinfo='text'
    ), row=1, col=1)

    # Loads
    for l in loads:
        x_s = cum_spans[int(l['span_idx'])]
        if l['type'] == 'P':
            fig.add_annotation(
                x=x_s + l['x'], y=0, ax=0, ay=-50,
                arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor='#EF4444',
                text=f"P={l['P']}", font=dict(color='#EF4444', size=11),
                bgcolor="white", bordercolor='#EF4444', borderpad=2,
                row=1, col=1
            )
        elif l['type'] == 'U':
            x_e = cum_spans[int(l['span_idx'])+1]
            fig.add_shape(type="rect", x0=x_s, y0=0.08, x1=x_e, y1=0.25, line_width=0, fillcolor='#EF4444', opacity=0.15, row=1, col=1)
            fig.add_annotation(x=(x_s+x_e)/2, y=0.3, showarrow=False, text=f"w={l['w']}", font=dict(color='#EF4444', size=11), bgcolor="white", row=1, col=1)

    # === 2. SHEAR FORCE (Textbook Style: Vertical Jumps) ===
    # Zero Line
    fig.add_shape(type="line", x0=0, x1=total_len, y0=0, y1=0, line=dict(color=C_ZERO_LINE, width=1.5), row=2, col=1)
    
    # Plot Graph
    # ใช้ mode='lines' ธรรมดา แต่ข้อมูลเรามีจุดซ้อนกัน (x, x+epsilon) ทำให้เกิดเส้นดิ่งเองตามธรรมชาติ
    fig.add_trace(go.Scatter(
        x=df['x'], y=df['shear'],
        mode='lines', 
        line=dict(color=C_SHEAR_LINE, width=2),
        fill='tozeroy', fillcolor=C_SHEAR_FILL,
        name='Shear'
    ), row=2, col=1)

    # Label Max Shear
    v_max = df['shear'].abs().max()
    if v_max > 1e-3:
        # หาจุด Max ที่ไม่ใช่จุดซ้ำ (เพื่อความสวยงามของการวาง Label)
        row_v = df.iloc[df['shear'].abs().argmax()]
        fig.add_annotation(
            x=row_v['x'], y=row_v['shear'],
            text=f"<b>{row_v['shear']:.2f}</b>",
            showarrow=True, arrowhead=1, ax=0, ay=-25 if row_v['shear'] > 0 else 25,
            font=dict(color=C_SHEAR_LINE), row=2, col=1
        )

    # === 3. BENDING MOMENT (Textbook Style: Curves & Cusps) ===
    # Zero Line
    fig.add_shape(type="line", x0=0, x1=total_len, y0=0, y1=0, line=dict(color=C_ZERO_LINE, width=1.5), row=3, col=1)

    # Plot Graph (No Spline Smoothing!)
    fig.add_trace(go.Scatter(
        x=df['x'], y=df['moment'],
        mode='lines', 
        line=dict(color=C_MOMENT_LINE, width=2), # ปล่อยเป็น Linear เพราะจุดถี่มาก จะเห็นเป็น Curve เอง
        fill='tozeroy', fillcolor=C_MOMENT_FILL,
        name='Moment'
    ), row=3, col=1)

    # Label Moments
    m_max = df['moment'].max()
    m_min = df['moment'].min()
    
    # Positive (Sagging)
    if m_max > 1e-3:
        xm = df.loc[df['moment'] == m_max, 'x'].iloc[0]
        fig.add_annotation(
            x=xm, y=m_max, text=f"<b>{m_max:.2f}</b>", 
            showarrow=True, arrowhead=1, ax=0, ay=-30, 
            font=dict(color=C_MOMENT_LINE), row=3, col=1
        )
    # Negative (Hogging)
    if m_min < -1e-3:
        xm = df.loc[df['moment'] == m_min, 'x'].iloc[0]
        fig.add_annotation(
            x=xm, y=m_min, text=f"<b>{m_min:.2f}</b>", 
            showarrow=True, arrowhead=1, ax=0, ay=30, 
            font=dict(color=C_MOMENT_LINE), row=3, col=1
        )

    # === GLOBAL LAYOUT ===
    fig.update_layout(
        height=700,
        margin=dict(l=50, r=20, t=40, b=40),
        paper_bgcolor='white', plot_bgcolor='white',
        hovermode="x unified", showlegend=False,
        font=dict(family="Sarabun, sans-serif", size=13)
    )
    
    axis_style = dict(
        showline=True, linewidth=1, linecolor='#E5E7EB',
        showgrid=True, gridcolor='#F3F4F6',
        zeroline=False, showticklabels=True
    )
    fig.update_xaxes(**axis_style)
    fig.update_yaxes(**axis_style)
    fig.update_yaxes(visible=False, range=[-0.5, 0.5], row=1, col=1) # Lock Model View

    st.plotly_chart(fig, use_container_width=True)

def render_result_tables(df, reac, spans):
    # (คงเดิมไว้ครับ ส่วนนี้ถูกต้องแล้ว)
    st.subheader("4. ผลการวิเคราะห์ (Analysis Results)")
    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Max Shear", f"{df['shear'].abs().max():.2f}")
    with c2: st.metric("Max Moment (+)", f"{df['moment'].max():.2f}")
    with c3: st.metric("Max Moment (-)", f"{df['moment'].min():.2f}")

    tab1, tab2 = st.tabs(["📍 แรงปฏิกิริยา (Reactions)", "📄 ตารางค่าละเอียด (Table)"])
    with tab1:
        r_data = []
        for i in range(len(spans)+1):
            r_data.append({"Node": f"{i+1}", "Ry": f"{reac[2*i]:.2f}", "Mz": f"{reac[2*i+1]:.2f}"})
        st.dataframe(pd.DataFrame(r_data), hide_index=True, use_container_width=True)
    with tab2:
        # กรองข้อมูลแสดงเฉพาะจุดที่ไม่ซ้ำ (เพื่อไม่ให้ตารางยาวเกินไปจากการทำกราฟละเอียด)
        df_show = df.drop_duplicates(subset=['x'], keep='first').copy()
        df_show.columns = ["Distance (x)", "Shear (V)", "Moment (M)"]
        st.dataframe(df_show.style.format("{:.2f}"), use_container_width=True, height=300)
