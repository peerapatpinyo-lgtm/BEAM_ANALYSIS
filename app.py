import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from solver import BeamSolver # ตรวจสอบว่าไฟล์ solver.py ของคุณพร้อมใช้งาน
import rc_design

st.set_page_config(page_title="Professional Beam Design", layout="wide")

# --- การจัดการ Load ให้ไม่หาย (Session State) ---
if 'load_list' not in st.session_state:
    st.session_state.load_list = []

st.title("🏗️ Beam Designer Pro")

# ส่วนรับค่า Geometry
with st.sidebar:
    st.header("🧱 Material & Section")
    fc = st.number_input("f'c (MPa)", 28)
    fy = st.number_input("fy (MPa)", 400)
    b_m = st.number_input("Width b (m)", 0.3)
    h_m = st.number_input("Height h (m)", 0.5)
    db_main = st.selectbox("Main Bar (DB)", [12, 16, 20, 25])
    cover = 35

# ส่วนการกรอก Load
with st.container(border=True):
    st.subheader("📥 Add Loads")
    c1, c2, c3, c4 = st.columns(4)
    n_spans = c1.number_input("Number of Spans", 1, 5, 1)
    span_lengths = [5.0] * n_spans # ตัวอย่าง
    
    l_type = c2.selectbox("Type", ["Point", "Uniform"])
    l_val = c3.number_input("Value (kN or kN/m)", 10.0)
    l_pos = c4.number_input("Position (m)", 0.0)
    
    if st.button("➕ Add Load"):
        st.session_state.load_list.append({'type': l_type[0], 'mag': l_val*1000, 'x': l_pos})
        st.rerun()

    if st.session_state.load_list:
        st.write("**Current Loads:**")
        st.dataframe(pd.DataFrame(st.session_state.load_list), use_container_width=True)
        if st.button("🗑️ Clear All Loads"):
            st.session_state.load_list = []
            st.rerun()

# --- ส่วนประมวลผลและวาดรูป ---
if st.button("🚀 Run Analysis & Design", type="primary", use_container_width=True):
    # จำลองการทำงานของ Solver (คุณต้องเชื่อมกับ solver.py จริงของคุณ)
    # สมมติว่าคืนค่า df ที่มีคอลัมน์ 'x', 'moment', 'shear'
    supports = [{'id': i, 'type': 'Pin' if i==0 else 'Roller'} for i in range(n_spans+1)]
    sol = BeamSolver([5.0]*n_spans, supports, st.session_state.load_list, 2e11, (b_m*h_m**3)/12)
    df, reactions, _ = sol.solve()
    
    # แก้ปัญหา KeyError 'x' โดยการตรวจสอบชื่อคอลัมน์
    x_key = 'x' if 'x' in df.columns else df.columns[0]
    
    st.divider()
    
    # --- รูปตัดแนวยาว (Longitudinal Section) ---
    st.subheader("📏 Longitudinal Detailing (รูปตัดแนวยาว)")
    
    
    fig_long = go.Figure()
    # วาดคอนกรีตคาน
    fig_long.add_shape(type="rect", x0=0, y0=0, x1=sum([5.0]*n_spans), y1=h_m, line=dict(color="Black"), fillcolor="LightGrey", opacity=0.3)
    # วาดเหล็กเส้น (ตัวอย่างเหล็กบน-ล่าง)
    fig_long.add_trace(go.Scatter(x=[0, sum([5.0]*n_spans)], y=[h_m-0.05, h_m-0.05], mode='lines', name='Top Steel', line=dict(color='Red', width=3)))
    fig_long.add_trace(go.Scatter(x=[0, sum([5.0]*n_spans)], y=[0.05, 0.05], mode='lines', name='Bottom Steel', line=dict(color='Blue', width=3)))
    
    fig_long.update_layout(height=200, showlegend=False, xaxis_title="Beam Length (m)", yaxis=dict(visible=False, scaleanchor="x"))
    st.plotly_chart(fig_long, use_container_width=True)

    # --- รูปตัดขวาง (Cross Sections) ---
    st.subheader("📋 Cross Section Design (รูปตัดขวาง)")
    
    
    cols = st.columns(n_spans)
    for i in range(n_spans):
        with cols[i]:
            st.write(f"**Span {i+1}**")
            # หาโมเมนต์สูงสุดใน Span นี้
            span_m = df[(df[x_key] >= i*5.0) & (df[x_key] <= (i+1)*5.0)]['moment']
            m_pos = span_m.max() / 1000
            m_neg = span_m.min() / 1000
            
            n_bot, _ = rc_design.design_section(m_pos, b_m, h_m, fc, fy, cover, db_main)
            n_top, _ = rc_design.design_section(m_neg, b_m, h_m, fc, fy, cover, db_main)
            
            # วาดหน้าตัดด้วย Plotly
            fig_sect = go.Figure()
            # ขอบคอนกรีต
            fig_sect.add_shape(type="rect", x0=0, y0=0, x1=b_m, y1=h_m, line=dict(color="Black", width=4))
            # วาดเหล็กปลอก (Stirrup)
            fig_sect.add_shape(type="rect", x0=0.04, y0=0.04, x1=b_m-0.04, y1=h_m-0.04, line=dict(color="Gray", width=2))
            
            # วาดจุดเหล็กเสริมบน (Top Bars)
            for b in range(int(n_top)):
                fig_sect.add_trace(go.Scatter(x=[(b_m/(n_top+1))*(b+1)], y=[h_m-0.06], mode='markers', marker=dict(color='Red', size=12)))
            
            # วาดจุดเหล็กเสริมล่าง (Bottom Bars)
            for b in range(int(n_bot)):
                fig_sect.add_trace(go.Scatter(x=[(b_m/(n_bot+1))*(b+1)], y=[0.06], mode='markers', marker=dict(color='Blue', size=12)))
                
            fig_sect.update_layout(width=200, height=250, showlegend=False, xaxis=dict(visible=False), yaxis=dict(visible=False), margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_sect)
            st.caption(f"Top: {int(n_top)}-DB{db_m} / Bot: {int(n_bot)}-DB{db_m}")
