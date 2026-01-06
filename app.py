import streamlit as st
import pandas as pd
import numpy as np
from solver import BeamSolver
import design_view

st.set_page_config(page_title="Beam Analysis Pro", layout="wide")

# Session State Initialize
if 'spans' not in st.session_state: st.session_state['spans'] = [5.0]
if 'supports' not in st.session_state: st.session_state['supports'] = [{'id': 0, 'type': 'Pin'}, {'id': 1, 'type': 'Roller'}]
if 'loads' not in st.session_state: st.session_state['loads'] = []

# Sidebar
st.sidebar.header("Section & Materials")
fc = st.sidebar.number_input("f'c (MPa)", 25.0)
fy = st.sidebar.number_input("fy (MPa)", 400.0)
b = st.sidebar.number_input("Width b (m)", 0.3)
h = st.sidebar.number_input("Depth h (m)", 0.5)
E_val = 2e11
I_val = (b * h**3) / 12

# Main UI Tabs (Spans, Supports, Loads)
# ... (ส่วนจัดการ Input เหมือนที่คุณมีอยู่แล้ว) ...

if st.button("🚀 Analyze", use_container_width=True):
    # ปรับปรุง Logic การรัน Analysis
    solver = BeamSolver(st.session_state['spans'], st.session_state['supports'], st.session_state['loads'], E_val, I_val, b=b, h=h)
    df, r, summ = solver.solve()
    
    if not df.empty:
        # 1. วาดกราฟ (เรียก design_view)
        # ส่ง List ของ Supports ที่ทำความสะอาดแล้วเพื่อให้ design_view ไม่ Error
        design_view.draw_interactive_diagrams(df, r, st.session_state['spans'], st.session_state['supports'], st.session_state['loads'])
        
        # 2. แสดงผล Summary
        st.markdown("### 📊 Summary Results")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Max Shear (V)", f"{summ['V_max']['value']/1000:.2f} kN")
        col2.metric("Max Moment (+)", f"{summ['M_pos']['value']/1000:.2f} kNm")
        col3.metric("Max Moment (-)", f"{summ['M_neg']['value']/1000:.2f} kNm")
        
        # 3. Deflection 2 วิธี (Elastic & Allowed)
        d_inst = summ['D_max']['value'] * 1000 # แปลงเป็น mm
        l_total = sum(st.session_state['spans'])
        d_allow = (l_total * 1000) / 240 # เกณฑ์ L/240
        
        col4.metric("Max Deflection", f"{d_inst:.2f} mm")
        st.write(f"**Deflection Check:** Immediate: {d_inst:.2f} mm | Allowable (L/240): {d_allow:.2f} mm")
        
        # 4. เรียก RC Design
        rc_res = solver.design_rc_section(fc, fy)
        st.success(f"Steel Required: Top {rc_res.get('as_neg',0):.2f} cm² | Bottom {rc_res.get('as_pos',0):.2f} cm²")
    else:
        st.error("Calculation failed. Please check your supports/loads.")
