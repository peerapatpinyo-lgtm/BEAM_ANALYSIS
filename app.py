import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from solver import BeamSolver
import rc_design
import design_view

st.set_page_config(page_title="Professional Beam Designer Pro", layout="wide")

# ระบบจำข้อมูล (Persistence)
if 'loads' not in st.session_state: st.session_state.loads = []
if 'analysis_results' not in st.session_state: st.session_state.analysis_results = None

# --- ส่วนควบคุมด้านข้าง (Sidebar) ---
with st.sidebar:
    st.header("⚙️ คอนฟิกโครงการ")
    with st.expander("คุณสมบัติวัสดุ", expanded=True):
        fc = st.number_input("f'c คอนกรีต (MPa)", 24.0, 50.0, 28.0)
        fy = st.number_input("fy เหล็กหลัก (MPa)", 300.0, 500.0, 400.0)
        fyt = st.number_input("fy เหล็กปลอก (MPa)", 230.0, 400.0, 240.0)
    
    with st.expander("ขนาดหน้าตัดคาน", expanded=True):
        b_m = st.number_input("ความกว้าง b (m)", 0.1, 1.0, 0.3)
        h_m = st.number_input("ความลึก h (m)", 0.2, 2.0, 0.5)
        cover = st.number_input("ระยะหุ้ม Covering (mm)", 20, 75, 30)

st.title("🏗️ โปรแกรมออกแบบคานคอนกรีตเสริมเหล็กมืออาชีพ")
st.caption("Design Method: Strength Design Method (SDM) | Reference: ACI 318-19, EIT 1008-38")

# --- 1. การจัดการแรงและโครงสร้าง ---
col_geom, col_load = st.columns([1, 2])

with col_geom:
    st.subheader("📏 ข้อมูลช่วงคาน (Span)")
    n_spans = st.number_input("จำนวนช่วงคาน", 1, 10, 1)
    spans = []
    for i in range(n_spans):
        spans.append(st.number_input(f"ความยาว Span {i+1} (m)", 0.1, 20.0, 5.0, key=f"span_l_{i}"))

with col_load:
    st.subheader("📥 เพิ่มแรงกระทำ (Load Manager)")
    with st.container(border=True):
        l_col1, l_col2, l_col3 = st.columns(3)
        l_span = l_col1.selectbox("เลือกช่วง", range(n_spans), format_func=lambda x: f"Span {x+1}")
        l_type = l_col2.selectbox("ประเภทแรง", ["Uniform (U)", "Point (P)"])
        l_mag = l_col3.number_input("ขนาดแรง (kN/m หรือ kN)", 0.0)
        
        if st.button("➕ เพิ่มแรงเข้าสู่ระบบ", use_container_width=True):
            st.session_state.loads.append({
                'span_index': l_span, 'type': l_type[0], 'mag': l_mag*1000, 
                'x': 0.0, 'dist': spans[l_span] if l_type[0]=='U' else 0.0
            })
            st.rerun()

    # แสดงรายการแรงที่ใส่ไปแล้ว
    if st.session_state.loads:
        st.write("รายการแรงปัจจุบัน:")
        for idx, ld in enumerate(st.session_state.loads):
            c1, c2 = st.columns([5, 1])
            c1.code(f"Span {ld['span_index']+1} | {ld['type']} | {ld['mag']/1000} kN")
            if c2.button("🗑️", key=f"del_{idx}"):
                st.session_state.loads.pop(idx); st.rerun()

# --- 2. การวิเคราะห์โครงสร้าง ---
st.divider()
if st.button("🚀 เริ่มการวิเคราะห์และออกแบบ", type="primary", use_container_width=True):
    supports = [{'id': i, 'type': 'Pin' if i==0 else 'Roller'} for i in range(n_spans+1)]
    sol = BeamSolver(spans, supports, st.session_state.loads, 2e11, b_m, h_m)
    st.session_state.analysis_results = sol.solve()

if st.session_state.analysis_results:
    df, reac, eq = st.session_state.analysis_results
    
    st.header("📊 ส่วนที่ 1: ผลการวิเคราะห์ (Analysis Results)")
    design_view.draw_interactive_diagrams(df, reac, spans, pd.DataFrame([{'type':'Pin'}]*(n_spans+1)), st.session_state.loads)

    # ตารางค่าแรงภายในและ Reaction
    c_tab1, c_tab2 = st.columns([2, 1])
    with c_tab1:
        st.subheader("แรงปฏิกิริยาและค่าสูงสุด")
        st.dataframe(reac, use_container_width=True)
    with c_tab2:
        st.subheader("การตรวจสอบสมดุล (ΣF=0)")
        st.metric("ค่าคลาดเคลื่อนแนวแกน Y", f"{abs(eq['l_fy']-eq['r_fy']):.4f} N")
        st.metric("ค่าคลาดเคลื่อน Moment", f"{abs(eq['l_m0']-eq['r_m0']):.4f} Nm")

    # --- 3. การออกแบบหน้าตัดและการปรับแก้จำนวนเหล็ก ---
    st.divider()
    st.header("🧱 ส่วนที่ 2: การออกแบบหน้าตัด (Reinforcement Design)")
    
    # ตัวเลือกขนาดเหล็กหลัก
    col_db1, col_db2 = st.columns(2)
    db_m = col_db1.selectbox("เลือกขนาดเหล็กหลัก (DB)", [12, 16, 20, 25, 28], index=2)
    db_s = col_db2.selectbox("เลือกขนาดเหล็กปลอก (RB/DB)", [6, 9, 12], index=1)

    cum_dist = [0] + list(np.cumsum(spans))
    
    for i in range(n_spans):
        with st.expander(f"📍 การออกแบบและปรับปรุง Span {i+1} (L = {spans[i]} m)", expanded=True):
            # ดึงค่าแรงสูงสุดในช่วงนั้น
            s_df = df[(df['x'] >= cum_dist[i]) & (df['x'] <= cum_dist[i+1])]
            res = rc_design.design_span(s_df['moment'].max()/1000, s_df['moment'].min()/1000, s_df['shear'].abs().max()/1000, b_m, h_m, fc, fy, fyt, cover, db_m, db_s)
            
            c_calc, c_edit, c_view = st.columns([1.5, 1, 1.2])
            
            with c_calc:
                st.write("**ผลการคำนวณอัตโนมัติ:**")
                st.latex(rf"M_{{u,pos}} = {res['mu_pos']:.2f} \text{{ kNm}}")
                st.latex(rf"A_{{s,req}} = {res['pos']['as_req']:.1f} \text{{ mm}}^2")
                st.info(f"แนะนา: บน {res['neg']['n']} เส้น | ล่าง {res['pos']['n']} เส้น")

            with c_edit:
                st.write("**🔧 ปรับแก้จำนวนเหล็ก:**")
                n_top_edit = st.number_input(f"จำนวนเหล็กบน (Span {i+1})", 2, 20, int(res['neg']['n']), key=f"top_{i}")
                n_bot_edit = st.number_input(f"จำนวนเหล็กล่าง (Span {i+1})", 2, 20, int(res['pos']['n']), key=f"bot_{i}")
                stirrup_s = st.number_input(f"ระยะปลอก (mm)", 50, 300, int(res['spacing']), step=25, key=f"s_{i}")

            with c_view:
                # วาดรูปหน้าตัดที่ปรับแก้แล้ว
                fig_cs = go.Figure()
                fig_cs.add_shape(type="rect", x0=0, y0=0, x1=b_m, y1=h_m, fillcolor="rgba(180,180,180,0.2)", line=dict(color="Black", width=3))
                # เหล็กบน
                for j in range(n_top_edit):
                    fig_cs.add_trace(go.Scatter(x=[(b_m/(n_top_edit+1))*(j+1)], y=[h_m-(cover/1000)], mode='markers', marker=dict(color='Red', size=db_m)))
                # เหล็กล่าง
                for j in range(n_bot_edit):
                    fig_cs.add_trace(go.Scatter(x=[(b_m/(n_bot_edit+1))*(j+1)], y=[cover/1000], mode='markers', marker=dict(color='Blue', size=db_m)))
                
                fig_cs.update_layout(width=220, height=280, showlegend=False, xaxis=dict(visible=False), yaxis=dict(visible=False), margin=dict(l=10,r=10,t=10,b=10))
                st.plotly_chart(fig_cs)

st.success("✅ ข้อมูลการวิเคราะห์จะถูกเก็บไว้เสมอจนกว่าจะมีการเปลี่ยนแปลงโครงสร้างหรือกด RUN ใหม่")
