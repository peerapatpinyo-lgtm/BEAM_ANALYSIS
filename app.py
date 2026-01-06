import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from solver import BeamSolver
import rc_design

st.set_page_config(page_title="Professional Beam Designer", layout="wide")

# ระบบจัดการ Session State เพื่อไม่ให้ Load หาย
if 'loads' not in st.session_state: st.session_state.loads = []
if 'results' not in st.session_state: st.session_state.results = None

st.title("🏗️ Beam Designer Pro (Fixed Version)")

# --- 1. การจัดการแรง (Load Management) ---
with st.container(border=True):
    col_g, col_l = st.columns([1, 2])
    with col_g:
        st.subheader("📏 Geometry")
        n_spans = st.number_input("Spans", 1, 5, 2)
        spans = [st.number_input("L %d (m)" % (i+1), 1.0, 15.0, 5.0, key="L_%d"%i) for i in range(n_spans)]
    
    with col_l:
        st.subheader("📥 Add Loads")
        c1, c2, c3, c4 = st.columns(4)
        l_idx = c1.selectbox("Span", range(n_spans))
        l_type = c2.selectbox("Type", ["Uniform", "Point", "Moment"])
        l_mag = c3.number_input("Mag (kN)", 10.0)
        l_x = c4.number_input("Pos (m)", 0.0)
        
        if st.button("➕ Add Load"):
            # กันเหนียว: ปรับ x ไม่ให้เกินความยาว Span
            safe_x = min(l_x, spans[l_idx] - 0.01)
            st.session_state.loads.append({
                'span': l_idx, 'type': l_type[0], 'mag': l_mag*1000, 
                'x': safe_x, 'dist': spans[l_idx] if l_type[0]=='U' else 0
            })
            st.rerun()

# ตารางแสดงแรงที่เพิ่มเข้าไป
if st.session_state.loads:
    with st.expander("📋 Current Load List"):
        st.table(pd.DataFrame(st.session_state.loads))
        if st.button("🗑️ Clear All"):
            st.session_state.loads = []; st.session_state.results = None; st.rerun()

# --- 2. ตั้งค่าวัสดุ ---
with st.sidebar:
    st.header("🧱 Materials")
    fc = st.number_input("f'c (MPa)", 28.0)
    fy = st.number_input("fy (MPa)", 400.0)
    b_m, h_m = st.number_input("b (m)", 0.3), st.number_input("h (m)", 0.5)
    cover, db_m = st.number_input("Cover (mm)", 35.0), st.selectbox("DB Main", [16, 20, 25])

# --- 3. ประมวลผลและวาดรูป ---
if st.button("🚀 EXECUTE ANALYSIS", type="primary", use_container_width=True):
    if not st.session_state.loads:
        st.error("Please add a load first!")
    else:
        supports = [{'id': i, 'type': 'Pin' if i==0 else 'Roller'} for i in range(n_spans+1)]
        sol = BeamSolver(spans, supports, st.session_state.loads, 2e11, (b_m * h_m**3)/12)
        st.session_state.results = sol.solve()

if st.session_state.results:
    df, reac, eq = st.session_state.results
    
    # 🖼️ DRAWING: Longitudinal Section (แนวยาว)
    st.header("🖼️ Part I: Longitudinal Detailing (รูปตัดแนวยาว)")
    fig_l = go.Figure()
    cum_l = 0
    for l in spans:
        # วาดคอนกรีต
        fig_l.add_shape(type="rect", x0=cum_l, y0=0, x1=cum_l+l, y1=h_m, line=dict(color="Black", width=2), fillcolor="rgba(200,200,200,0.2)")
        # วาดเหล็กเส้น (แบบจำลอง)
        fig_l.add_trace(go.Scatter(x=[cum_l, cum_l+l], y=[h_m-0.05, h_m-0.05], mode="lines", line=dict(color="red", width=2)))
        fig_l.add_trace(go.Scatter(x=[cum_l, cum_l+l], y=[0.05, 0.05], mode="lines", line=dict(color="blue", width=2)))
        cum_l += l
    fig_l.update_layout(height=200, showlegend=False, xaxis=dict(title="Length (m)"), yaxis=dict(visible=False, scaleanchor="x"))
    st.plotly_chart(fig_l, use_container_width=True)

    # 📋 DRAWING: Cross Sections (หน้าตัด)
    st.header("📋 Part II: Engineering Design Sheets")
    cum_dist = [0] + list(np.cumsum(spans))
    for i in range(n_spans):
        with st.container(border=True):
            s_df = df[(df['x'] >= cum_dist[i]) & (df['x'] <= cum_dist[i+1])]
            res = rc_design.design_span_expert(s_df['moment'].max()/1000, s_df['moment'].min()/1000, 
                                               s_df['shear'].abs().max()/1000, b_m, h_m, fc, fy, cover, db_m)
            
            c_calc, c_img = st.columns([1, 1])
            with c_calc:
                st.subheader("Span %d Results" % (i+1))
                st.latex(r"M_u^{(+)} = %.1f, M_u^{(-)} = %.1f \text{ kNm}" % (res['mu_pos'], res['mu_neg']))
                st.write("**Reinforcement:**")
                st.write(f"- Top: {res['neg']['n']} x DB{db_m} ({res['neg']['status']})")
                st.write(f"- Bottom: {res['pos']['n']} x DB{db_m} ({res['pos']['status']})")
            
            with c_img:
                st.write("**Cross-Section Drawing**")
                n_t, n_b = int(res['neg']['n']), int(res['pos']['n'])
                fig_cs = go.Figure()
                fig_cs.add_shape(type="rect", x0=0, y0=0, x1=b_m, y1=h_m, line=dict(color="Black", width=3))
                # เหล็กบน (Red)
                for j in range(n_t):
                    fig_cs.add_trace(go.Scatter(x=[(b_m/(n_t+1))*(j+1)], y=[h_m-0.05], mode="markers", marker=dict(color="Red", size=15)))
                # เหล็กล่าง (Blue)
                for j in range(n_b):
                    fig_cs.add_trace(go.Scatter(x=[(b_m/(n_b+1))*(j+1)], y=[0.05], mode="markers", marker=dict(color="Blue", size=15)))
                fig_cs.update_layout(width=250, height=250, showlegend=False, xaxis=dict(visible=False), yaxis=dict(visible=False))
                st.plotly_chart(fig_cs)
