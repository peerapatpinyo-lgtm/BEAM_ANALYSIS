import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from solver import BeamSolver
import rc_design

st.set_page_config(page_title="Professional Beam Studio", layout="wide")

# ระบบจัดการ Load ที่เสถียร (Session State)
if 'loads' not in st.session_state: st.session_state.loads = []
if 'trigger_solve' not in st.session_state: st.session_state.trigger_solve = False

st.title("🏗️ Professional Beam Studio (Expert Mode)")

# --- 1. GEOMETRY & STABLE LOADING ---
with st.container(border=True):
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("📏 Geometry")
        n_spans = st.number_input("Spans", 1, 10, 1)
        spans = [st.number_input(f"L{i+1} (m)", 0.1, 20.0, 5.0, key=f"L_{i}") for i in range(n_spans)]
    
    with col2:
        st.subheader("📥 Load Input")
        lc1, lc2, lc3, lc4, lc5 = st.columns(5)
        l_span = lc1.selectbox("Span", range(n_spans))
        l_type = lc2.selectbox("Type", ["P (Point)", "U (Uniform)", "M (Moment)"])
        l_mag = lc3.number_input("Value", 10.0)
        l_x = lc4.number_input("Start x", 0.0, float(spans[l_span]))
        l_end = lc5.number_input("End x", float(spans[l_span])) if "U" in l_type else l_x

        if st.button("➕ Add Load"):
            st.session_state.loads.append({
                'span': l_span, 'type': l_type[0], 'mag': l_mag*1000, 
                'x': l_x, 'dist': l_end - l_x if "U" in l_type else 0.0
            })
            st.rerun()

# ตารางแสดง Load (ตรวจสอบและลบได้)
if st.session_state.loads:
    with st.expander("📝 Current Load List", expanded=True):
        load_df = pd.DataFrame(st.session_state.loads)
        st.dataframe(load_df, use_container_width=True)
        if st.button("🗑️ Clear All Loads"):
            st.session_state.loads = []
            st.rerun()

# --- 2. MATERIAL PROPERTIES ---
with st.sidebar:
    st.header("🧱 Material & Section")
    fc = st.number_input("f'c (MPa)", 28.0)
    fy = st.number_input("fy (MPa)", 400.0)
    b_m, h_m = st.number_input("Width b (m)", 0.3), st.number_input("Height h (m)", 0.5)
    cover, db_m = st.number_input("Cover (mm)", 30.0), st.selectbox("DB Main", [12, 16, 20, 25])

# --- 3. ANALYSIS & DESIGN ---
if st.button("🚀 RUN FULL DESIGN", type="primary", use_container_width=True):
    st.session_state.trigger_solve = True

if st.session_state.trigger_solve:
    supports = [{'id': i, 'type': 'Pin' if i==0 else 'Roller'} for i in range(n_spans+1)]
    # จำลองการคำนวณ (ใช้ Solver ของคุณ)
    sol = BeamSolver(spans, supports, st.session_state.loads, 2e11, b_m, h_m)
    df, reac, eq = sol.solve()

    if not df.empty:
        # SECTION: Longitudinal Drawing (รูปตัดตามยาว)
        st.header("🖼️ Longitudinal Detailing (รูปตัดตามยาว)")
        fig_long = go.Figure()
        cum_l = 0
        for l in spans:
            fig_long.add_shape(type="rect", x0=cum_l, y0=0, x1=cum_l+l, y1=h_m, line=dict(color="Black"))
            cum_l += l
        fig_long.update_layout(height=200, showlegend=False)
        st.plotly_chart(fig_long, use_container_width=True)
        

        # SECTION: Cross Section & Calculation
        st.header("📋 Detailed Design per Span")
        cum_dist = [0] + list(np.cumsum(spans))
        for i in range(n_spans):
            with st.container(border=True):
                st.subheader(f"Span {i+1} ({spans[i]}m)")
                s_df = df[(df['x'] >= cum_dist[i]) & (df['x'] <= cum_dist[i+1])]
                res = rc_design.design_span_expert(s_df['moment'].max()/1000, s_df['moment'].min()/1000, 
                                                   s_df['shear'].abs().max()/1000, b_m, h_m, fc, fy, 240, cover, db_m, 9)
                
                c1, c2 = st.columns([1, 1])
                with c1:
                    st.markdown("**Structural Report**")
                    st.latex(r"M_u^{(+)} = %.2f, M_u^{(-)} = %.2f \text{ kNm}" % (res['pos']['mu_val'], res['neg']['mu_val']))
                    st.latex(r"\epsilon_t = %.5f, V_u/\phi V_c = %.2f" % (res['pos']['et'], res['vu']/res['phi_vc']))
                
                with c2:
                    st.markdown("**Cross Section (รูปตัดขวาง)**")
                    n_t, n_b = int(res['neg']['n']), int(res['pos']['n'])
                    fig_cs = go.Figure()
                    fig_cs.add_shape(type="rect", x0=0, y0=0, x1=b_m, y1=h_m, fillcolor="lightgrey")
                    # วาดเหล็ก
                    for j in range(n_t): fig_cs.add_trace(go.Scatter(x=[(b_m/(n_t+1))*(j+1)], y=[h_m-0.05], mode='markers', marker=dict(color='red', size=10)))
                    for j in range(n_b): fig_cs.add_trace(go.Scatter(x=[(b_m/(n_b+1))*(j+1)], y=[0.05], mode='markers', marker=dict(color='blue', size=10)))
                    fig_cs.update_layout(width=250, height=250, showlegend=False, xaxis=dict(visible=False), yaxis=dict(visible=False))
                    st.plotly_chart(fig_cs)
