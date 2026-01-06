import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from solver import BeamSolver
import rc_design

st.set_page_config(page_title="Professional Beam Suite", layout="wide")

# ป้องกันข้อมูลหายด้วย Session State
if 'loads' not in st.session_state: st.session_state.loads = []
if 'analysis_done' not in st.session_state: st.session_state.analysis_done = False
if 'results' not in st.session_state: st.session_state.results = None

st.title("🏗️ Professional Beam Studio (Stable Version)")

# --- 1. GEOMETRY & STABLE LOADING ---
with st.container(border=True):
    col_g, col_l = st.columns([1, 2])
    with col_g:
        st.subheader("📏 Geometry")
        n_spans = st.number_input("Spans", 1, 10, 1)
        spans = [st.number_input("L %d (m)" % (i+1), 0.1, 20.0, 5.0, key="L_%d" % i) for i in range(n_spans)]
    
    with col_l:
        st.subheader("📥 Add Loads")
        c1, c2, c3, c4 = st.columns(4)
        l_span = c1.selectbox("Span", range(n_spans))
        l_type = c2.selectbox("Type", ["Point", "Uniform", "Moment"])
        l_mag = c3.number_input("Value", 10.0)
        l_x = c4.number_input("Pos x (m)", 0.0, float(spans[l_span]))
        
        if st.button("➕ Add Load"):
            st.session_state.loads.append({
                'span': l_span, 'type': l_type[0], 'mag': l_mag*1000, 'x': l_x, 'dist': 0.0
            })
            st.session_state.analysis_done = False # Reset เมื่อมีการแก้ข้อมูล
            st.rerun()

# ตารางจัดการ Load
if st.session_state.loads:
    with st.expander("📝 Load List"):
        for i, ld in enumerate(st.session_state.loads):
            cols = st.columns([4, 1])
            cols[0].write("Load %d: Span %d | %s | %.2f kN" % (i+1, ld['span']+1, ld['type'], ld['mag']/1000))
            if cols[1].button("🗑️", key="del_%d" % i):
                st.session_state.loads.pop(i)
                st.rerun()

# --- 2. SIDEBAR MATERIALS ---
with st.sidebar:
    st.header("🧱 Section & Material")
    fc = st.number_input("f'c (MPa)", 28.0)
    fy = st.number_input("fy (MPa)", 400.0)
    b_m, h_m = st.number_input("Width b (m)", 0.3), st.number_input("Height h (m)", 0.5)
    cover = st.number_input("Cover (mm)", 30.0)
    db_m = st.selectbox("DB Size", [12, 16, 20, 25])

# --- 3. EXECUTION ---
if st.button("🚀 ANALYZE & DESIGN", type="primary", use_container_width=True):
    supports = [{'id': i, 'type': 'Pin' if i==0 else 'Roller'} for i in range(n_spans+1)]
    sol = BeamSolver(spans, supports, st.session_state.loads, 2e11, b_m, h_m)
    st.session_state.results = sol.solve()
    st.session_state.analysis_done = True

# --- 4. DISPLAY RESULTS (STABLE) ---
if st.session_state.analysis_done and st.session_state.results:
    df, reac, eq = st.session_state.results
    
    # รูปตัดตามยาว (Longitudinal Section)
    st.header("🖼️ PART I: Longitudinal Detailing")
    
    fig_long = go.Figure()
    cum_l = 0
    for l in spans:
        fig_long.add_shape(type="rect", x0=cum_l, y0=0, x1=cum_l+l, y1=h_m, line=dict(color="Black", width=2))
        cum_l += l
    fig_long.update_layout(height=150, margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(title="Length (m)"), yaxis=dict(visible=False))
    st.plotly_chart(fig_long, use_container_width=True)

    # รายละเอียดแต่ละ Span
    st.header("📋 PART II: Engineering Design Sheets")
    cum_dist = [0] + list(np.cumsum(spans))
    for i in range(n_spans):
        with st.container(border=True):
            st.subheader("SPAN %d (%s m)" % (i+1, spans[i]))
            s_df = df[(df['x'] >= cum_dist[i]) & (df['x'] <= cum_dist[i+1])]
            res = rc_design.design_span_expert(s_df['moment'].max()/1000, s_df['moment'].min()/1000, 
                                               s_df['shear'].abs().max()/1000, b_m, h_m, fc, fy, 240, cover, db_m, 9)
            
            col_txt, col_img = st.columns([1, 1])
            with col_txt:
                st.latex(r"M_u^{(+)} = %.2f \text{ kNm}" % res['pos']['mu_val'])
                st.latex(r"M_u^{(-)} = %.2f \text{ kNm}" % res['neg']['mu_val'])
                st.write("**Shear:** $V_u = %.2f \text{ kN}$ vs $\phi V_c = %.2f \text{ kN}$" % (res['vu'], res['phi_vc']))
                st.info("Status: %s" % res['pos']['status'])
            
            with col_img:
                st.write("**Cross Section (รูปตัดขวาง)**")
                
                n_t, n_b = int(res['neg']['n']), int(res['pos']['n'])
                fig_cs = go.Figure()
                fig_cs.add_shape(type="rect", x0=0, y0=0, x1=b_m, y1=h_m, fillcolor="rgba(0,0,0,0.1)", line=dict(color="black"))
                # Draw bars
                for j in range(n_t): fig_cs.add_trace(go.Scatter(x=[(b_m/(n_t+1))*(j+1)], y=[h_m-0.05], mode='markers', marker=dict(color='Red', size=12)))
                for j in range(n_b): fig_cs.add_trace(go.Scatter(x=[(b_m/(n_b+1))*(j+1)], y=[0.05], mode='markers', marker=dict(color='Blue', size=12)))
                fig_cs.update_layout(width=220, height=220, showlegend=False, xaxis=dict(visible=False), yaxis=dict(visible=False), margin=dict(l=5,r=5,t=5,b=5))
                st.plotly_chart(fig_cs)
