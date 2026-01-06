import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from solver import BeamSolver
import rc_design

st.set_page_config(page_title="Professional Beam Suite", layout="wide")

# ระบบจัดการ Session State ให้เสถียร 100%
if 'loads' not in st.session_state: st.session_state.loads = []
if 'results' not in st.session_state: st.session_state.results = None

st.title("🏗️ Professional Beam Studio (Stable Build)")

# --- 1. GEOMETRY & STABLE LOADING ---
with st.container(border=True):
    col_input, col_table = st.columns([1.5, 2])
    
    with col_input:
        st.subheader("📏 Geometry & Input")
        n_spans = st.number_input("Number of Spans", 1, 10, 2)
        spans = [st.number_input(f"L{i+1} (m)", 0.5, 20.0, 5.0, key=f"L_{i}") for i in range(n_spans)]
        
        st.divider()
        st.write("**Add New Load**")
        c1, c2, c3, c4 = st.columns(4)
        l_span = c1.selectbox("Span", range(n_spans))
        l_type = c2.selectbox("Type", ["Uniform", "Point", "Moment"])
        l_mag = c3.number_input("Value", 10.0)
        l_x = c4.number_input("Start x", 0.0, float(spans[l_span]))

        if st.button("➕ Add Load", use_container_width=True):
            # ป้องกันปัญหา x เกินขอบเขต
            safe_x = min(l_x, spans[l_span] - 0.01)
            st.session_state.loads.append({
                'span_index': l_span, 'type': l_type[0], 'mag': l_mag * 1000, 
                'x': safe_x, 'dist': spans[l_span] if l_type == "Uniform" else 0.0
            })
            st.rerun()

    with col_table:
        st.subheader("📝 Load Management")
        if st.session_state.loads:
            load_df = pd.DataFrame(st.session_state.loads)
            st.dataframe(load_df, use_container_width=True)
            if st.button("🗑️ Clear All Loads"):
                st.session_state.loads = []
                st.session_state.results = None
                st.rerun()
        else:
            st.info("No loads added yet.")

# --- 2. MATERIALS SIDEBAR ---
with st.sidebar:
    st.header("🧱 Section Properties")
    fc = st.number_input("Concrete f'c (MPa)", 28.0)
    fy = st.number_input("Rebar fy (MPa)", 400.0)
    b_m, h_m = st.number_input("Width (m)", 0.3), st.number_input("Height (m)", 0.5)
    cover, db_m = st.number_input("Cover (mm)", 35), st.selectbox("Main DB", [16, 20, 25])

# --- 3. EXECUTION ---
if st.button("🚀 EXECUTE ANALYSIS", type="primary", use_container_width=True):
    if not st.session_state.loads:
        st.error("Please add at least one load.")
    else:
        supports = [{'id': i, 'type': 'Pin' if i==0 else 'Roller'} for i in range(n_spans+1)]
        sol = BeamSolver(spans, supports, st.session_state.loads, 2e11, (b_m * h_m**3)/12)
        st.session_state.results = sol.solve()

# --- 4. DESIGN & BLUEPRINT (IMPROVED) ---
if st.session_state.results:
    df, reac, eq = st.session_state.results
    
    if df is not None and not df.empty:
        st.header("📊 Part I: Force Diagrams")
        

[Image of shear force and bending moment diagrams for a continuous beam]

        # (ส่วนวาดกราฟ SFD/BMD ของคุณ)

        st.header("🖼️ Part II: Longitudinal Reinforcement (รูปตัดแนวยาว)")
        
        fig_long = go.Figure()
        cum_l = 0
        for l in spans:
            # Concrete Outline
            fig_long.add_shape(type="rect", x0=cum_l, y0=0, x1=cum_l+l, y1=h_m, fillcolor="rgba(128,128,128,0.1)", line=dict(color="Black"))
            # Symbolic Steel Bars
            fig_long.add_trace(go.Scatter(x=[cum_l, cum_l+l], y=[h_m-0.05, h_m-0.05], mode="lines", line=dict(color="red", width=2)))
            fig_long.add_trace(go.Scatter(x=[cum_l, cum_l+l], y=[0.05, 0.05], mode="lines", line=dict(color="blue", width=2)))
            cum_l += l
        fig_long.update_layout(height=200, showlegend=False, xaxis=dict(title="Length (m)"), yaxis=dict(visible=False, scaleanchor="x"))
        st.plotly_chart(fig_long, use_container_width=True)

        st.header("📋 Part III: Span Design & Cross-Sections")
        cum_dist = [0] + list(np.cumsum(spans))
        for i in range(n_spans):
            with st.container(border=True):
                # กรองข้อมูลเฉพาะ Span นี้
                s_df = df[(df['x'] >= cum_dist[i]) & (df['x'] <= cum_dist[i+1])]
                res = rc_design.design_span_expert(s_df['moment'].max()/1000, s_df['moment'].min()/1000, 
                                                   s_df['shear'].abs().max()/1000, b_m, h_m, fc, fy, cover, db_m)
                
                c1, c2 = st.columns([1, 1])
                with c1:
                    st.subheader(f"Span {i+1} Calculation")
                    st.latex(r"M_u^{(+)} = %.2f \text{ kNm}" % res['mu_pos'])
                    st.latex(r"M_u^{(-)} = %.2f \text{ kNm}" % res['mu_neg'])
                    st.write(f"**Rebar:** Top {res['neg']['n']}xDB{db_m} | Bot {res['pos']['n']}xDB{db_m}")
                    st.write(f"**Shear:** Vu = {res['vu']:.1f} kN vs PhiVc = {res['phi_vc']:.1f} kN")

                with c2:
                    st.write("**Cross-Section (รูปตัดขวาง)**")
                    
                    n_t, n_b = int(res['neg']['n']), int(res['pos']['n'])
                    fig_cs = go.Figure()
                    fig_cs.add_shape(type="rect", x0=0, y0=0, x1=b_m, y1=h_m, fillcolor="rgba(0,0,0,0.05)", line=dict(width=3))
                    # Draw Rebars
                    for j in range(n_t):
                        fig_cs.add_trace(go.Scatter(x=[(b_m/(n_t+1))*(j+1)], y=[h_m-0.05], mode="markers", marker=dict(color="Red", size=12)))
                    for j in range(n_b):
                        fig_cs.add_trace(go.Scatter(x=[(b_m/(n_b+1))*(j+1)], y=[0.05], mode="markers", marker=dict(color="Blue", size=12)))
                    fig_cs.update_layout(width=280, height=280, showlegend=False, xaxis=dict(visible=False), yaxis=dict(visible=False))
                    st.plotly_chart(fig_cs)
    else:
        st.error("❌ Solver returned empty results. Check Span length and Load positions.")
