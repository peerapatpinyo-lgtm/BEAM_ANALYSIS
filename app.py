import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from solver import BeamSolver
import rc_design

st.set_page_config(page_title="Professional Beam Design Studio", layout="wide")

# Persistent State Management
if 'loads' not in st.session_state: st.session_state.loads = []
if 'analysis_results' not in st.session_state: st.session_state.analysis_results = None

st.title("🏗️ Beam Designer Pro: Continuous Beam Suite")

# --- 1. CONFIGURATION & LOAD MANAGEMENT ---
with st.container(border=True):
    c_geom, c_load = st.columns([1, 2])
    
    with c_geom:
        st.subheader("📏 Geometry")
        n_spans = st.number_input("Number of Spans", 1, 10, 2)
        spans = [st.number_input("Span %d Length (m)" % (i+1), 0.5, 20.0, 5.0, key="s_%d"%i) for i in range(n_spans)]
    
    with c_load:
        st.subheader("📥 Load Input")
        lc1, lc2, lc3, lc4 = st.columns(4)
        l_idx = lc1.selectbox("On Span", range(n_spans))
        l_type = lc2.selectbox("Type", ["Uniform (U)", "Point (P)", "Moment (M)"])
        l_mag = lc3.number_input("Magnitude (kN/kNm)", 10.0)
        # Sanitizing x input to stay within span limits
        l_x = lc4.number_input("Start x (m)", 0.0, float(spans[l_idx]))

        if st.button("➕ Add Load Case", use_container_width=True):
            # Final sanitize check before saving
            safe_x = min(l_x, spans[l_idx] - 0.001)
            st.session_state.loads.append({
                'span': l_idx, 'type': l_type[0], 'mag': l_mag*1000, 
                'x': safe_x, 'dist': spans[l_idx] if l_type[0]=='U' else 0.0
            })
            st.session_state.analysis_results = None
            st.rerun()

# --- LIVE LOAD MONITOR ---
if st.session_state.loads:
    with st.expander("📝 Review Current Loading Plan", expanded=True):
        load_df = pd.DataFrame(st.session_state.loads)
        st.table(load_df)
        if st.button("🗑️ Clear All Loads"):
            st.session_state.loads = []
            st.session_state.analysis_results = None
            st.rerun()

# --- 2. SIDEBAR MATERIALS ---
with st.sidebar:
    st.header("🧱 Section & Materials")
    fc = st.number_input("f'c (Concrete MPa)", 28.0)
    fy = st.number_input("fy (Main Bar MPa)", 400.0)
    b_m = st.number_input("Beam Width b (m)", 0.3)
    h_m = st.number_input("Beam Height h (m)", 0.5)
    cover = st.number_input("Concrete Cover (mm)", 35)
    db_m = st.selectbox("Main Rebar Size (DB)", [12, 16, 20, 25, 28])

# --- 3. EXECUTE SOLVER ---
if st.button("🚀 EXECUTE FULL ANALYSIS", type="primary", use_container_width=True):
    if not st.session_state.loads:
        st.warning("Please add at least one load case.")
    else:
        supports = [{'id': i, 'type': 'Pin' if i==0 else 'Roller'} for i in range(n_spans+1)]
        I_val = (b_m * h_m**3) / 12
        # Initializing solver with persistent data
        sol = BeamSolver(spans, supports, st.session_state.loads, 2e11, I_val)
        st.session_state.analysis_results = sol.solve()

# --- 4. PROFESSIONAL CONSTRUCTION DRAWINGS ---
if st.session_state.analysis_results:
    df, reac, eq = st.session_state.analysis_results
    
    if df is not None and not df.empty:
        st.header("🛠️ Part I: Longitudinal Detailing")
        
        
        # Longitudinal View using Plotly
        fig_long = go.Figure()
        cum_l = 0
        for i, l in enumerate(spans):
            # Draw Beam Concrete
            fig_long.add_shape(type="rect", x0=cum_l, y0=0, x1=cum_l+l, y1=h_m, line=dict(color="Black", width=3), fillcolor="rgba(200,200,200,0.2)")
            # Draw Main Top/Bottom Reinforcement (Schematic)
            fig_long.add_trace(go.Scatter(x=[cum_l, cum_l+l], y=[h_m-0.05, h_m-0.05], mode="lines", line=dict(color="red", width=2)))
            fig_long.add_trace(go.Scatter(x=[cum_l, cum_l+l], y=[0.05, 0.05], mode="lines", line=dict(color="blue", width=2)))
            # Draw Supports
            fig_long.add_trace(go.Scatter(x=[cum_l], y=[-0.05], mode="markers", marker=dict(symbol="triangle-up", size=20, color="Black")))
            cum_l += l
        fig_long.add_trace(go.Scatter(x=[cum_l], y=[-0.05], mode="markers", marker=dict(symbol="triangle-up", size=20, color="Black")))
        
        fig_long.update_layout(height=300, showlegend=False, xaxis=dict(title="Length along beam (m)"), yaxis=dict(visible=False, scaleanchor="x"))
        st.plotly_chart(fig_long, use_container_width=True)

        st.header("📋 Part II: Engineering Calculation & Cross-Sections")
        cum_dist = [0] + list(np.cumsum(spans))
        
        for i in range(n_spans):
            with st.container(border=True):
                # Analyze this specific span
                span_data = df[(df['x'] >= cum_dist[i]) & (df['x'] <= cum_dist[i+1])]
                res = rc_design.design_span_expert(span_data['moment'].max()/1000, span_data['moment'].min()/1000, 
                                                   span_data['shear'].abs().max()/1000, b_m, h_m, fc, fy, cover, db_m)
                
                col_calc, col_drawing = st.columns([1, 1])
                
                with col_calc:
                    st.subheader("Span %d Report" % (i+1))
                    st.write("**Design Moments:**")
                    st.latex(r"M_u^{(+)} = %.2f \text{ kNm}, \quad M_u^{(-)} = %.2f \text{ kNm}" % (res['mu_pos'], res['mu_neg']))
                    st.write("**Reinforcement Requirement:**")
                    st.success("Top: %d x DB%d Bars" % (res['neg']['n'], db_m))
                    st.success("Bottom: %d x DB%d Bars" % (res['pos']['n'], db_m))
                    st.write("**Shear Check:** $V_u = %.1f$ kN vs $\phi V_c = %.1f$ kN" % (res['vu'], res['phi_vc']))
                    if res['vu'] > res['phi_vc']:
                        st.warning("⚠️ Stirrups required for shear reinforcement.")

                with col_drawing:
                    st.write("**Cross-Section Drawing**")
                    
                    fig_cs = go.Figure()
                    # Section box
                    fig_cs.add_shape(type="rect", x0=0, y0=0, x1=b_m, y1=h_m, line=dict(color="Black", width=4), fillcolor="rgba(100,100,100,0.1)")
                    # Draw Stirrup
                    fig_cs.add_shape(type="rect", x0=0.03, y0=0.03, x1=b_m-0.03, y1=h_m-0.03, line=dict(color="grey", width=2))
                    # Draw Bars
                    n_t, n_b = int(res['neg']['n']), int(res['pos']['n'])
                    for j in range(n_t):
                        fig_cs.add_trace(go.Scatter(x=[(b_m/(n_t+1))*(j+1)], y=[h_m-0.05], mode="markers", marker=dict(color="Red", size=15)))
                    for j in range(n_b):
                        fig_cs.add_trace(go.Scatter(x=[(b_m/(n_b+1))*(j+1)], y=[0.05], mode="markers", marker=dict(color="Blue", size=15)))
                    
                    fig_cs.update_layout(width=300, height=300, showlegend=False, xaxis=dict(visible=False), yaxis=dict(visible=False), margin=dict(l=10,r=10,t=10,b=10))
                    st.plotly_chart(fig_cs)
    else:
        st.error("❌ The solver returned no data. Check that your load locations (x) are within the span lengths.")
