import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from solver import BeamSolver
import rc_design

st.set_page_config(page_title="Professional Beam Designer", layout="wide")

if 'loads' not in st.session_state: st.session_state.loads = []
if 'analysis_results' not in st.session_state: st.session_state.analysis_results = None

st.title("🏗️ Professional Continuous Beam Designer")

# --- DEBUG & SAMPLE DATA ---
with st.expander("🛠️ System Tools & Debug"):
    if st.button("🔄 Reset App"):
        st.session_state.loads = []
        st.session_state.analysis_results = None
        st.rerun()
    if st.button("📥 Load Sample 2-Span Case"):
        st.session_state.loads = [
            {'span': 0, 'type': 'U', 'mag': 20000, 'x': 0, 'dist': 5.0},
            {'span': 1, 'type': 'P', 'mag': 50000, 'x': 2.5, 'dist': 0}
        ]
        st.rerun()

# --- 1. INPUTS ---
with st.sidebar:
    st.header("🧱 Section & Materials")
    fc = st.number_input("f'c (MPa)", 28.0)
    fy = st.number_input("fy (MPa)", 400.0)
    b_m = st.number_input("Width (m)", 0.3)
    h_m = st.number_input("Height (m)", 0.5)
    cover = st.number_input("Cover (mm)", 35.0)
    db_m = st.selectbox("Main DB", [16, 20, 25])

col_geom, col_load = st.columns([1, 2])
with col_geom:
    st.subheader("📏 Spans")
    n_spans = st.number_input("Count", 1, 5, 2)
    spans = [st.number_input("L %d (m)" % (i+1), 1.0, 15.0, 5.0, key="s_%d"%i) for i in range(n_spans)]

with col_load:
    st.subheader("📥 Load Manager")
    c1, c2, c3, c4 = st.columns(4)
    l_idx = c1.selectbox("Span", range(n_spans))
    l_type = c2.selectbox("Type", ["Uniform", "Point", "Moment"])
    l_mag = c3.number_input("Mag (kN)", 10.0)
    l_x = c4.number_input("Pos (m)", 0.0)
    if st.button("➕ Add Load", use_container_width=True):
        st.session_state.loads.append({
            'span': l_idx, 'type': l_type[0], 'mag': l_mag*1000, 'x': l_x, 'dist': spans[l_idx] if l_type[0]=='U' else 0
        })
        st.rerun()

# --- 2. SOLVE ---
if st.button("🚀 EXECUTE ANALYSIS", type="primary", use_container_width=True):
    if not st.session_state.loads:
        st.error("Please add at least one load!")
    else:
        supports = [{'id': i, 'type': 'Pin' if i==0 else 'Roller'} for i in range(n_spans+1)]
        # I = (b*h^3)/12
        I_val = (b_m * h_m**3) / 12
        sol = BeamSolver(spans, supports, st.session_state.loads, 2e11, I_val) 
        st.session_state.analysis_results = sol.solve()

# --- 3. OUTPUTS & DRAWINGS ---
if st.session_state.analysis_results:
    df, reac, eq = st.session_state.analysis_results
    
    if df is not None and not df.empty:
        # Drawing Longitudinal Section
        st.header("🖼️ Longitudinal Detailing")
        
        fig_l = go.Figure()
        cum_l = 0
        for l in spans:
            fig_l.add_shape(type="rect", x0=cum_l, y0=0, x1=cum_l+l, y1=h_m, line=dict(color="Black"))
            fig_l.add_trace(go.Scatter(x=[cum_l], y=[-0.05], mode="markers", marker=dict(symbol="triangle-up", size=15, color="black")))
            cum_l += l
        fig_l.add_trace(go.Scatter(x=[cum_l], y=[-0.05], mode="markers", marker=dict(symbol="triangle-up", size=15, color="black")))
        fig_l.update_layout(height=180, showlegend=False, margin=dict(t=10,b=10), yaxis=dict(visible=False))
        st.plotly_chart(fig_l, use_container_width=True)

        # Drawing Span Details
        st.header("📋 Detailed Design Sheets")
        cum_dist = [0] + list(np.cumsum(spans))
        for i in range(n_spans):
            with st.container(border=True):
                # Using columns for result and cross section
                res_col, img_col = st.columns([1, 1])
                
                # Get max/min moments for this span
                s_df = df[(df['x'] >= cum_dist[i]) & (df['x'] <= cum_dist[i+1])]
                res = rc_design.design_span_expert(s_df['moment'].max()/1000, s_df['moment'].min()/1000, 
                                                   s_df['shear'].abs().max()/1000, b_m, h_m, fc, fy, cover, db_m)
                
                with res_col:
                    st.subheader("Span %d Results" % (i+1))
                    st.write("**Moments:** Max = %.1f, Min = %.1f kNm" % (res['mu_pos'], res['mu_neg']))
                    st.write("**Reinforcement:**")
                    st.write("- Top: %d x DB%d (%s)" % (res['neg']['n'], db_m, res['neg']['status']))
                    st.write("- Bottom: %d x DB%d (%s)" % (res['pos']['n'], db_m, res['pos']['status']))
                    st.write("**Shear:** $V_u$ = %.1f kN vs $\phi V_c$ = %.1f kN" % (res['vu'], res['phi_vc']))

                with img_col:
                    
                    n_t, n_b = int(res['neg']['n']), int(res['pos']['n'])
                    fig_cs = go.Figure()
                    fig_cs.add_shape(type="rect", x0=0, y0=0, x1=b_m, y1=h_m, fillcolor="rgba(0,0,0,0.05)")
                    # Top Bars
                    for j in range(n_t):
                        fig_cs.add_trace(go.Scatter(x=[(b_m/(n_t+1))*(j+1)], y=[h_m-0.05], mode="markers", marker=dict(color="Red", size=12)))
                    # Bot Bars
                    for j in range(n_b):
                        fig_cs.add_trace(go.Scatter(x=[(b_m/(n_b+1))*(j+1)], y=[0.05], mode="markers", marker=dict(color="Blue", size=12)))
                    fig_cs.update_layout(width=250, height=250, showlegend=False, xaxis=dict(visible=False), yaxis=dict(visible=False))
                    st.plotly_chart(fig_cs)
    else:
        st.error("The solver returned an empty dataset. Please check if your loads are within the span lengths.")
