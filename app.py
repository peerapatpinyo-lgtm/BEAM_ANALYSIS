import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from solver import BeamSolver
import rc_design

st.set_page_config(page_title="Professional Beam Suite", layout="wide")

if 'loads' not in st.session_state: st.session_state.loads = []
if 'analysis_results' not in st.session_state: st.session_state.analysis_results = None

st.title("🏗️ Professional Beam Studio")

# --- 1. GEOMETRY & STABLE LOADING ---
with st.container(border=True):
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("📏 Spans")
        n_spans = st.number_input("Count", 1, 10, 1)
        spans = [st.number_input(f"L{i+1} (m)", 0.5, 20.0, 5.0, key=f"s_{i}") for i in range(n_spans)]
    with col2:
        st.subheader("📥 Load Manager")
        c1, c2, c3, c4 = st.columns(4)
        l_idx = c1.selectbox("Span", range(n_spans))
        l_type = c2.selectbox("Type", ["Uniform", "Point", "Moment"])
        l_mag = c3.number_input("Value", 10.0)
        l_x = c4.number_input("Pos x", 0.0, float(spans[l_idx]))
        if st.button("➕ Add Load"):
            st.session_state.loads.append({'span': l_idx, 'type': l_type[0], 'mag': l_mag*1000, 'x': l_x})
            st.session_state.analysis_results = None
            st.rerun()

if st.session_state.loads:
    with st.expander("Current Loads"):
        for i, ld in enumerate(st.session_state.loads):
            st.write(f"{i+1}. Span {ld['span']+1}: {ld['type']} {ld['mag']/1000}kN at {ld['x']}m")

# --- 2. EXECUTION & DATA SAFETY ---
with st.sidebar:
    st.header("🧱 Section Properties")
    fc = st.number_input("f'c (MPa)", 28.0)
    fy = st.number_input("fy (MPa)", 400.0)
    b_m, h_m = st.number_input("b (m)", 0.3), st.number_input("h (m)", 0.5)
    cover, db_m = st.number_input("Cover (mm)", 35.0), st.selectbox("DB Main", [16, 20, 25])

if st.button("🚀 ANALYZE & DESIGN", type="primary", use_container_width=True):
    supports = [{'id': i, 'type': 'Pin' if i==0 else 'Roller'} for i in range(n_spans+1)]
    sol = BeamSolver(spans, supports, st.session_state.loads, 2e11, b_m, h_m)
    st.session_state.analysis_results = sol.solve()

# --- 3. DRAWING PART (PRO LEVEL) ---
if st.session_state.analysis_results:
    df, reac, eq = st.session_state.analysis_results
    
    # 🚩 Safety Check for DataFrame
    if df is not None and not df.empty and len(df.columns) > 0:
        x_col = 'x' if 'x' in df.columns else df.columns[0]
        
        st.header("🖼️ Construction Detailing")
        
        # ภาพตัดแนวยาว (Longitudinal)
        st.subheader("1. Longitudinal Section")
        
        fig_long = go.Figure()
        cum_l = 0
        for l in spans:
            fig_long.add_shape(type="rect", x0=cum_l, y0=0, x1=cum_l+l, y1=h_m, line=dict(color="Black", width=2))
            fig_long.add_trace(go.Scatter(x=[cum_l], y=[-0.05], mode="markers", marker=dict(symbol="triangle-up", size=15, color="Black")))
            cum_l += l
        fig_long.add_trace(go.Scatter(x=[cum_l], y=[-0.05], mode="markers", marker=dict(symbol="triangle-up", size=15, color="Black")))
        fig_long.update_layout(height=200, showlegend=False, xaxis=dict(title="Length (m)"), yaxis=dict(visible=False, scaleanchor="x"))
        st.plotly_chart(fig_long, use_container_width=True)

        # ภาพตัดขวาง (Cross Section)
        st.subheader("2. Span Detail & Cross-Sections")
        cum_dist = [0] + list(np.cumsum(spans))
        for i in range(n_spans):
            mask = (df[x_col] >= cum_dist[i]) & (df[x_col] <= cum_dist[i+1])
            s_df = df[mask]
            
            if not s_df.empty:
                res = rc_design.design_span_expert(s_df['moment'].max()/1000, s_df['moment'].min()/1000, 
                                                   s_df['shear'].abs().max()/1000, b_m, h_m, fc, fy, cover, db_m)
                
                with st.container(border=True):
                    c1, c2 = st.columns([1, 1])
                    with c1:
                        st.write(f"**SPAN {i+1} REPORT**")
                        st.latex(r"M_u^{(+)} = %.1f, M_u^{(-)} = %.1f \text{ kNm}" % (res['mu_pos'], res['mu_neg']))
                        st.write(f"Top Rebar: {res['neg']['n']} x DB{db_m}")
                        st.write(f"Bottom Rebar: {res['pos']['n']} x DB{db_m}")
                        st.info(f"Shear Check: Vu({res['vu']:.1f}) {'<' if res['vu'] < res['phi_vc'] else '>'} PhiVc({res['phi_vc']:.1f})")

                    with c2:
                        
                        fig_cs = go.Figure()
                        fig_cs.add_shape(type="rect", x0=0, y0=0, x1=b_m, y1=h_m, fillcolor="rgba(0,0,0,0.05)", line=dict(color="Black"))
                        # Top Bars (Red)
                        for j in range(int(res['neg']['n'])):
                            fig_cs.add_trace(go.Scatter(x=[(b_m/(res['neg']['n']+1))*(j+1)], y=[h_m-0.05], mode="markers", marker=dict(color="Red", size=12)))
                        # Bottom Bars (Blue)
                        for j in range(int(res['pos']['n'])):
                            fig_cs.add_trace(go.Scatter(x=[(b_m/(res['pos']['n']+1))*(j+1)], y=[0.05], mode="markers", marker=dict(color="Blue", size=12)))
                        fig_cs.update_layout(width=250, height=250, showlegend=False, xaxis=dict(visible=False), yaxis=dict(visible=False))
                        st.plotly_chart(fig_cs)
    else:
        st.warning("No data found. Please check your loads and run analysis again.")
