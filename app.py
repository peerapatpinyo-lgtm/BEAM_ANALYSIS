import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from solver import BeamSolver
import rc_design

st.set_page_config(page_title="Professional Structural Designer", layout="wide")

# Persistent State Management
if 'loads' not in st.session_state: st.session_state.loads = []
if 'solved_data' not in st.session_state: st.session_state.solved_data = None

st.title("🏗️ Beam Designer Pro: Engineering Analysis")

# --- 1. LOAD INPUT SYSTEM (STABLE) ---
with st.sidebar:
    st.header("🧱 Material & Geometry")
    fc = st.number_input("f'c (MPa)", 21.0, 50.0, 28.0)
    fy = st.number_input("fy (MPa)", 240.0, 500.0, 400.0)
    b_m = st.number_input("Width (m)", 0.1, 1.0, 0.3)
    h_m = st.number_input("Height (m)", 0.1, 2.0, 0.5)
    cover = st.number_input("Cover (mm)", 20, 75, 35)
    db_m = st.selectbox("Main DB Bar", [12, 16, 20, 25, 28])

st.header("1. Loading & Span Configuration")
col1, col2 = st.columns([1, 2])
with col1:
    n_spans = st.number_input("Number of Spans", 1, 10, 1)
    spans = [st.number_input(f"L{i+1} (m)", 0.5, 20.0, 5.0, key=f"span_{i}") for i in range(n_spans)]

with col2:
    st.subheader("Add Point/Uniform/Moment Load")
    lc1, lc2, lc3, lc4, lc5 = st.columns(5)
    l_idx = lc1.selectbox("Span", range(n_spans))
    l_type = lc2.selectbox("Type", ["Uniform (U)", "Point (P)", "Moment (M)"])
    l_mag = lc3.number_input("Value (kN)", 10.0)
    l_x1 = lc4.number_input("Start x (m)", 0.0, float(spans[l_idx]))
    l_x2 = lc5.number_input("End x (m)", float(spans[l_idx])) if "U" in l_type else l_x1

    if st.button("➕ Add Load", use_container_width=True):
        st.session_state.loads.append({
            'span': l_idx, 'type': l_type[0], 'mag': l_mag*1000, 
            'x': l_x1, 'dist': l_x2 - l_x1 if "U" in l_type else 0.0
        })
        st.rerun()

# Display and Delete Loads
if st.session_state.loads:
    with st.expander("Current Loads", expanded=True):
        for idx, ld in enumerate(st.session_state.loads):
            c_txt, c_btn = st.columns([4, 1])
            c_txt.write(f"Span {ld['span']+1}: {ld['type']} = {ld['mag']/1000} kN at x={ld['x']}m")
            if c_btn.button("🗑️", key=f"del_{idx}"):
                st.session_state.loads.pop(idx)
                st.rerun()

# --- 2. SOLVE & DESIGN ---
if st.button("🚀 RUN ANALYSIS & DESIGN", type="primary", use_container_width=True):
    supports = [{'id': i, 'type': 'Pin' if i==0 else 'Roller'} for i in range(n_spans+1)]
    sol = BeamSolver(spans, supports, st.session_state.loads, 2e11, b_m, h_m)
    st.session_state.solved_data = sol.solve()

# --- 3. RESULTS & ENGINEERING DRAWINGS ---
if st.session_state.solved_data:
    df, reac, eq = st.session_state.solved_data
    
    # Check if 'x' exists to prevent KeyError
    x_col = 'x' if 'x' in df.columns else df.columns[0] 
    
    st.header("2. Engineering Detailing")
    
    # DRAWING: Longitudinal Section
    st.subheader("📏 Longitudinal Detailing (ภาพตัดแนวยาว)")
    
    fig_long = go.Figure()
    cum_l = 0
    for i, l in enumerate(spans):
        # Beam outline
        fig_long.add_shape(type="rect", x0=cum_l, y0=0, x1=cum_l+l, y1=h_m, line=dict(color="Black", width=2))
        # Support
        fig_long.add_trace(go.Scatter(x=[cum_l], y=[-0.05], mode="markers", marker=dict(symbol="triangle-up", size=15, color="Black")))
        cum_l += l
    fig_long.add_trace(go.Scatter(x=[cum_l], y=[-0.05], mode="markers", marker=dict(symbol="triangle-up", size=15, color="Black")))
    fig_long.update_layout(height=250, showlegend=False, xaxis=dict(title="Length (m)"), yaxis=dict(visible=False, scaleanchor="x"))
    st.plotly_chart(fig_long, use_container_width=True)

    # DRAWING: Cross Section per Span
    st.subheader("📋 Cross-Section Design (ภาพตัดขวางราย Span)")
    cum_dist = [0] + list(np.cumsum(spans))
    
    for i in range(n_spans):
        # Filtering data with safety check
        mask = (df[x_col] >= cum_dist[i]) & (df[x_col] <= cum_dist[i+1])
        span_df = df[mask]
        
        if not span_df.empty:
            res = rc_design.design_span_expert(
                span_df['moment'].max()/1000, span_df['moment'].min()/1000, 
                span_df['shear'].abs().max()/1000, b_m, h_m, fc, fy, cover, db_m
            )
            
            with st.container(border=True):
                c_data, c_img = st.columns([1, 1])
                with c_data:
                    st.write(f"**SPAN {i+1} Calculation**")
                    st.latex(r"M_u^{(+)} = %.2f, \quad M_u^{(-)} = %.2f \text{ kNm}" % (res['mu_pos'], res['mu_neg']))
                    st.write(f"**Top:** {res['neg']['n']} x DB{db_m} (Status: {res['neg']['status']})")
                    st.write(f"**Bottom:** {res['pos']['n']} x DB{db_m} (Status: {res['pos']['status']})")
                    st.write(f"**Shear:** $V_u = %.2f$ kN (Concrete Capacity $\phi V_c = %.2f$ kN)" % (res['vu'], res['phi_vc']))
                
                with c_img:
                    
                    fig_cs = go.Figure()
                    # Concrete Section
                    fig_cs.add_shape(type="rect", x0=0, y0=0, x1=b_m, y1=h_m, fillcolor="rgba(0,0,0,0.1)", line=dict(color="Black"))
                    # Top Bars
                    for j in range(int(res['neg']['n'])):
                        fig_cs.add_trace(go.Scatter(x=[(b_m/(res['neg']['n']+1))*(j+1)], y=[h_m - (cover/1000)], mode="markers", marker=dict(color="Red", size=10)))
                    # Bottom Bars
                    for j in range(int(res['pos']['n'])):
                        fig_cs.add_trace(go.Scatter(x=[(b_m/(res['pos']['n']+1))*(j+1)], y=[cover/1000], mode="markers", marker=dict(color="Blue", size=10)))
                    
                    fig_cs.update_layout(width=250, height=250, showlegend=False, xaxis=dict(visible=False), yaxis=dict(visible=False))
                    st.plotly_chart(fig_cs)
