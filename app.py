import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from solver import BeamSolver
import rc_design

st.set_page_config(page_title="Professional Beam App", layout="wide")

# --- 1. SESSION STATE (กันค่าหาย) ---
if 'loads' not in st.session_state: st.session_state.loads = []
if 'results' not in st.session_state: st.session_state.results = None

st.title("🏗️ Professional Beam Studio (Final Fix)")

# --- 2. INPUTS & CONFIG ---
with st.sidebar:
    st.header("🧱 Material Config")
    fc = st.number_input("f'c (MPa)", 24.0, 50.0, 28.0)
    fy = st.number_input("fy (MPa)", 240.0, 500.0, 400.0)
    b_m = st.number_input("Width (m)", 0.15, 1.0, 0.3)
    h_m = st.number_input("Height (m)", 0.3, 2.0, 0.5)
    cover = st.number_input("Cover (mm)", 20, 75, 30)
    db_m = st.selectbox("Main Bar (mm)", [12, 16, 20, 25, 28])

with st.container(border=True):
    c1, c2 = st.columns([1, 2])
    with c1:
        st.subheader("📏 Geometry")
        n_spans = st.number_input("Num Spans", 1, 10, 2)
        spans = [st.number_input(f"L{i+1} (m)", 1.0, 20.0, 5.0, key=f"s{i}") for i in range(n_spans)]
    
    with c2:
        st.subheader("📥 Load Manager")
        lc1, lc2, lc3, lc4 = st.columns(4)
        l_idx = lc1.selectbox("Span Index", range(n_spans))
        l_type = lc2.selectbox("Type", ["Uniform", "Point", "Moment"])
        l_mag = lc3.number_input("Value (kN)", 10.0)
        l_x = lc4.number_input("Dist x (m)", 0.0, float(spans[l_idx]))
        
        if st.button("➕ Add Load", use_container_width=True):
            # Safe Logic: ปรับ x ไม่ให้ชนขอบเป๊ะๆ (0.01 margin) ช่วย Solver คำนวณได้ชัวร์กว่า
            safe_x = min(l_x, spans[l_idx] - 0.01) 
            safe_x = max(safe_x, 0.01)
            
            st.session_state.loads.append({
                'span': l_idx, 
                'type': l_type[0], 
                'mag': l_mag * 1000, 
                'x': safe_x,
                'dist': spans[l_idx] if l_type == "Uniform" else 0
            })
            st.rerun()

# Show Loads
if st.session_state.loads:
    st.write(f"**Current Loads ({len(st.session_state.loads)})**")
    st.table(pd.DataFrame(st.session_state.loads))
    if st.button("🗑️ Clear Loads"):
        st.session_state.loads = []
        st.session_state.results = None
        st.rerun()

# --- 3. ROBUST SOLVER EXECUTION ---
if st.button("🚀 RUN ANALYSIS", type="primary", use_container_width=True):
    if not st.session_state.loads:
        st.error("⚠️ Please add at least one load.")
    else:
        try:
            supports = [{'id': i, 'type': 'Pin' if i==0 else 'Roller'} for i in range(n_spans+1)]
            I_val = (b_m * h_m**3) / 12
            
            # เรียก Solver
            sol = BeamSolver(spans, supports, st.session_state.loads, 2e11, I_val)
            st.session_state.results = sol.solve()
            
        except Exception as e:
            st.error(f"Solver Error: {str(e)}")
            st.session_state.results = None

# --- 4. SAFE DISPLAY & DRAWINGS ---
if st.session_state.results:
    df, reac, eq = st.session_state.results
    
    # 🔴 CRITICAL FIX: ตรวจสอบ DataFrame ก่อนใช้งานเสมอ
    if df is None or df.empty or len(df.columns) == 0:
        st.error("❌ Solver returned empty data. Please adjust load positions or span lengths.")
    else:
        # หาชื่อคอลัมน์ที่ถูกต้อง
        x_col = 'x' if 'x' in df.columns else df.columns[0]
        
        st.success("Analysis Complete!")
        
        # --- A. LONGITUDINAL SECTION ---
        st.header("🖼️ Part 1: Longitudinal View")
        
        fig_long = go.Figure()
        cum_l = 0
        for i, l in enumerate(spans):
            # Beam Body
            fig_long.add_shape(type="rect", x0=cum_l, y0=0, x1=cum_l+l, y1=h_m, 
                               line=dict(color="Black", width=2), fillcolor="rgba(200,200,200,0.3)")
            # Rebars (Schematic)
            fig_long.add_trace(go.Scatter(x=[cum_l, cum_l+l], y=[h_m-0.05, h_m-0.05], mode="lines", line=dict(color="red", width=2)))
            fig_long.add_trace(go.Scatter(x=[cum_l, cum_l+l], y=[0.05, 0.05], mode="lines", line=dict(color="blue", width=2)))
            # Support
            fig_long.add_trace(go.Scatter(x=[cum_l], y=[-0.05], mode="markers", marker=dict(symbol="triangle-up", size=15, color="black")))
            cum_l += l
        # Last Support
        fig_long.add_trace(go.Scatter(x=[cum_l], y=[-0.05], mode="markers", marker=dict(symbol="triangle-up", size=15, color="black")))
        
        fig_long.update_layout(height=200, showlegend=False, xaxis=dict(title="Length (m)"), yaxis=dict(visible=False, scaleanchor="x"))
        st.plotly_chart(fig_long, use_container_width=True)

        # --- B. CROSS SECTIONS & CALCULATIONS ---
        st.header("📋 Part 2: Design Results")
        cum_dist = [0] + list(np.cumsum(spans))
        
        for i in range(n_spans):
            with st.container(border=True):
                # Filter Data safely
                mask = (df[x_col] >= cum_dist[i]) & (df[x_col] <= cum_dist[i+1])
                span_data = df[mask]
                
                if span_data.empty:
                    st.warning(f"Span {i+1}: No data points found.")
                    continue

                res = rc_design.design_span_expert(
                    span_data['moment'].max()/1000, span_data['moment'].min()/1000, 
                    span_data['shear'].abs().max()/1000, b_m, h_m, fc, fy, cover, db_m
                )

                c1, c2 = st.columns([1, 1])
                with c1:
                    st.subheader(f"Span {i+1}")
                    st.latex(r"M_u^+ = %.2f, M_u^- = %.2f \text{ kNm}" % (res['mu_pos'], res['mu_neg']))
                    st.write(f"**Top:** {res['neg']['n']} x DB{db_m} ({res['neg']['status']})")
                    st.write(f"**Bot:** {res['pos']['n']} x DB{db_m} ({res['pos']['status']})")
                    st.info(f"Shear: {res['vu']:.2f} kN / Cap: {res['phi_vc']:.2f} kN")

                with c2:
                    st.write("**Section Detail**")
                    
                    n_t, n_b = int(res['neg']['n']), int(res['pos']['n'])
                    fig_cs = go.Figure()
                    # Concrete
                    fig_cs.add_shape(type="rect", x0=0, y0=0, x1=b_m, y1=h_m, line=dict(color="black", width=3))
                    # Stirrup
                    fig_cs.add_shape(type="rect", x0=0.04, y0=0.04, x1=b_m-0.04, y1=h_m-0.04, line=dict(color="gray", dash="dot"))
                    # Top Bars
                    for k in range(n_t):
                        fig_cs.add_trace(go.Scatter(x=[(b_m/(n_t+1))*(k+1)], y=[h_m-0.05], mode="markers", marker=dict(color="red", size=12)))
                    # Bot Bars
                    for k in range(n_b):
                        fig_cs.add_trace(go.Scatter(x=[(b_m/(n_b+1))*(k+1)], y=[0.05], mode="markers", marker=dict(color="blue", size=12)))
                    
                    fig_cs.update_layout(width=200, height=250, showlegend=False, xaxis=dict(visible=False), yaxis=dict(visible=False), margin=dict(l=10,r=10,t=10,b=10))
                    st.plotly_chart(fig_cs)
