import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from solver import BeamSolver
import rc_design

st.set_page_config(page_title="Professional Beam Design", layout="wide")

# --- 1. Session State Setup ---
if 'loads' not in st.session_state: st.session_state.loads = []
if 'results' not in st.session_state: st.session_state.results = None

st.title("🏗️ Professional Beam Studio (Fixed Keys)")

# --- 2. Input Section ---
with st.sidebar:
    st.header("🧱 Material Config")
    fc = st.number_input("f'c (MPa)", 24.0, 50.0, 28.0)
    fy = st.number_input("fy (MPa)", 240.0, 500.0, 400.0)
    b_m = st.number_input("Width (m)", 0.15, 1.0, 0.3)
    h_m = st.number_input("Height (m)", 0.3, 2.0, 0.5)
    cover = st.number_input("Cover (mm)", 20, 75, 30)
    db_m = st.selectbox("Main Bar (mm)", [12, 16, 20, 25, 28])

with st.container(border=True):
    c_geo, c_load = st.columns([1, 2])
    
    with c_geo:
        st.subheader("📏 Geometry")
        n_spans = st.number_input("Num Spans", 1, 10, 2)
        spans = [st.number_input(f"L{i+1} (m)", 1.0, 20.0, 5.0, key=f"span_len_{i}") for i in range(n_spans)]
        
    with c_load:
        st.subheader("📥 Load Input")
        cols = st.columns(4)
        l_span_idx = cols[0].selectbox("Span Index", range(n_spans))
        l_type = cols[1].selectbox("Type", ["Point", "Uniform", "Moment"])
        l_mag = cols[2].number_input("Mag (kN)", 10.0)
        l_x = cols[3].number_input("Dist x (m)", 0.0, float(spans[l_span_idx]))
        
        if st.button("➕ Add Load", use_container_width=True):
            # Safe logic: ป้องกัน x ชนขอบ
            safe_x = min(l_x, spans[l_span_idx] - 0.01)
            safe_x = max(safe_x, 0.01)
            
            # ✅ FIX: ใช้ key 'span_index' แทน 'span' เพื่อให้ Solver อ่านค่าถูก
            st.session_state.loads.append({
                'span_index': l_span_idx,  # <-- จุดที่แก้ไข (Critical Fix)
                'type': l_type[0],         # 'P', 'U', 'M'
                'mag': l_mag * 1000, 
                'x': safe_x,
                'dist': spans[l_span_idx] if l_type == "Uniform" else 0.0
            })
            st.rerun()

# Show Current Loads
if st.session_state.loads:
    st.write("---")
    st.subheader(f"📋 Current Loads ({len(st.session_state.loads)})")
    
    # Debug Table
    debug_df = pd.DataFrame(st.session_state.loads)
    st.dataframe(debug_df, use_container_width=True)
    
    if st.button("🗑️ Clear All Loads"):
        st.session_state.loads = []
        st.session_state.results = None
        st.rerun()

# --- 3. Run Analysis ---
if st.button("🚀 EXECUTE ANALYSIS", type="primary", use_container_width=True):
    if not st.session_state.loads:
        st.warning("⚠️ Please add loads first.")
    else:
        try:
            supports = [{'id': i, 'type': 'Pin' if i==0 else 'Roller'} for i in range(n_spans+1)]
            I_val = (b_m * h_m**3) / 12
            
            # เรียก Solver
            sol = BeamSolver(spans, supports, st.session_state.loads, 2e11, I_val)
            st.session_state.results = sol.solve()
            
        except Exception as e:
            st.error(f"Solver Crash: {e}")

# --- 4. Display Results ---
if st.session_state.results:
    df, reac, eq = st.session_state.results
    
    # Safety Check
    if df is None or df.empty:
        st.error("❌ Solver returned empty data. (Try checking Span Index or Load Types)")
    else:
        # หาชื่อคอลัมน์ตำแหน่ง x (บางทีเป็น 'x', บางทีเป็น 'dist')
        x_col = 'x' if 'x' in df.columns else df.columns[0]
        
        st.success("✅ Analysis Successful!")
        
        # A. Longitudinal View
        st.subheader("🖼️ Longitudinal Detailing")
        
        fig_long = go.Figure()
        cum_l = 0
        for i, l in enumerate(spans):
            # Concrete Body
            fig_long.add_shape(type="rect", x0=cum_l, y0=0, x1=cum_l+l, y1=h_m, 
                               line=dict(color="black", width=2), fillcolor="rgba(220,220,220,0.5)")
            # Rebar Lines (Schematic)
            fig_long.add_trace(go.Scatter(x=[cum_l, cum_l+l], y=[h_m-0.05, h_m-0.05], mode="lines", line=dict(color="red", width=2), name="Top"))
            fig_long.add_trace(go.Scatter(x=[cum_l, cum_l+l], y=[0.05, 0.05], mode="lines", line=dict(color="blue", width=2), name="Bottom"))
            # Support Triangles
            fig_long.add_trace(go.Scatter(x=[cum_l], y=[-0.05], mode="markers", marker=dict(symbol="triangle-up", size=15, color="black"), name="Support"))
            cum_l += l
        # Last Support
        fig_long.add_trace(go.Scatter(x=[cum_l], y=[-0.05], mode="markers", marker=dict(symbol="triangle-up", size=15, color="black"), showlegend=False))
        
        fig_long.update_layout(height=250, showlegend=False, xaxis=dict(title="Length (m)"), yaxis=dict(visible=False, scaleanchor="x"))
        st.plotly_chart(fig_long, use_container_width=True)

        # B. Cross-Section Design
        st.subheader("📋 Span Calculations & Sections")
        cum_dist = [0] + list(np.cumsum(spans))
        
        for i in range(n_spans):
            with st.container(border=True):
                # Filter Data
                span_data = df[(df[x_col] >= cum_dist[i]) & (df[x_col] <= cum_dist[i+1])]
                
                if span_data.empty:
                    st.warning(f"No forces found for Span {i+1}")
                    continue

                # Design Calculation
                m_max = span_data['moment'].max()/1000
                m_min = span_data['moment'].min()/1000
                v_max = span_data['shear'].abs().max()/1000
                
                res = rc_design.design_span_expert(m_max, m_min, v_max, b_m, h_m, fc, fy, cover, db_m)

                col_text, col_draw = st.columns([1, 1])
                
                with col_text:
                    st.markdown(f"**Span {i+1}: Design Output**")
                    st.latex(r"M_u^{(+)} = %.2f \text{ kNm}" % res['mu_pos'])
                    st.latex(r"M_u^{(-)} = %.2f \text{ kNm}" % res['mu_neg'])
                    st.markdown(f"**Top Bars:** {res['neg']['n']} x DB{db_m} <span style='color:red'>({res['neg']['status']})</span>", unsafe_allow_html=True)
                    st.markdown(f"**Bot Bars:** {res['pos']['n']} x DB{db_m} <span style='color:blue'>({res['pos']['status']})</span>", unsafe_allow_html=True)
                    st.info(f"Shear Vu: {res['vu']:.1f} kN (PhiVc: {res['phi_vc']:.1f} kN)")

                with col_draw:
                    
                    n_t, n_b = int(res['neg']['n']), int(res['pos']['n'])
                    
                    fig_cs = go.Figure()
                    # Section Box
                    fig_cs.add_shape(type="rect", x0=0, y0=0, x1=b_m, y1=h_m, line=dict(color="black", width=3))
                    # Stirrup Box
                    cover_m = cover/1000
                    fig_cs.add_shape(type="rect", x0=cover_m, y0=cover_m, x1=b_m-cover_m, y1=h_m-cover_m, line=dict(color="gray", dash="dot"))
                    
                    # Draw Rebars (Real Scale)
                    # Top (Red)
                    for k in range(n_t):
                        cx = (b_m - 2*cover_m) / (n_t + 1) * (k+1) + cover_m
                        fig_cs.add_trace(go.Scatter(x=[cx], y=[h_m - cover_m - 0.01], mode="markers", marker=dict(color="red", size=10)))
                    # Bot (Blue)
                    for k in range(n_b):
                        cx = (b_m - 2*cover_m) / (n_b + 1) * (k+1) + cover_m
                        fig_cs.add_trace(go.Scatter(x=[cx], y=[cover_m + 0.01], mode="markers", marker=dict(color="blue", size=10)))
                        
                    fig_cs.update_layout(width=200, height=220, showlegend=False, xaxis=dict(visible=False, range=[-0.05, b_m+0.05]), yaxis=dict(visible=False, range=[-0.05, h_m+0.05]), margin=dict(l=10,r=10,t=10,b=10))
                    st.plotly_chart(fig_cs)
