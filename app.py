import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from solver import BeamSolver
import rc_design

st.set_page_config(page_title="Professional Beam Design", layout="wide")

# --- 1. Session State ---
if 'loads' not in st.session_state: st.session_state.loads = []
if 'results' not in st.session_state: st.session_state.results = None

st.title("🏗️ Professional Beam Studio (Full Option)")

# --- 2. Input Section ---
with st.sidebar:
    st.header("🧱 Material Config")
    fc = st.number_input("f'c (MPa)", 18.0, 50.0, 24.0)
    fy = st.number_input("fy (MPa)", 240.0, 500.0, 400.0)
    b_m = st.number_input("Width (m)", 0.15, 1.0, 0.25)
    h_m = st.number_input("Height (m)", 0.3, 2.0, 0.50)
    cover = st.number_input("Cover (mm)", 20.0, 75.0, 30.0)
    db_m = st.selectbox("Main Bar (mm)", [12, 16, 20, 25, 28])

with st.container(border=True):
    c1, c2 = st.columns([1, 2])
    with c1:
        st.subheader("📏 Geometry")
        n_spans = st.number_input("Num Spans", 1, 10, 2)
        spans = [st.number_input(f"L{i+1} (m)", 1.0, 20.0, 5.0, key=f"span_in_{i}") for i in range(n_spans)]
        
    with c2:
        st.subheader("📥 Load Manager")
        lc1, lc2, lc3, lc4 = st.columns(4)
        l_idx = lc1.selectbox("Span Index", range(n_spans))
        l_type = lc2.selectbox("Type", ["Uniform", "Point", "Moment"])
        l_mag = lc3.number_input("Value (kN)", 10.0)
        l_x = lc4.number_input("Dist x (m)", 0.0, float(spans[l_idx]))
        
        if st.button("➕ Add Load", use_container_width=True):
            safe_x = min(l_x, spans[l_idx] - 0.01)
            safe_x = max(safe_x, 0.01)
            st.session_state.loads.append({
                'span_index': l_idx,
                'type': l_type[0],
                'mag': l_mag * 1000, 
                'x': safe_x,
                'dist': spans[l_idx] if l_type[0] == "U" else 0.0
            })
            st.rerun()

if st.session_state.loads:
    st.info(f"Loaded {len(st.session_state.loads)} load cases.")
    if st.button("🗑️ Reset Loads"):
        st.session_state.loads = []
        st.session_state.results = None
        st.rerun()

# --- 3. Execution ---
if st.button("🚀 RUN ANALYSIS", type="primary", use_container_width=True):
    if not st.session_state.loads:
        st.error("Please add loads first.")
    else:
        try:
            supports = [{'id': i, 'type': 'Pin' if i==0 else 'Roller'} for i in range(n_spans+1)]
            I_val = (b_m * h_m**3) / 12
            sol = BeamSolver(spans, supports, st.session_state.loads, 2e11, I_val)
            st.session_state.results = sol.solve()
        except Exception as e:
            st.error(f"Solver Error: {str(e)}")

# --- 4. Display Results ---
if st.session_state.results:
    df, reac, eq = st.session_state.results
    
    if df is None or df.empty:
        st.error("❌ Solver returned empty data.")
    else:
        x_col = 'x' if 'x' in df.columns else df.columns[0]
        st.success("Analysis Complete!")
        
        # --- A. Force Diagrams (SFD & BMD) ---
        # ส่วนนี้ที่หายไป ผมเพิ่มกลับมาให้แล้วครับ!
        st.header("📈 Part 1: Force Diagrams (SFD & BMD)")
        

[Image of shear force and bending moment diagrams for a continuous beam]


        fig_main = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                                 subplot_titles=("Shear Force Diagram (SFD)", "Bending Moment Diagram (BMD)"))
        
        # Shear (Area Chart)
        fig_main.add_trace(go.Scatter(x=df[x_col], y=df['shear']/1000, fill='tozeroy', 
                                      line=dict(color='green'), name='Shear (kN)'), row=1, col=1)
        # Moment (Area Chart) - กลับด้านตามหลักวิศวกรรมโยธา (Invert Y)
        fig_main.add_trace(go.Scatter(x=df[x_col], y=df['moment']/1000, fill='tozeroy', 
                                      line=dict(color='orange'), name='Moment (kNm)'), row=2, col=1)
        
        # วาดเส้นแบ่ง Span
        cum_dist_list = [0] + list(np.cumsum(spans))
        for d in cum_dist_list:
            fig_main.add_vline(x=d, line_dash="dash", line_color="gray")
            
        fig_main.update_layout(height=500, showlegend=False)
        fig_main.update_yaxes(autorange="reversed", row=2, col=1) # กลับหัวกราฟโมเมนต์
        
        # ✅ ใส่ Key เพื่อป้องกัน Duplicate ID
        st.plotly_chart(fig_main, use_container_width=True, key="sfd_bmd_chart")


        # --- B. Longitudinal View (รูปตัดยาว) ---
        st.header("🖼️ Part 2: Longitudinal Detailing")
        
        
        fig_long = go.Figure()
        cum_l = 0
        for l in spans:
            fig_long.add_shape(type="rect", x0=cum_l, y0=0, x1=cum_l+l, y1=h_m, 
                               line=dict(color="black", width=2), fillcolor="rgba(200,200,200,0.3)")
            fig_long.add_trace(go.Scatter(x=[cum_l, cum_l+l], y=[h_m-0.05, h_m-0.05], mode="lines", line=dict(color="red", width=2), name="Top"))
            fig_long.add_trace(go.Scatter(x=[cum_l, cum_l+l], y=[0.05, 0.05], mode="lines", line=dict(color="blue", width=2), name="Bot"))
            fig_long.add_trace(go.Scatter(x=[cum_l], y=[-0.05], mode="markers", marker=dict(symbol="triangle-up", size=15, color="black"), showlegend=False))
            cum_l += l
        fig_long.add_trace(go.Scatter(x=[cum_l], y=[-0.05], mode="markers", marker=dict(symbol="triangle-up", size=15, color="black"), showlegend=False))
        fig_long.update_layout(height=200, showlegend=False, xaxis=dict(title="Length (m)"), yaxis=dict(visible=False, scaleanchor="x"))
        
        # ✅ ใส่ Key
        st.plotly_chart(fig_long, use_container_width=True, key="long_detailing_chart")


        # --- C. Cross Sections (รูปตัดขวาง) ---
        st.header("📋 Part 3: Section Design")
        
        for i in range(n_spans):
            with st.container(border=True):
                mask = (df[x_col] >= cum_dist_list[i]) & (df[x_col] <= cum_dist_list[i+1])
                span_data = df[mask]
                
                if span_data.empty:
                    st.warning(f"Span {i+1}: No data found.")
                    continue

                m_max = span_data['moment'].max() / 1000 
                m_min = span_data['moment'].min() / 1000 
                v_max = span_data['shear'].abs().max() / 1000 
                
                # เรียก Design
                res = rc_design.design_span_expert(m_max, m_min, v_max, b_m, h_m, fc, fy, cover, db_m)
                
                c1, c2 = st.columns([1, 1])
                with c1:
                    st.subheader(f"Span {i+1} Results")
                    st.write(f"**M+:** {res['mu_pos']:.2f} | **M-:** {res['mu_neg']:.2f} kNm")
                    st.write(f"**V:** {res['vu']:.2f} kN")
                    st.divider()
                    st.markdown(f"🔴 **Top:** {res['neg']['n']} x DB{db_m}")
                    st.markdown(f"🔵 **Bot:** {res['pos']['n']} x DB{db_m}")

                with c2:
                    st.write("**Cross-Section Drawing**")
                    
                    
                    fig_cs = go.Figure()
                    fig_cs.add_shape(type="rect", x0=0, y0=0, x1=b_m, y1=h_m, line=dict(color="black", width=3))
                    c_m = cover/1000
                    fig_cs.add_shape(type="rect", x0=c_m, y0=c_m, x1=b_m-c_m, y1=h_m-c_m, line=dict(color="gray", dash="dot"))
                    
                    n_t = res['neg']['n']
                    n_b = res['pos']['n']
                    
                    for k in range(n_t):
                        x_pos = (b_m - 2*c_m)/(n_t+1)*(k+1) + c_m
                        fig_cs.add_trace(go.Scatter(x=[x_pos], y=[h_m - c_m - 0.01], mode="markers", marker=dict(color="red", size=12)))
                    for k in range(n_b):
                        x_pos = (b_m - 2*c_m)/(n_b+1)*(k+1) + c_m
                        fig_cs.add_trace(go.Scatter(x=[x_pos], y=[c_m + 0.01], mode="markers", marker=dict(color="blue", size=12)))
                        
                    fig_cs.update_layout(width=200, height=220, showlegend=False, xaxis=dict(visible=False, range=[-0.05, b_m+0.05]), yaxis=dict(visible=False, range=[-0.05, h_m+0.05]), margin=dict(l=10,r=10,t=10,b=10))
                    
                    # ✅ ใส่ Key ตาม Index i
                    st.plotly_chart(fig_cs, use_container_width=True, key=f"sec_chart_{i}")
