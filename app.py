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

st.title("🏗️ Professional Beam Studio (Complete Edition)")

# ==========================================
# 🧱 INPUTS
# ==========================================
with st.sidebar:
    st.header("1. Material Properties")
    fc = st.number_input("Concrete f'c (MPa)", 18.0, 50.0, 24.0)
    fy = st.number_input("Rebar fy (MPa)", 240.0, 500.0, 400.0)
    b_m = st.number_input("Beam Width b (m)", 0.15, 1.0, 0.25)
    h_m = st.number_input("Beam Depth h (m)", 0.3, 2.0, 0.50)
    cover = st.number_input("Covering (mm)", 20.0, 75.0, 30.0)
    db_m = st.selectbox("Main Rebar DB (mm)", [12, 16, 20, 25, 28])
    
    # แสดงค่า I ที่คำนวณได้ทันที
    I_show = (b_m * h_m**3) / 12
    st.info(f"ℹ️ Moment of Inertia (I): {I_show:.6f} m⁴")

with st.container(border=True):
    col_geo, col_load = st.columns([1, 2])
    
    # --- Geometry ---
    with col_geo:
        st.subheader("2. Geometry")
        n_spans = st.number_input("Number of Spans", 1, 10, 2)
        spans = []
        for i in range(n_spans):
            val = st.number_input(f"Length Span {i+1} (m)", 1.0, 20.0, 5.0, key=f"span_len_{i}")
            spans.append(val)

    # --- Load Input ---
    with col_load:
        st.subheader("3. Loads")
        lc1, lc2, lc3, lc4 = st.columns(4)
        l_idx = lc1.selectbox("On Span", range(n_spans))
        l_type = lc2.selectbox("Type", ["Uniform", "Point", "Moment"])
        l_mag = lc3.number_input("Value (kN or kNm)", 10.0)
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

# ==========================================
# 🖼️ LOAD DIAGRAM (ส่วนแสดง Loads บนคาน)
# ==========================================
if st.session_state.loads:
    st.markdown("### 👁️ Model Preview (Loads & Structure)")
    
    # สร้างกราฟจำลองคานและโหลด
    fig_struct = go.Figure()
    
    # 1. วาดเส้นคาน
    cum_dist = [0] + list(np.cumsum(spans))
    total_len = cum_dist[-1]
    fig_struct.add_trace(go.Scatter(x=[0, total_len], y=[0, 0], mode="lines", 
                                    line=dict(color="black", width=4), name="Beam"))
    
    # 2. วาด Supports
    for x_sup in cum_dist:
        fig_struct.add_trace(go.Scatter(x=[x_sup], y=[-0.2], mode="markers", 
                                        marker=dict(symbol="triangle-up", size=15, color="black"), 
                                        name="Support", showlegend=False))

    # 3. วาด Loads
    for l in st.session_state.loads:
        start_x = cum_dist[l['span_index']]
        abs_x = start_x + l['x']
        
        if l['type'] == 'P': # Point Load
            fig_struct.add_annotation(x=abs_x, y=0, ax=0, ay=-40, arrowhead=2, arrowcolor="red", text=f"{l['mag']/1000}kN")
        elif l['type'] == 'U': # Uniform Load
            # วาดเส้น Uniform แบบง่าย
            end_x = start_x + l['dist']
            fig_struct.add_shape(type="rect", x0=start_x, y0=0, x1=end_x, y1=0.5, 
                                 fillcolor="rgba(255, 0, 0, 0.2)", line=dict(width=0))
            fig_struct.add_annotation(x=(start_x+end_x)/2, y=0.5, text=f"UDL {l['mag']/1000} kN/m", showarrow=False)
        elif l['type'] == 'M': # Moment
            fig_struct.add_annotation(x=abs_x, y=0, text=f"M {l['mag']/1000}", showarrow=True, arrowhead=1)

    fig_struct.update_layout(height=200, showlegend=False, 
                             xaxis=dict(title="Length (m)", range=[-0.5, total_len+0.5]), 
                             yaxis=dict(visible=False, range=[-1, 2]))
    st.plotly_chart(fig_struct, use_container_width=True, key="structure_preview")
    
    if st.button("🗑️ Reset All Loads"):
        st.session_state.loads = []
        st.session_state.results = None
        st.rerun()

# ==========================================
# 🚀 EXECUTION
# ==========================================
if st.button("🚀 RUN ANALYSIS", type="primary", use_container_width=True):
    if not st.session_state.loads:
        st.error("Please add loads first.")
    else:
        try:
            supports = [{'id': i, 'type': 'Pin' if i==0 else 'Roller'} for i in range(n_spans+1)]
            # --- คำนวณ I ตรงนี้ ---
            I_val = (b_m * h_m**3) / 12
            
            sol = BeamSolver(spans, supports, st.session_state.loads, 2e11, I_val)
            st.session_state.results = sol.solve()
        except Exception as e:
            st.error(f"Solver Error: {str(e)}")

# ==========================================
# 📊 RESULTS
# ==========================================
if st.session_state.results:
    df, reac, eq = st.session_state.results
    
    if df is None or df.empty:
        st.error("Error in calculation.")
    else:
        x_col = 'x' if 'x' in df.columns else df.columns[0]
        st.success("Analysis Complete!")

        # --- 1. Reactions ---
        st.subheader("📌 Support Reactions")
        reac_df = pd.DataFrame([reac/1000], columns=[f"Sup {i+1}" for i in range(len(reac))])
        reac_df.index = ["Reaction (kN)"]
        st.table(reac_df)

        # --- 2. Analysis Graphs (Classic Style) ---
        st.subheader("📈 Force Diagrams (SFD & BMD)")
        
        # ใช้ make_subplots แบบเส้น Clean (ไม่ Fill สีหนักๆ แบบอันใหม่)
        fig_res = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                                subplot_titles=("Shear Force (SFD)", "Bending Moment (BMD)"))
        
        # SFD
        fig_res.add_trace(go.Scatter(x=df[x_col], y=df['shear']/1000, mode='lines', 
                                     line=dict(color='green', width=2), fill='tozeroy', fillcolor='rgba(0, 255, 0, 0.1)'), row=1, col=1)
        # BMD (Inverted Y)
        fig_res.add_trace(go.Scatter(x=df[x_col], y=df['moment']/1000, mode='lines', 
                                     line=dict(color='orange', width=2), fill='tozeroy', fillcolor='rgba(255, 165, 0, 0.1)'), row=2, col=1)
        
        # เส้นแบ่ง Span
        for d in cum_dist:
            fig_res.add_vline(x=d, line_dash="dash", line_color="gray")

        fig_res.update_layout(height=500, showlegend=False)
        fig_res.update_yaxes(title_text="V (kN)", row=1, col=1)
        fig_res.update_yaxes(title_text="M (kNm)", autorange="reversed", row=2, col=1) # กลับหัวโมเมนต์ตามหลักโยธา
        
        st.plotly_chart(fig_res, use_container_width=True, key="analysis_results")

        # --- 3. Longitudinal & Sections ---
        st.divider()
        st.header("🏗️ Detailed Design")
        
        # Longitudinal
        
        fig_long = go.Figure()
        cum_l = 0
        for l in spans:
            fig_long.add_shape(type="rect", x0=cum_l, y0=0, x1=cum_l+l, y1=h_m, line=dict(color="black", width=2), fillcolor="#f0f0f0")
            fig_long.add_trace(go.Scatter(x=[cum_l, cum_l+l], y=[h_m-0.05, h_m-0.05], mode="lines", line=dict(color="red", width=2)))
            fig_long.add_trace(go.Scatter(x=[cum_l, cum_l+l], y=[0.05, 0.05], mode="lines", line=dict(color="blue", width=2)))
            fig_long.add_trace(go.Scatter(x=[cum_l], y=[-0.1], mode="markers", marker=dict(symbol="triangle-up", size=15, color="black")))
            cum_l += l
        fig_long.add_trace(go.Scatter(x=[cum_l], y=[-0.1], mode="markers", marker=dict(symbol="triangle-up", size=15, color="black")))
        fig_long.update_layout(height=200, showlegend=False, xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor="x"))
        st.plotly_chart(fig_long, use_container_width=True, key="long_detailing")

        # Cross Sections Loop
        for i in range(n_spans):
            with st.container(border=True):
                mask = (df[x_col] >= cum_dist[i]) & (df[x_col] <= cum_dist[i+1])
                span_data = df[mask]
                if span_data.empty: continue

                m_max = span_data['moment'].max() / 1000 
                m_min = span_data['moment'].min() / 1000 
                v_max = span_data['shear'].abs().max() / 1000 
                
                res = rc_design.design_span_expert(m_max, m_min, v_max, b_m, h_m, fc, fy, cover, db_m)
                
                c1, c2 = st.columns([1, 1])
                with c1:
                    st.subheader(f"Span {i+1} Design")
                    st.write(f"Forces: M+ {m_max:.2f}, M- {m_min:.2f}, V {v_max:.2f}")
                    st.markdown(f"🔴 Top: **{res['neg']['n']}** - DB{db_m}")
                    st.markdown(f"🔵 Bot: **{res['pos']['n']}** - DB{db_m}")
                
                with c2:
                    fig_cs = go.Figure()
                    fig_cs.add_shape(type="rect", x0=0, y0=0, x1=b_m, y1=h_m, line=dict(color="black", width=3))
                    c_val = cover/1000
                    fig_cs.add_shape(type="rect", x0=c_val, y0=c_val, x1=b_m-c_val, y1=h_m-c_val, line=dict(color="gray", dash="dot"))
                    
                    # Rebars visualization
                    n_t = int(res['neg']['n'])
                    n_b = int(res['pos']['n'])
                    for k in range(n_t):
                        fig_cs.add_trace(go.Scatter(x=[(b_m - 2*c_val)/(n_t+1)*(k+1) + c_val], y=[h_m - c_val - 0.01], mode="markers", marker=dict(color="red", size=12)))
                    for k in range(n_b):
                        fig_cs.add_trace(go.Scatter(x=[(b_m - 2*c_val)/(n_b+1)*(k+1) + c_val], y=[c_val + 0.01], mode="markers", marker=dict(color="blue", size=12)))
                        
                    fig_cs.update_layout(width=200, height=200, showlegend=False, xaxis=dict(visible=False, range=[-0.05, b_m+0.05]), yaxis=dict(visible=False, range=[-0.05, h_m+0.05]), margin=dict(l=10,r=10,t=10,b=10))
                    
                    # Key is critical here
                    st.plotly_chart(fig_cs, use_container_width=False, key=f"section_{i}")
