import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
# Import โมดูลที่แยกไว้
from solver import BeamSolver
import rc_design

st.set_page_config(page_title="Professional Beam Design", layout="wide")

# --- Session State ---
if 'loads' not in st.session_state: st.session_state.loads = []
if 'results' not in st.session_state: st.session_state.results = None

st.title("🏗️ Professional Beam Studio")

# ==========================================
# 1. SIDEBAR & INPUTS
# ==========================================
with st.sidebar:
    st.header("1. Material Properties")
    fc = st.number_input("Concrete f'c (MPa)", 18.0, 50.0, 24.0)
    fy = st.number_input("Rebar fy (MPa)", 240.0, 500.0, 400.0)
    b_m = st.number_input("Beam Width b (m)", 0.15, 1.0, 0.25)
    h_m = st.number_input("Beam Depth h (m)", 0.3, 2.0, 0.50)
    cover = st.number_input("Covering (mm)", 20.0, 75.0, 30.0)
    db_m = st.selectbox("Main Rebar DB (mm)", [12, 16, 20, 25, 28])
    
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
# 2. PREVIEW
# ==========================================
if st.session_state.loads:
    st.markdown("### 👁️ Model Preview")
    fig_struct = go.Figure()
    cum_dist = [0] + list(np.cumsum(spans))
    total_len = cum_dist[-1]
    
    # Beam
    fig_struct.add_trace(go.Scatter(x=[0, total_len], y=[0, 0], mode="lines", 
                                    line=dict(color="black", width=4), name="Beam"))
    
    # Supports
    for i, x_sup in enumerate(cum_dist):
        sup_type = "Pin" if i == 0 else "Roller"
        fig_struct.add_trace(go.Scatter(x=[x_sup], y=[-0.2], mode="markers+text", 
                                        marker=dict(symbol="triangle-up", size=15, color="black"), 
                                        text=sup_type, textposition="bottom center",
                                        name=f"Sup {i+1}", showlegend=False))

    # Loads
    for l in st.session_state.loads:
        start_x = cum_dist[l['span_index']]
        abs_x = start_x + l['x']
        if l['type'] == 'P':
            fig_struct.add_annotation(x=abs_x, y=0, ax=0, ay=-40, arrowhead=2, arrowcolor="red", text=f"{l['mag']/1000}kN")
        elif l['type'] == 'U':
            end_x = start_x + l['dist']
            fig_struct.add_shape(type="rect", x0=start_x, y0=0, x1=end_x, y1=0.5, 
                                 fillcolor="rgba(255, 0, 0, 0.2)", line=dict(width=0))
            fig_struct.add_annotation(x=(start_x+end_x)/2, y=0.5, text=f"UDL {l['mag']/1000} kN/m", showarrow=False)
        elif l['type'] == 'M':
            fig_struct.add_annotation(x=abs_x, y=0, text=f"M {l['mag']/1000}", showarrow=True, arrowhead=1)

    fig_struct.update_layout(height=250, showlegend=False, xaxis=dict(visible=True), yaxis=dict(visible=False, range=[-1, 2]))
    st.plotly_chart(fig_struct, use_container_width=True)
    
    if st.button("🗑️ Reset All Loads"):
        st.session_state.loads = []
        st.session_state.results = None
        st.rerun()

# ==========================================
# 3. RUN SOLVER
# ==========================================
if st.button("🚀 RUN ANALYSIS", type="primary", use_container_width=True):
    if not st.session_state.loads:
        st.error("Please add loads first.")
    else:
        try:
            supports = [{'id': i, 'type': 'Pin' if i==0 else 'Roller'} for i in range(n_spans+1)]
            I_val = (b_m * h_m**3) / 12
            # เรียกใช้ Class จากไฟล์ solver.py
            sol = BeamSolver(spans, supports, st.session_state.loads, 2e11, I_val)
            st.session_state.results = sol.solve()
        except Exception as e:
            st.error(f"Solver Error: {str(e)}")

# ==========================================
# 4. RESULTS
# ==========================================
if st.session_state.results:
    df, reac, eq = st.session_state.results
    
    if df is None or df.empty:
        st.error("Calculation Error.")
    else:
        x_col = 'x' if 'x' in df.columns else df.columns[0]
        st.success("Analysis Complete!")

        # --- A. Reaction Table (Fixed Logic) ---
        st.subheader("📌 Support Reactions")
        try:
            # ดึงค่าออกมาไม่ว่าจะเป็น Dict หรือ List
            raw_values = list(reac.values()) if isinstance(reac, dict) else reac
            clean_values = [float(v) for v in raw_values] # แปลงเป็น float
            reac_vals = np.array(clean_values) / 1000
            
            reac_data = []
            for i, r in enumerate(reac_vals):
                reac_data.append({
                    "Support ID": f"#{i+1}",
                    "Type": "Pin" if i==0 else "Roller",
                    "Reaction (kN)": f"{r:.2f}"
                })
            st.table(pd.DataFrame(reac_data))
        except Exception as e:
            st.error(f"Error displaying reactions: {e}")

        # --- B. Diagrams (Detailed) ---
        st.subheader("📈 Force Diagrams")
        fig_res = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                                subplot_titles=("Shear Force (SFD)", "Bending Moment (BMD)"))
        
        # Helper to find peaks
        def add_peak_labels(fig, x_data, y_data, row_idx, color, name, invert=False):
            y_scale = -1 if invert else 1
            fig.add_trace(go.Scatter(x=x_data, y=y_data, mode='lines', line=dict(color=color, width=2), 
                                     fill='tozeroy', fillcolor=f"rgba{color[3:-1]}, 0.1)", name=name), row=row_idx, col=1)
            # Max/Min Labels
            mx, mn = y_data.max(), y_data.min()
            mx_idx, mn_idx = y_data.idxmax(), y_data.idxmin()
            
            fig.add_annotation(x=x_data[mx_idx], y=mx, text=f"{mx:.2f}", showarrow=True, row=row_idx, col=1)
            fig.add_annotation(x=x_data[mn_idx], y=mn, text=f"{mn:.2f}", showarrow=True, row=row_idx, col=1)

        add_peak_labels(fig_res, df[x_col], df['shear']/1000, 1, 'rgb(46, 125, 50)', "Shear")
        add_peak_labels(fig_res, df[x_col], df['moment']/1000, 2, 'rgb(239, 108, 0)', "Moment", invert=True)

        # Draw Supports on Graph
        cum_dist = [0] + list(np.cumsum(spans))
        for d in cum_dist:
            fig_res.add_vline(x=d, line_dash="dash", line_color="gray")
            fig_res.add_trace(go.Scatter(x=[d], y=[0], mode="markers", marker=dict(symbol="triangle-up", color="black", size=10), showlegend=False), row=1, col=1)
            fig_res.add_trace(go.Scatter(x=[d], y=[0], mode="markers", marker=dict(symbol="triangle-up", color="black", size=10), showlegend=False), row=2, col=1)

        fig_res.update_layout(height=600, showlegend=False)
        fig_res.update_yaxes(title="V (kN)", row=1, col=1, zeroline=True, zerolinecolor='black')
        fig_res.update_yaxes(title="M (kNm)", row=2, col=1, autorange="reversed", zeroline=True, zerolinecolor='black')
        st.plotly_chart(fig_res, use_container_width=True)

        # --- C. Design Section ---
        st.divider()
        st.header("🏗️ Reinforcement Design")
        
        # เรียกใช้ฟังก์ชันจากไฟล์ rc_design.py
        for i in range(n_spans):
            with st.container(border=True):
                mask = (df[x_col] >= cum_dist[i]) & (df[x_col] <= cum_dist[i+1])
                span_data = df[mask]
                if span_data.empty: continue

                m_max = span_data['moment'].max() / 1000 
                m_min = span_data['moment'].min() / 1000 
                v_max = span_data['shear'].abs().max() / 1000 
                
                # Design Function Call
                res = rc_design.design_span_expert(m_max, m_min, v_max, b_m, h_m, fc, fy, cover, db_m)
                
                c1, c2 = st.columns([1, 1])
                with c1:
                    st.subheader(f"Span {i+1}")
                    st.write(f"**Forces:** M+ {m_max:.2f}, M- {m_min:.2f}, V {v_max:.2f}")
                    st.markdown(f"🔴 Top: **{res['neg']['n']}** - DB{db_m}")
                    st.markdown(f"🔵 Bot: **{res['pos']['n']}** - DB{db_m}")
                
                with c2:
                     # (Cross section drawing code omitted for brevity but same as before)
                     st.info("Cross section details available")
