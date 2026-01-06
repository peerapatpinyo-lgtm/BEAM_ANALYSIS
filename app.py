import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from solver import BeamSolver
import rc_design
import design_view

st.set_page_config(page_title="Professional Beam Designer", layout="wide")

# Persistent State
if 'loads' not in st.session_state: st.session_state.loads = []
if 'spans' not in st.session_state: st.session_state.spans = [5.0]
if 'supports' not in st.session_state: 
    st.session_state.supports = [{'id': 0, 'type': 'Pin'}, {'id': 1, 'type': 'Roller'}]

# --- SIDEBAR: PRIMARY GEOMETRY ---
with st.sidebar:
    st.header("🏢 Beam Geometry")
    fc = st.number_input("Concrete f'c (MPa)", 28.0)
    b_m = st.number_input("Width b (m)", 0.30)
    h_m = st.number_input("Height h (m)", 0.50)
    
    st.divider()
    n_spans = st.number_input("Number of Spans", 1, 10, len(st.session_state.spans))
    if n_spans != len(st.session_state.spans):
        st.session_state.spans = [5.0] * n_spans
        st.session_state.supports = [{'id': i, 'type': 'Pin' if i==0 else 'None'} for i in range(n_spans+1)]
        st.rerun()

st.title("🏗️ Professional Beam Studio")

# --- LOAD MANAGEMENT ---
col_geo, col_load = st.columns([1, 1])
with col_geo:
    st.subheader("⚓ Support Configuration")
    df_sup = pd.DataFrame([{'id': i, 'type': next((s['type'] for s in st.session_state.supports if s['id'] == i), 'None')} for i in range(n_spans+1)])
    ed_sup = st.data_editor(df_sup, column_config={"type": st.column_config.SelectboxColumn("Type", options=["None", "Pin", "Roller", "Fixed"])}, hide_index=True)
    st.session_state.supports = ed_sup.to_dict('records')

with col_load:
    st.subheader("📥 Add Loads")
    la, lb, lc = st.columns(3)
    l_idx = la.selectbox("Span", range(n_spans))
    l_type = lb.selectbox("Type", ["P", "U", "M"])
    l_mag = lc.number_input("Mag (kN)", 10.0)
    if st.button("➕ Add Load"):
        st.session_state.loads.append({'span_index': l_idx, 'type': l_type, 'mag': l_mag*1000, 'x': 0.0, 'dist': st.session_state.spans[l_idx]})
        st.rerun()
    
    for i, ld in enumerate(st.session_state.loads):
        c1, c2 = st.columns([4, 1])
        c1.write(f"{i+1}. {ld['type']} | {ld['mag']/1000}kN | Span {ld['span_index']+1}")
        if c2.button("🗑️", key=f"del_{i}"):
            st.session_state.loads.pop(i); st.rerun()

# --- EXECUTION ---
st.divider()
if st.button("🚀 RUN ANALYSIS", type="primary", use_container_width=True) or 'df_result' in st.session_state:
    sol = BeamSolver(st.session_state.spans, st.session_state.supports, st.session_state.loads, 2e11, b_m, h_m)
    df, reac, eq = sol.solve()
    st.session_state.df_result = df # Store result to keep it on screen

    if not df.empty:
        # ANALYSIS OUTPUT (Always on top)
        st.header("📊 PART I: Analysis Results")
        # Fixed: Passing the actual dataframe instead of None
        sup_df_fixed = pd.DataFrame(st.session_state.supports)
        design_view.draw_interactive_diagrams(df, reac, st.session_state.spans, sup_df_fixed, st.session_state.loads)
        
        # --- NEW: RC DESIGN AREA (POST-ANALYSIS INPUTS) ---
        st.divider()
        st.header("🧱 PART II: RC Design & Reinforcement")
        
        with st.container(border=True):
            st.subheader("⚙️ Tuning Parameters")
            rc1, rc2, rc3, rc4, rc5 = st.columns(5)
            fy = rc1.number_input("Main fy (MPa)", 400)
            fyt = rc2.number_input("Stirrup fyt (MPa)", 240)
            cover = rc3.slider("Cover (mm)", 20, 50, 30)
            db_m = rc4.selectbox("Main Bar (DB)", [12, 16, 20, 25])
            db_s = rc5.selectbox("Stirrup (RB)", [6, 9, 12], index=1)

        # Calculation per Span for Longitudinal View
        cum_dist = [0] + list(np.cumsum(st.session_state.spans))
        span_designs = []
        for i in range(len(st.session_state.spans)):
            span_df = df[(df['x'] >= cum_dist[i]) & (df['x'] <= cum_dist[i+1])]
            m_pos = span_df['moment'].max()/1000
            m_neg = span_df['moment'].min()/1000
            v_max = span_df['shear'].abs().max()/1000
            span_designs.append(rc_design.design_section(m_pos, m_neg, v_max, b_m, h_m, fc, fy, fyt, cover, db_m, db_s))

        # DRAWING: LONGITUDINAL SECTION
        
        fig_long = go.Figure()
        fig_long.add_shape(type="rect", x0=0, y0=0, x1=cum_dist[-1], y1=h_m, line=dict(color="Black", width=3), fillcolor="rgba(100,100,100,0.1)")
        
        c_off = cover/1000
        for i, des in enumerate(span_designs):
            # Bar Lines
            fig_long.add_trace(go.Scatter(x=[cum_dist[i]+0.05, cum_dist[i+1]-0.05], y=[h_m-c_off, h_m-c_off], 
                                          mode='lines+text', line=dict(color='Red', width=des['n_top']), 
                                          text=[f"{des['n_top']}-DB{db_m}"], textposition="top center"))
            fig_long.add_trace(go.Scatter(x=[cum_dist[i]+0.05, cum_dist[i+1]-0.05], y=[c_off, c_off], 
                                          mode='lines+text', line=dict(color='Blue', width=des['n_bot']), 
                                          text=[f"{des['n_bot']}-DB{db_m}"], textposition="bottom center"))
            # Stirrup visualization (simplified)
            for sx in np.linspace(cum_dist[i], cum_dist[i+1], 10):
                fig_long.add_shape(type="line", x0=sx, y0=c_off, x1=sx, y1=h_m-c_off, line=dict(color="lightgray", width=1))

        fig_long.update_layout(title="Beam Longitudinal Reinforcement", showlegend=False, height=400)
        st.plotly_chart(fig_long, use_container_width=True)

        # OPTIMIZATION REPORT
        st.subheader("📋 Automatic Optimization Report")
        for i, des in enumerate(span_designs):
            k = (max(abs(mu_pos), abs(mu_neg))*1e6) / (0.9 * des['b'] * des['d']**2)
            status = "✅ OPTIMIZED" if k < 0.15*fc else "⚠️ SECTION TOO SMALL"
            st.write(f"**Span {i+1}:** {status} (k={k:.2f}) | Recommend Stirrup spacing: {des['spacing']} mm")
