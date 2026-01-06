import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from solver import BeamSolver
import rc_design
import design_view

st.set_page_config(page_title="World-Class Beam Designer", layout="wide")

if 'loads' not in st.session_state: st.session_state.loads = []
if 'spans' not in st.session_state: st.session_state.spans = [5.0]

# --- SIDEBAR ---
with st.sidebar:
    st.title("🏆 Global Settings")
    fc = st.number_input("f'c (MPa)", 28.0)
    fy = st.number_input("fy (Main MPa)", 400.0)
    fyt = st.number_input("fyt (Stirrup MPa)", 240.0)
    b_m = st.number_input("Width b (m)", 0.30)
    h_m = st.number_input("Height h (m)", 0.50)
    cover = st.slider("Cover (mm)", 20, 75, 30)
    db_m = st.selectbox("Main Bar (mm)", [12, 16, 20, 25])
    db_s = st.selectbox("Stirrup (mm)", [6, 9, 12], index=1)

# --- 1. LOAD MANAGEMENT ---
st.header("1. Load Management")
c_in, c_list = st.columns([1, 2])
with c_in:
    with st.container(border=True):
        l_span = st.selectbox("Span", range(len(st.session_state.spans)))
        l_type = st.selectbox("Type", ["P","U","M"])
        l_mag = st.number_input("Mag (kN)", 10.0)
        l_x = st.number_input("x (m)", 0.0)
        l_d = st.number_input("Len (m)", 0.0) if l_type == "U" else 0.0
        if st.button("➕ Add Load"):
            st.session_state.loads.append({'span_index': l_span, 'type': l_type[0], 'mag': l_mag*1000, 'x': l_x, 'dist': l_d})
            st.rerun()

with c_list:
    for i, ld in enumerate(st.session_state.loads):
        cols = st.columns([5, 1])
        cols[0].write(f"Span {ld['span_index']+1}: {ld['type']} {ld['mag']/1000}kN at {ld['x']}m")
        if cols[1].button("🗑️", key=f"d_{i}"): st.session_state.loads.pop(i); st.rerun()

# --- 2. EXECUTION ---
if st.button("🚀 ANALYZE & DESIGN", type="primary", use_container_width=True):
    sol = BeamSolver(st.session_state.spans, [{'id':i,'type':'Pin' if i==0 else 'Roller'} for i in range(len(st.session_state.spans)+1)], st.session_state.loads, 2e11, b_m, h_m)
    df, reac, eq = sol.solve()
    
    if not df.empty:
        # ANALYSIS PLOTS
        design_view.draw_interactive_diagrams(df, reac, st.session_state.spans, None, st.session_state.loads)

        # DESIGN PER SPAN
        st.header("2. Longitudinal Reinforcement Design")
        cum_dist = [0] + list(np.cumsum(st.session_state.spans))
        span_designs = []
        
        for i in range(len(st.session_state.spans)):
            span_df = df[(df['x'] >= cum_dist[i]) & (df['x'] <= cum_dist[i+1])]
            m_pos = span_df['moment'].max()/1000
            m_neg = span_df['moment'].min()/1000
            v_max = span_df['shear'].abs().max()/1000
            
            res = rc_design.design_section(m_pos, m_neg, v_max, b_m, h_m, fc, fy, fyt, cover, db_m, db_s)
            span_designs.append(res)

        # DRAW LONGITUDINAL SECTION
        
        fig_long = go.Figure()
        # Beam Outline
        total_l = cum_dist[-1]
        fig_long.add_shape(type="rect", x0=0, y0=0, x1=total_l, y1=h_m, line=dict(color="Black", width=3), fillcolor="rgba(200,200,200,0.2)")
        
        c_off = cover/1000
        for i, des in enumerate(span_designs):
            s_start = cum_dist[i]
            s_end = cum_dist[i+1]
            # Top Bars (Red)
            fig_long.add_trace(go.Scatter(x=[s_start+0.1, s_end-0.1], y=[h_m-c_off, h_m-c_off], mode='lines+text', 
                                          line=dict(color='Red', width=des['n_top']*2), text=[f"{des['n_top']}-DB{db_m}"], textposition="top center"))
            # Bottom Bars (Blue)
            fig_long.add_trace(go.Scatter(x=[s_start+0.1, s_end-0.1], y=[c_off, c_off], mode='lines+text', 
                                          line=dict(color='Blue', width=des['n_bot']*2), text=[f"{des['n_bot']}-DB{db_m}"], textposition="bottom center"))
            # Stirrups (Vertical lines)
            stirrup_x = np.arange(s_start, s_end, des['spacing']/1000)
            for sx in stirrup_x:
                fig_long.add_shape(type="line", x0=sx, y0=c_off, x1=sx, y1=h_m-c_off, line=dict(color="Gray", width=1))

        fig_long.update_layout(title="Longitudinal Reinforcement Detail", showlegend=False, height=400)
        st.plotly_chart(fig_long, use_container_width=True)
        
        # DRAW CROSS SECTION (LAST SPAN)
        st.subheader("Typical Cross Section")
