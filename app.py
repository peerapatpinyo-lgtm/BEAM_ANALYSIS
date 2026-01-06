import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from solver import BeamSolver
import rc_design
import design_view

st.set_page_config(page_title="Expert Beam Designer PRO", layout="wide")

if 'loads' not in st.session_state: st.session_state.loads = []

# --- 1. LOAD INPUT SYSTEM (PRECISE CONTROL) ---
st.header("1. Global Load & Geometry Manager")
with st.container(border=True):
    col_g1, col_g2 = st.columns([1, 2])
    with col_g1:
        n_spans = st.number_input("Number of Spans", 1, 10, 1)
        spans = [st.number_input(f"L{i+1} (m)", 0.1, 25.0, 5.0, key=f"L_{i}") for i in range(n_spans)]
        
    with col_g2:
        st.subheader("Add Loads with Coordinates")
        c1, c2, c3, c4, c5 = st.columns([1.5, 1, 1, 1, 1])
        l_span = c1.selectbox("Span", range(n_spans))
        l_type = c2.selectbox("Type", ["Point (P)", "Uniform (U)", "Moment (M)"])
        l_mag = c3.number_input("Mag (kN or kNm)", 10.0)
        l_x = c4.number_input("Start x (m)", 0.0, spans[l_span])
        l_end = c5.number_input("End x (m)", spans[l_span]) if l_type == "Uniform (U)" else 0.0

        if st.button("➕ Add Precise Load", use_container_width=True):
            st.session_state.loads.append({
                'span_index': l_span, 'type': l_type[0], 
                'mag': l_mag*1000, 'x': l_x, 'dist': l_end - l_x if l_type == "Uniform (U)" else 0.0
            })
            st.rerun()

# --- 2. SUPPORT & MATERIAL ---
with st.sidebar:
    st.header("🏗️ Material & Section")
    fc = st.number_input("Concrete f'c (MPa)", 28.0)
    fy = st.number_input("Rebar fy (MPa)", 400.0)
    b_m = st.number_input("Width b (m)", 0.30)
    h_m = st.number_input("Height h (m)", 0.50)
    cover = st.number_input("Cover (mm)", 30.0)

# --- 3. ANALYSIS & DESIGN RESULTS ---
if st.button("🚀 RUN EXPERT ANALYSIS", type="primary", use_container_width=True):
    # Standard Support (Can be expanded with data_editor as before)
    supports = [{'id': i, 'type': 'Pin' if i==0 else 'Roller'} for i in range(n_spans+1)]
    sol = BeamSolver(spans, supports, st.session_state.loads, 2e11, b_m, h_m)
    df, reac, eq = sol.solve()

    if not df.empty:
        st.header("📊 Analysis & Force Diagrams")
        design_view.draw_interactive_diagrams(df, reac, spans, pd.DataFrame(supports), st.session_state.loads)
        

        # Detailed Design Section
        st.header("🧱 Expert Design Report (SDM)")
        db_m = st.selectbox("Main Bar (DB)", [12, 16, 20, 25, 28])
        
        cum_dist = [0] + list(np.cumsum(spans))
        for i in range(n_spans):
            with st.expander(f"SPAN {i+1} CALCULATION DETAILS", expanded=True):
                s_df = df[(df['x'] >= cum_dist[i]) & (df['x'] <= cum_dist[i+1])]
                res = rc_design.design_span_expert(s_df['moment'].max()/1000, s_df['moment'].min()/1000, s_df['shear'].abs().max()/1000, b_m, h_m, fc, fy, 240, cover, db_m, 9)
                
                c1, c2, c3 = st.columns([1, 1, 1])
                with c1:
                    st.markdown("**Flexural Check**")
                    st.latex(rf"M_u^{(+)} = {res['mu_pos']:.2f} \text{{ kNm}}")
                    st.latex(rf"M_u^{{(-)}} = {res['mu_neg']:.2f} \text{{ kNm}}")
                    st.write(f"Section Status: {res['pos']['status']}")
                with c2:
                    st.markdown("**Ductility & Strain**")
                    st.latex(rf"\epsilon_t = {res['pos'].get('et', 0):.5f}")
                    st.caption("εt > 0.005 = Tension Controlled")
                    st.latex(rf"a = {res['pos'].get('a', 0):.2f} \text{{ mm}}")
                with c3:
                    st.markdown("**Recommended Bars**")
                    st.info(f"Top: {res['neg']['n']}xDB{db_m}\nBot: {res['pos']['n']}xDB{db_m}")
                
                # Manual Adjustment Plot
                fig_cs = go.Figure()
                fig_cs.add_shape(type="rect", x0=0, y0=0, x1=b_m, y1=h_m, fillcolor="rgba(0,0,0,0.05)", line=dict(color="Black", width=3))
                # Visualize bars
                for j in range(int(res['neg']['n'])): fig_cs.add_trace(go.Scatter(x=[(b_m/(res['neg']['n']+1))*(j+1)], y=[h_m-(cover/1000)], mode='markers', marker=dict(color='Red')))
                for j in range(int(res['pos']['n'])): fig_cs.add_trace(go.Scatter(x=[(b_m/(res['pos']['n']+1))*(j+1)], y=[cover/1000], mode='markers', marker=dict(color='Blue')))
                fig_cs.update_layout(width=200, height=250, xaxis=dict(visible=False), yaxis=dict(visible=False), showlegend=False)
                st.plotly_chart(fig_cs)
