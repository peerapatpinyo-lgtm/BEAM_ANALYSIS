import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from solver import BeamSolver
import rc_design
import design_view

st.set_page_config(page_title="Professional Structural Designer", layout="wide")

if 'loads' not in st.session_state: st.session_state.loads = []

st.title("🏗️ Professional Structural Beam Studio")

# --- 1. COORDINATE-BASED LOAD MANAGER ---
st.header("1. Advanced Loading & Geometry")
with st.container(border=True):
    col_g1, col_g2 = st.columns([1, 3])
    with col_g1:
        n_spans = st.number_input("Number of Spans", 1, 10, 1)
        spans = [st.number_input(f"L{i+1} (m)", 0.1, 25.0, 5.0, key=f"L_{i}") for i in range(n_spans)]
        
    with col_g2:
        c1, c2, c3, c4, c5 = st.columns([1, 1.2, 1, 1, 1])
        l_span = c1.selectbox("Select Span", range(n_spans))
        l_type = c2.selectbox("Load Type", ["Point (P)", "Uniform (U)", "Moment (M)"])
        l_mag = c3.number_input("Mag (kN/kNm)", 15.0)
        l_x = c4.number_input("Start Pos (m)", 0.0, float(spans[l_span]))
        l_end = c5.number_input("End Pos (m)", float(spans[l_span])) if l_type == "Uniform (U)" else l_x

        if st.button("➕ Add Load Case", use_container_width=True):
            st.session_state.loads.append({
                'span_index': l_span, 'type': l_type[0], 
                'mag': l_mag*1000, 'x': l_x, 'dist': l_end - l_x if l_type == "Uniform (U)" else 0.0
            })
            st.rerun()

# --- 2. MATERIAL & SECTION ---
with st.sidebar:
    st.header("🧱 Material Properties")
    fc = st.number_input("Concrete f'c (MPa)", 28.0)
    fy = st.number_input("Main Rebar fy (MPa)", 400.0)
    b_m = st.number_input("Beam Width b (m)", 0.30)
    h_m = st.number_input("Beam Height h (m)", 0.50)
    cover = st.number_input("Clear Cover (mm)", 30.0)

# --- 3. ANALYSIS & DESIGN SHEETS ---
if st.button("🚀 EXECUTE STRUCTURAL ANALYSIS", type="primary", use_container_width=True):
    supports = [{'id': i, 'type': 'Pin' if i==0 else 'Roller'} for i in range(n_spans+1)]
    sol = BeamSolver(spans, supports, st.session_state.loads, 2e11, b_m, h_m)
    df, reac, eq = sol.solve()

    if not df.empty:
        st.header("📊 Analysis Result: Internal Forces")
        design_view.draw_interactive_diagrams(df, reac, spans, pd.DataFrame(supports), st.session_state.loads)
        

[Image of shear force and bending moment diagrams for a continuous beam]


        st.header("📋 Technical Design Sheet (SDM/ACI)")
        db_m = st.selectbox("Rebar Size (DB)", [12, 16, 20, 25, 28])
        
        cum_dist = [0] + list(np.cumsum(spans))
        for i in range(n_spans):
            with st.expander(f"SPAN {i+1} DETAILED CALCULATION", expanded=True):
                s_df = df[(df['x'] >= cum_dist[i]) & (df['x'] <= cum_dist[i+1])]
                res = rc_design.design_span_expert(s_df['moment'].max()/1000, s_df['moment'].min()/1000, 
                                                   s_df['shear'].abs().max()/1000, b_m, h_m, fc, fy, 240, cover, db_m, 9)
                
                c1, c2, c3 = st.columns([1.5, 1, 1.5])
                with c1:
                    st.markdown("**1. Flexural Capacity**")
                    # Corrected LaTeX escaping by separating string from logic
                    pos_m = res['pos']['mu_val']
                    neg_m = res['neg']['mu_val']
                    st.latex(rf"M_{{u}}^{{(+)}} = {pos_m:.2f} \text{{ kNm}}")
                    st.latex(rf"M_{{u}}^{{(-)}} = {neg_m:.2f} \text{{ kNm}}")
                    
                with c2:
                    st.markdown("**2. Strain & Ductility**")
                    st.latex(rf"\epsilon_t = {res['pos']['et']:.5f}")
                    st.latex(rf"a = {res['pos']['a']:.2f} \text{{ mm}}")
                    st.write(f"Status: {res['pos']['status']}")

                with c3:
                    st.markdown("**3. Reinforcement & Section**")
                    n_top = st.number_input(f"Top DB{db_m}", 2, 12, int(res['neg']['n']), key=f"t_{i}")
                    n_bot = st.number_input(f"Bot DB{db_m}", 2, 12, int(res['pos']['n']), key=f"b_{i}")
                    
                    fig_cs = go.Figure()
                    fig_cs.add_shape(type="rect", x0=0, y0=0, x1=b_m, y1=h_m, fillcolor="rgba(0,0,0,0.05)", line=dict(color="Black", width=3))
                    for j in range(n_top): 
                        fig_cs.add_trace(go.Scatter(x=[(b_m/(n_top+1))*(j+1)], y=[h_m-(cover/1000)], mode='markers', marker=dict(color='Red', size=db_m)))
                    for j in range(n_bot): 
                        fig_cs.add_trace(go.Scatter(x=[(b_m/(n_bot+1))*(j+1)], y=[cover/1000], mode='markers', marker=dict(color='Blue', size=db_m)))
                    fig_cs.update_layout(width=200, height=220, xaxis=dict(visible=False), yaxis=dict(visible=False), showlegend=False, margin=dict(l=5,r=5,t=5,b=5))
                    st.plotly_chart(fig_cs)
                    

                st.divider()
                st.markdown("**4. Shear Design Summary**")
                st.latex(rf"V_u = {res['vu']:.2f} \text{{ kN}}, \quad \phi V_c = {res['phi_vc']:.2f} \text{{ kN}}")
                if res['vu'] > res['phi_vc']:
                    st.warning("Stirrups required for shear reinforcement.")
                else:
                    st.success("Concrete capacity sufficient (Nominal stirrups applied).")
