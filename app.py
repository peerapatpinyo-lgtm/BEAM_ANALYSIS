import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from solver import BeamSolver
import rc_design
import design_view

st.set_page_config(page_title="ProBeam Structural Suite", layout="wide")

# Persistent State Initialization
if 'loads' not in st.session_state: st.session_state.loads = []
if 'analysis_results' not in st.session_state: st.session_state.analysis_results = None

with st.sidebar:
    st.header("🛠️ Material & Section")
    with st.container(border=True):
        fc = st.number_input("f'c Concrete (MPa)", 28.0)
        fy = st.number_input("fy Main Rebar (MPa)", 400.0)
        fyt = st.number_input("fy Stirrups (MPa)", 240.0)
        b_m = st.number_input("Width b (m)", 0.30)
        h_m = st.number_input("Height h (m)", 0.50)
        cover = st.number_input("Clear Cover (mm)", 30.0)

st.title("🏗️ Professional Beam Analysis & Design Studio")
st.caption("Method: Strength Design (SDM) | ACI 318-19 / EIT Standard")

# --- 1. GEOMETRY & LOADS ---
tab_geo, tab_load = st.tabs(["📏 Geometry & Supports", "📥 Load Manager"])

with tab_geo:
    st.subheader("Span Configuration")
    n_spans = st.number_input("Number of Spans", 1, 10, 1)
    spans = [st.number_input(f"Span {i+1} Length (m)", 0.1, 25.0, 5.0, key=f"L_{i}") for i in range(n_spans)]
    
    st.subheader("Support Conditions")
    df_sup = pd.DataFrame([{'Support ID': i, 'Type': 'Pin' if i==0 else 'Roller'} for i in range(n_spans+1)])
    ed_sup = st.data_editor(df_sup, column_config={"Type": st.column_config.SelectboxColumn("Type", options=["Pin", "Roller", "Fixed", "None"])}, hide_index=True, use_container_width=True)
    supports = ed_sup.to_dict('records')

with tab_load:
    st.subheader("Apply Loads")
    with st.container(border=True):
        lc1, lc2, lc3, lc4 = st.columns(4)
        l_span = lc1.selectbox("Target Span", range(n_spans), format_func=lambda x: f"Span {x+1}")
        l_type = lc2.selectbox("Load Type", ["Uniform (U)", "Point (P)"])
        l_mag = lc3.number_input("Magnitude (kN or kN/m)", 20.0)
        if lc4.button("➕ Add Load", use_container_width=True):
            st.session_state.loads.append({'span_index': l_span, 'type': l_type[0], 'mag': l_mag*1000, 'x': 0.0, 'dist': spans[l_span]})
            st.rerun()

    for idx, ld in enumerate(st.session_state.loads):
        c1, c2 = st.columns([5, 1])
        c1.code(f"Load {idx+1}: Span {ld['span_index']+1} | {ld['type']} | {ld['mag']/1000} kN")
        if c2.button("🗑️", key=f"del_{idx}"):
            st.session_state.loads.pop(idx); st.rerun()

# --- 2. ANALYSIS EXECUTION ---
st.divider()
if st.button("🚀 EXECUTE FULL ANALYSIS", type="primary", use_container_width=True):
    # Map support names to solver types
    formatted_supports = [{'id': s['Support ID'], 'type': s['Type']} for s in supports]
    sol = BeamSolver(spans, formatted_supports, st.session_state.loads, 2e11, b_m, h_m)
    st.session_state.analysis_results = sol.solve()

if st.session_state.analysis_results:
    df, reac, eq = st.session_state.analysis_results
    
    st.header("📊 PART I: Internal Forces & Reactions")
    design_view.draw_interactive_diagrams(df, reac, spans, pd.DataFrame(supports), st.session_state.loads)
    

[Image of shear force and bending moment diagrams for a continuous beam]


    # Analysis Verification Tables
    c_reac, c_check = st.columns([2, 1])
    with c_reac:
        st.subheader("Support Reactions")
        st.dataframe(reac, use_container_width=True)
    with c_check:
        st.subheader("Equilibrium Check")
        st.metric("Vertical Balance (ΣFy)", f"{abs(eq['l_fy'] - eq['r_fy']):.4f} N")
        st.metric("Moment Balance (ΣM@0)", f"{abs(eq['l_m0'] - eq['r_m0']):.4f} Nm")

    # --- 3. RC DESIGN & MANUAL OVERRIDE ---
    st.divider()
    st.header("🧱 PART II: Professional RC Design Sheet")
    
    col_sel1, col_sel2 = st.columns(2)
    db_m = col_sel1.selectbox("Main Reinforcement (DB)", [12, 16, 20, 25, 28, 32], index=2)
    db_s = col_sel2.selectbox("Stirrup Diameter (RB/DB)", [6, 9, 12, 16], index=1)

    cum_dist = [0] + list(np.cumsum(spans))
    for i in range(n_spans):
        with st.expander(f"SPAN {i+1} CALCULATION & REINFORCEMENT TUNING", expanded=True):
            s_df = df[(df['x'] >= cum_dist[i]) & (df['x'] <= cum_dist[i+1])]
            res = rc_design.design_span_detailed(s_df['moment'].max()/1000, s_df['moment'].min()/1000, s_df['shear'].abs().max()/1000, b_m, h_m, fc, fy, fyt, cover, db_m, db_s)
            
            calc_col, edit_col, view_col = st.columns([1.5, 1, 1.2])
            
            with calc_col:
                st.markdown("**Design Calculations (SDM):**")
                st.latex(rf"M_{{u,max}} = {max(abs(res['mu_pos']), abs(res['mu_neg'])):.2f} \text{{ kNm}}")
                st.latex(rf"k = \frac{{M_u}}{{\phi b d^2}} = {res['pos']['k']:.4f}")
                st.latex(rf"a = \frac{{A_s f_y}}{{0.85 f'_c b}} = {res['pos']['a']:.2f} \text{{ mm}}")
                st.latex(rf"\epsilon_t = {res['pos']['et']:.5f}")
                st.write(f"**Required As:** {res['pos']['as_req']:.1f} mm²")

            with edit_col:
                st.markdown("**Manual Adjustment:**")
                n_top = st.number_input(f"Top Bars (Span {i+1})", 2, 20, int(res['neg']['n']), key=f"t_{i}")
                n_bot = st.number_input(f"Bottom Bars (Span {i+1})", 2, 20, int(res['pos']['n']), key=f"b_{i}")
                stir_sp = st.number_input(f"Stirrup Spacing (mm)", 50, 400, int(res['spacing']), step=25, key=f"s_{i}")
                st.info(f"Stirrup: RB{db_s} @ {stir_sp} mm")

            with view_col:
                # Dynamic Cross-Section Plot
                fig_cs = go.Figure()
                fig_cs.add_shape(type="rect", x0=0, y0=0, x1=b_m, y1=h_m, fillcolor="rgba(150,150,150,0.1)", line=dict(color="Black", width=4))
                # Top Bars (Red)
                for j in range(n_top):
                    fig_cs.add_trace(go.Scatter(x=[(b_m/(n_top+1))*(j+1)], y=[h_m-(cover/1000)], mode='markers', marker=dict(color='Red', size=db_m)))
                # Bottom Bars (Blue)
                for j in range(n_bot):
                    fig_cs.add_trace(go.Scatter(x=[(b_m/(n_bot+1))*(j+1)], y=[cover/1000], mode='markers', marker=dict(color='Blue', size=db_m)))
                
                fig_cs.update_layout(width=250, height=300, showlegend=False, xaxis=dict(visible=False), yaxis=dict(visible=False), margin=dict(l=5,r=5,t=5,b=5))
                st.plotly_chart(fig_cs)
