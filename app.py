import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from solver import BeamSolver
import rc_design
import design_view

st.set_page_config(page_title="Professional Structural Beam Pro", layout="wide")

# Initialize Session States
if 'loads' not in st.session_state: st.session_state.loads = []
if 'analysis_done' not in st.session_state: st.session_state.analysis_done = False

with st.sidebar:
    st.header("⚙️ Global Parameters")
    fc = st.number_input("Concrete f'c (MPa)", 28.0)
    b_m = st.number_input("Width b (m)", 0.30)
    h_m = st.number_input("Height h (m)", 0.50)
    n_spans = st.number_input("Total Spans", 1, 10, 1)
    spans = [st.number_input(f"L{i+1} (m)", 0.1, 25.0, 5.0, key=f"s_{i}") for i in range(n_spans)]

st.title("🏗️ Professional Structural Analysis & Design")

# --- 1. LOAD MANAGER ---
st.header("1. Load Manager")
with st.container(border=True):
    c1, c2, c3, c4, c5 = st.columns([1,1,1,1,1])
    l_span = c1.selectbox("Span", range(n_spans), format_func=lambda x: f"Span {x+1}")
    l_type = c2.selectbox("Type", ["Point (P)", "Uniform (U)"])
    l_mag = c3.number_input("Magnitude (kN)", 20.0)
    l_x = c4.number_input("Pos x (m)", 0.0)
    l_dist = c5.number_input("Length (m)", spans[l_span]) if "U" in l_type else 0.0
    
    if st.button("➕ Add Load", use_container_width=True):
        st.session_state.loads.append({'span_index': l_span, 'type': l_type[0], 'mag': l_mag*1000, 'x': l_x, 'dist': l_dist})
        st.rerun()

if st.session_state.loads:
    for i, ld in enumerate(st.session_state.loads):
        cols = st.columns([5, 1])
        cols[0].info(f"Load #{i+1}: Span {ld['span_index']+1} | {ld['type']} | {ld['mag']/1000} kN")
        if cols[1].button("🗑️", key=f"del_{i}"): st.session_state.loads.pop(i); st.rerun()

# --- 2. EXECUTION & ANALYSIS RESULTS ---
st.divider()
if st.button("🚀 RUN ANALYSIS", type="primary", use_container_width=True):
    st.session_state.analysis_done = True

if st.session_state.analysis_done:
    # Always keep supports as standard for this version
    supports = [{'id': i, 'type': 'Pin' if i==0 else 'Roller'} for i in range(n_spans+1)]
    sol = BeamSolver(spans, supports, st.session_state.loads, 2e11, b_m, h_m)
    df, reac, eq = sol.solve()
    
    if not df.empty:
        st.header("📊 PART I: Analysis Results (SDM)")
        st.info("**Methodology:** Finite Element Method (Linear Elastic) for Internal Forces.")
        
        # Graphs
        design_view.draw_interactive_diagrams(df, reac, spans, pd.DataFrame(supports), st.session_state.loads)
        

        # Max/Min Tables & Equation Check (Don't Hide This)
        c_reac, c_eq = st.columns([2, 1])
        with c_reac:
            st.subheader("Support Reactions & Force Summary")
            st.dataframe(reac, use_container_width=True)
            st.write(f"**Max Moment:** {df['moment'].max()/1000:.2f} kNm | **Min Moment:** {df['moment'].min()/1000:.2f} kNm")
        with c_eq:
            st.subheader("Equation Check (ΣF=0)")
            st.metric("Vertical Balance (N)", f"{abs(eq['l_fy'] - eq['r_fy']):.4f}")
            st.metric("Moment Balance (Nm)", f"{abs(eq['l_m0'] - eq['r_m0']):.4f}")

        # --- 3. PROFESSIONAL RC DESIGN ---
        st.divider()
        st.header("🧱 PART II: RC Design (Strength Design Method)")
        st.caption("Design Method: SDM (ACI 318-19 / EIT 1008-38) | Load Factor: 1.0 (Service shown as factored)")
        
        with st.container(border=True):
            st.subheader("Design Parameters")
            dc1, dc2, dc3, dc4, dc5 = st.columns(5)
            fy = dc1.number_input("Main Steel fy (MPa)", 400)
            fyt = dc2.number_input("Stirrup fyt (MPa)", 240)
            cover = dc3.number_input("Covering (mm)", 30)
            db_m = dc4.selectbox("Main DB", [12, 16, 20, 25])
            db_s = dc5.selectbox("Stirrup RB", [6, 9, 12], index=1)

        cum_dist = [0] + list(np.cumsum(spans))
        span_data = []
        for i in range(n_spans):
            s_df = df[(df['x'] >= cum_dist[i]) & (df['x'] <= cum_dist[i+1])]
            res = rc_design.design_section_comprehensive(s_df['moment'].max()/1000, s_df['moment'].min()/1000, s_df['shear'].abs().max()/1000, b_m, h_m, fc, fy, fyt, cover, db_m, db_s)
            span_data.append(res)

        # Longitudinal Drawing
        fig_long = go.Figure()
        fig_long.add_shape(type="rect", x0=0, y0=0, x1=cum_dist[-1], y1=h_m, line=dict(color="Black", width=4), fillcolor="rgba(100,100,100,0.1)")
        for i, res in enumerate(span_data):
            x0, x1 = cum_dist[i], cum_dist[i+1]
            fig_long.add_trace(go.Scatter(x=[x0, x1], y=[h_m-(cover/1000), h_m-(cover/1000)], mode='lines+text', line=dict(color='Red', width=res['neg']['n']), text=[f"{res['neg']['n']}-DB{db_m}"]))
            fig_long.add_trace(go.Scatter(x=[x0, x1], y=[cover/1000, cover/1000], mode='lines+text', line=dict(color='Blue', width=res['pos']['n']), text=[f"{res['pos']['n']}-DB{db_m}"], textposition="bottom center"))
        st.plotly_chart(fig_long, use_container_width=True)

        # Detailed Calculations
        st.header("📝 Detailed Design Sheets")
        for i, res in enumerate(span_data):
            with st.expander(f"SPAN {i+1} CALCULATION DETAILS", expanded=True):
                c_txt, c_cs = st.columns([2, 1])
                with c_txt:
                    st.markdown(f"**Flexural Design (Positive Moment)**")
                    st.latex(rf"M_u = {res['mu_pos']:.2f} \text{{ kNm}} \rightarrow k = {res['pos']['k']:.4f}")
                    st.latex(rf"a = \frac{{A_s f_y}}{{0.85 f'_c b}} = {res['pos']['a']:.2f} \text{{ mm}}")
                    st.markdown(f"**Flexural Design (Negative Moment)**")
                    st.latex(rf"M_u = {res['mu_neg']:.2f} \text{{ kNm}} \rightarrow A_{{s,req}} = {res['neg']['as_req']:.1f} \text{{ mm}}^2")
                    st.markdown(f"**Shear Design**")
                    st.latex(rf"V_u = {res['vu']:.2f} \text{{ kN}}, \phi V_c = {0.75*res['vc']:.2f} \text{{ kN}}")
                    st.success(f"Result: Top {res['neg']['n']}xDB{db_m} | Bot {res['pos']['n']}xDB{db_m} | Stirrup RB{db_s}@{res['spacing']}mm")
                with c_cs:
                    # Individual Cross Section per Span
                    fig_cs = go.Figure()
                    fig_cs.add_shape(type="rect", x0=0, y0=0, x1=b_m, y1=h_m, fillcolor="rgba(200,200,200,0.2)")
                    for j in range(res['neg']['n']): fig_cs.add_trace(go.Scatter(x=[(b_m/(res['neg']['n']+1))*(j+1)], y=[h_m-(cover/1000)], mode='markers', marker=dict(color='Red')))
                    for j in range(res['pos']['n']): fig_cs.add_trace(go.Scatter(x=[(b_m/(res['pos']['n']+1))*(j+1)], y=[cover/1000], mode='markers', marker=dict(color='Blue')))
                    fig_cs.update_layout(width=200, height=250, xaxis=dict(visible=False), yaxis=dict(visible=False), showlegend=False)
                    st.plotly_chart(fig_cs)
