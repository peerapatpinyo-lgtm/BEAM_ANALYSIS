import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from solver import BeamSolver
import rc_design
import design_view

st.set_page_config(page_title="Professional Structural Beam Analyzer", layout="wide")

# --- DATA PERSISTENCE ---
if 'loads' not in st.session_state: st.session_state.loads = []
if 'spans' not in st.session_state: st.session_state.spans = [5.0]
if 'supports' not in st.session_state: st.session_state.supports = [{'id': 0, 'type': 'Pin'}, {'id': 1, 'type': 'Roller'}]

# --- SIDEBAR: MATERIAL & STIFFNESS ---
with st.sidebar:
    st.title("🛠️ Global Settings")
    with st.expander("Material Properties", expanded=True):
        fc = st.number_input("f'c (Concrete MPa)", 28.0)
        fy = st.number_input("fy (Main Steel MPa)", 400.0)
        fyt = st.number_input("fyt (Stirrup MPa)", 240.0)
    
    with st.expander("Section Dimensions", expanded=True):
        b_m = st.number_input("Width b (m)", 0.30)
        h_m = st.number_input("Height h (m)", 0.50)
        
    with st.expander("Stiffness (I) Control"):
        ig = (b_m * h_m**3) / 12
        i_mode = st.radio("Inertia Type", ["Gross (Ig)", "Manual Override"])
        I_actual = st.number_input("I Value (m⁴)", value=ig, format="%.6e") if i_mode == "Manual Override" else ig
        st.info(f"Ig: {ig:.6e} m⁴")

st.title("🏗️ Beam Analysis & Design Studio")

# --- TABBED INPUT INTERFACE ---
tab_geo, tab_load, tab_settings = st.tabs(["📏 Geometry", "📥 Loads", "⚙️ Design Parameters"])

with tab_geo:
    c1, c2 = st.columns([1, 2])
    with c1:
        n_spans = st.number_input("Number of Spans", 1, 10, len(st.session_state.spans))
        if n_spans != len(st.session_state.spans):
            st.session_state.spans = [5.0] * n_spans
            st.session_state.supports = [{'id': i, 'type': 'Pin' if i==0 else 'None'} for i in range(n_spans+1)]
            st.rerun()
        for i in range(n_spans):
            st.session_state.spans[i] = st.number_input(f"Span {i+1} Length (m)", 0.1, 25.0, float(st.session_state.spans[i]))
    with c2:
        df_sup = pd.DataFrame([{'id': i, 'type': next((s['type'] for s in st.session_state.supports if s['id'] == i), 'None')} for i in range(n_spans+1)])
        ed_sup = st.data_editor(df_sup, column_config={"type": st.column_config.SelectboxColumn("Support", options=["None", "Pin", "Roller", "Fixed"])}, hide_index=True, use_container_width=True)
        st.session_state.supports = ed_sup.to_dict('records')

with tab_load:
    cl1, cl2 = st.columns([1, 2])
    with cl1:
        st.subheader("Add Load")
        l_idx = st.selectbox("Select Span", range(n_spans), format_func=lambda x: f"Span {x+1}")
        l_type = st.selectbox("Load Type", ["Point Load (P)", "Uniform Load (U)", "Moment (M)"])
        l_mag = st.number_input("Magnitude (kN or kNm)", 10.0)
        l_pos = st.number_input("Position from Left (m)", 0.0)
        l_dist = st.number_input("Load Length (m)", 0.0) if "U" in l_type else 0.0
        if st.button("➕ Add Load", use_container_width=True):
            st.session_state.loads.append({'id': len(st.session_state.loads), 'span': l_idx, 'type': l_type[0], 'mag': l_mag*1000, 'x': l_pos, 'dist': l_dist})
            st.rerun()
    with cl2:
        st.subheader("Load Inventory")
        if st.session_state.loads:
            for i, ld in enumerate(st.session_state.loads):
                col_txt, col_btn = st.columns([4, 1])
                col_txt.code(f"{ld['type']} | {ld['mag']/1000} kN | Span {ld['span']+1} @ {ld['x']}m")
                if col_btn.button("🗑️", key=f"del_{i}"):
                    st.session_state.loads.pop(i)
                    st.rerun()
        else: st.info("No loads added yet.")

with tab_settings:
    st.subheader("Detailed RC Design Settings")
    cs1, cs2, cs3 = st.columns(3)
    cover = cs1.number_input("Concrete Cover (mm)", 30)
    db_main = cs2.selectbox("Main Bar Size (mm)", [12, 16, 20, 25, 28], index=2)
    db_stirrup = cs3.selectbox("Stirrup Size (mm)", [6, 9, 12], index=1)

# --- ANALYSIS & RESULTS ---
st.divider()
if st.button("🚀 RUN FULL ANALYSIS & DESIGN", type="primary", use_container_width=True):
    # Convert loads for solver
    solver_loads = [{'span_index': l['span'], 'type': l['type'], 'mag': l['mag'], 'x': l['x'], 'dist': l['dist']} for l in st.session_state.loads]
    sol = BeamSolver(st.session_state.spans, st.session_state.supports, solver_loads, 2e11, b_m, h_m, I_actual)
    df, reac, eq = sol.solve()
    
    if not df.empty:
        # SECTION 1: STRUCTURAL ANALYSIS (Always persistent)
        st.header("1. Structural Analysis Results")
        design_view.draw_interactive_diagrams(df, reac, st.session_state.spans, pd.DataFrame(st.session_state.supports), solver_loads)
        
        c_reac, c_equi = st.columns([2, 1])
        with c_reac:
            st.subheader("Support Reactions")
            st.dataframe(reac, use_container_width=True, hide_index=True)
        with c_equi:
            st.subheader("Equilibrium Check")
            st.metric("Total Fy (kN)", f"{eq['l_fy']/1000:.2f}", delta=f"Err: {abs(eq['l_fy']-eq['r_fy']):.4f} N")
            st.metric("Total M@0 (kNm)", f"{eq['l_m0']/1000:.2f}", delta=f"Err: {abs(eq['l_m0']-eq['r_m0']):.4f} Nm")

        # SECTION 2: RC DESIGN & OPTIMIZATION
        st.divider()
        st.header("2. Professional RC Design & Optimization")
        mu_pos, mu_neg = df['moment'].max()/1000, df['moment'].min()/1000
        vu_max = df['shear'].abs().max()/1000
        
        rc = rc_design.calculate_advanced_rc(mu_pos, mu_neg, vu_max, b_m, h_m, fc, fy, fyt, cover, db_main, db_stirrup)
        
        # Optimization Alert
        if rc['opt_status'] == "Success": st.success(rc['opt_msg'])
        elif rc['opt_status'] == "Warning": st.warning(rc['opt_msg'])
        else: st.error(rc['opt_msg'])

        col_rep, col_draw = st.columns([1, 1.2])
        with col_rep:
            st.subheader("Design Calculations")
            st.write(f"**Flexure:** Mu(+) = {mu_pos:.2f} kNm | Mu(-) = {mu_neg:.2f} kNm")
            st.write(f"**Shear:** Vu = {vu_max:.2f} kN")
            st.write(f"**Stiffness Ratio:** {(I_actual/ig)*100:.1f}% of Gross Section")
            
            # Show summary
            st.info(f"**Proposed Reinforcement:**\n\n- Top: {rc['n_top']} x DB{db_main}\n\n- Bottom: {rc['n_bot']} x DB{db_main}\n\n- Stirrups: RB{db_stirrup} @ {int(rc['spacing'])} mm")

        with col_draw:
            st.subheader("Cross-Section Drawing")
            fig = go.Figure()
            # Concrete Outer
            fig.add_shape(type="rect", x0=0, y0=0, x1=rc['b'], y1=rc['h'], line=dict(color="Black", width=4), fillcolor="rgba(200,200,200,0.3)")
            # Stirrup
            s_off = cover + (db_stirrup/2)
            fig.add_shape(type="rect", x0=s_off, y0=s_off, x1=rc['b']-s_off, y1=rc['h']-s_off, line=dict(color="Gray", width=2, dash="dash"))
            # Top Bars
            for i in range(rc['n_top']):
                x_pos = s_off + (i * (rc['b'] - 2*s_off) / (rc['n_top'] - 1)) if rc['n_top'] > 1 else rc['b']/2
                fig.add_trace(go.Scatter(x=[x_pos], y=[rc['h']-s_off-db_main/2], mode='markers', marker=dict(size=db_main, color='DarkRed')))
            # Bottom Bars
            for i in range(rc['n_bot']):
                x_pos = s_off + (i * (rc['b'] - 2*s_off) / (rc['n_bot'] - 1)) if rc['n_bot'] > 1 else rc['b']/2
                fig.add_trace(go.Scatter(x=[x_pos], y=[s_off+db_main/2], mode='markers', marker=dict(size=db_main, color='DarkBlue')))
            
            fig.update_layout(width=350, height=450, showlegend=False, xaxis=dict(visible=False), yaxis=dict(visible=False))
            st.plotly_chart(fig)

    else:
        st.error("Matrix Singular: Check if the beam is properly supported.")
