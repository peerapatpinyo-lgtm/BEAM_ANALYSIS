import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from solver import BeamSolver
import rc_design
import design_view

st.set_page_config(page_title="Structural Lab Pro", layout="wide")

# Persistent state for Loads
if 'loads' not in st.session_state: st.session_state.loads = []
if 'spans' not in st.session_state: st.session_state.spans = [5.0]

# --- SIDEBAR: GEOMETRY & GLOBAL MATERIALS ---
with st.sidebar:
    st.title("🏆 Project Specs")
    with st.expander("Concrete & Stiffness", expanded=True):
        fc = st.number_input("f'c (MPa)", 28.0)
        b_m = st.number_input("Width b (m)", 0.30)
        h_m = st.number_input("Height h (m)", 0.50)
        ig = (b_m * h_m**3) / 12
        st.info(f"Gross Inertia (Ig): {ig:.6e} m⁴")
    
    with st.expander("Span & Support Configuration", expanded=True):
        n_spans = st.number_input("Number of Spans", 1, 10, len(st.session_state.spans))
        if n_spans != len(st.session_state.spans):
            st.session_state.spans = [5.0] * n_spans
            st.session_state.supports = [{'id': i, 'type': 'Pin' if i==0 else 'None'} for i in range(n_spans+1)]
            st.rerun()
        for i in range(n_spans):
            st.session_state.spans[i] = st.number_input(f"L{i+1} (m)", 0.1, 20.0, float(st.session_state.spans[i]))

st.title("🏗️ Beam Analysis & Design Studio")

# --- PART 1: LOAD MANAGEMENT ---
st.header("1. Load Management")
c_in, c_list = st.columns([1, 2])
with c_in:
    with st.container(border=True):
        l_span = st.selectbox("Target Span", range(n_spans), format_func=lambda x: f"Span {x+1}")
        l_type = st.selectbox("Type", ["P (Point)", "U (Uniform)", "M (Moment)"])
        l_mag = st.number_input("Magnitude (kN or kNm)", 10.0)
        l_x = st.number_input("Position x (m)", 0.0)
        l_d = st.number_input("Length (m)", 0.0) if "U" in l_type else 0.0
        if st.button("➕ Add Load", use_container_width=True):
            st.session_state.loads.append({'span': l_span, 'type': l_type[0], 'mag': l_mag*1000, 'x': l_x, 'dist': l_d})
            st.rerun()
with c_list:
    if not st.session_state.loads: st.info("No loads applied.")
    for i, ld in enumerate(st.session_state.loads):
        cols = st.columns([5, 1])
        cols[0].code(f"Load {i+1}: {ld['type']} | {ld['mag']/1000}kN | Span {ld['span']+1} at {ld['x']}m")
        if cols[1].button("🗑️", key=f"del_{i}"):
            st.session_state.loads.pop(i); st.rerun()

st.divider()

# --- ANALYSIS EXECUTION ---
if st.button("🚀 EXECUTE STRUCTURAL ANALYSIS", type="primary", use_container_width=True):
    solver_loads = [{'span_index': l['span'], 'type': l['type'], 'mag': l['mag'], 'x': l['x'], 'dist': l['dist']} for l in st.session_state.loads]
    # Static supports for demo - can be editted in sidebar in real use
    supports = [{'id': i, 'type': 'Pin' if i==0 else 'Roller'} for i in range(n_spans+1)]
    
    sol = BeamSolver(st.session_state.spans, supports, solver_loads, 2e11, b_m, h_m, ig)
    df, reac, eq = sol.solve()
    
    if not df.empty:
        # --- ANALYSIS RESULTS (STAY VISIBLE) ---
        st.header("📊 PART I: Structural Analysis Results")
        design_view.draw_interactive_diagrams(df, reac, st.session_state.spans, pd.DataFrame(supports), solver_loads)
        
        # --- PART 2: DESIGN SECTION (WITH INPUTS) ---
        st.divider()
        st.header("🧱 PART II: Professional RC Design")
        
        # Move Design Inputs here as requested
        with st.container(border=True):
            st.subheader("Interactive Design Tuning")
            di1, di2, di3, di4, di5 = st.columns(5)
            fy = di1.number_input("Main Steel fy (MPa)", 400)
            fyt = di2.number_input("Stirrup fyt (MPa)", 240)
            cover = di3.slider("Covering (mm)", 20, 75, 30)
            db_m = di4.selectbox("Main Bar (DB)", [12, 16, 20, 25, 28], index=2)
            db_s = di5.selectbox("Stirrup (RB/DB)", [6, 9, 12], index=1)

        mu_pos, mu_neg = df['moment'].max()/1000, df['moment'].min()/1000
        vu_max = df['shear'].abs().max()/1000
        
        # Re-calculate RC design based on updated tuning
        rc = rc_design.calculate_advanced_rc(mu_pos, mu_neg, vu_max, b_m, h_m, fc, fy, fyt, cover, db_m, db_s)
        
        st.markdown(f"### Status: :{rc['opt_color']}[{rc['opt_status']}]")

        c_rep, c_viz = st.columns([1, 1.2])
        with c_rep:
            st.write("**Reinforcement Summary:**")
            st.info(f"Top Reinforcement: **{rc['n_top']} x DB{db_m}**")
            st.success(f"Bottom Reinforcement: **{rc['n_bot']} x DB{db_m}**")
            st.warning(f"Shear Reinforcement: **RB{db_s} @ {int(rc['spacing'])} mm**")
            st.write(f"Effective Depth (d): {rc['d']:.1f} mm")

        with c_viz:
            st.subheader("Typical Cross-Section")
            
            fig = go.Figure()
            # Concrete
            fig.add_shape(type="rect", x0=0, y0=0, x1=rc['b'], y1=rc['h'], line=dict(color="Black", width=3), fillcolor="rgba(128,128,128,0.1)")
            # Stirrup line
            s_off = cover + db_s/2
            fig.add_shape(type="rect", x0=s_off, y0=s_off, x1=rc['b']-s_off, y1=rc['h']-s_off, line=dict(color="Gray", width=2, dash="dash"))
            # Top Bars
            for i in range(rc['n_top']):
                x = s_off + (i * (rc['b']-2*s_off)/(rc['n_top']-1)) if rc['n_top'] > 1 else rc['b']/2
                fig.add_trace(go.Scatter(x=[x], y=[rc['h']-s_off-db_m/2], mode='markers', marker=dict(size=db_m, color='DarkRed')))
            # Bottom Bars
            for i in range(rc['n_bot']):
                x = s_off + (i * (rc['b']-2*s_off)/(rc['n_bot']-1)) if rc['n_bot'] > 1 else rc['b']/2
                fig.add_trace(go.Scatter(x=[x], y=[s_off+db_m/2], mode='markers', marker=dict(size=db_m, color='DarkBlue')))
            
            fig.update_layout(width=400, height=500, xaxis=dict(visible=False), yaxis=dict(visible=False), showlegend=False)
            st.plotly_chart(fig)
