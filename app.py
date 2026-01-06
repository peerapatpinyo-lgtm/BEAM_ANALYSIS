import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from solver import BeamSolver
import rc_design
import design_view

st.set_page_config(page_title="World-Class Beam Studio", layout="wide")

# Persistent State Management
if 'loads' not in st.session_state: st.session_state.loads = []
if 'spans' not in st.session_state: st.session_state.spans = [5.0]
if 'supports' not in st.session_state: 
    st.session_state.supports = [{'id': 0, 'type': 'Pin'}, {'id': 1, 'type': 'Roller'}]

# --- SIDEBAR: GLOBAL SPECS ---
with st.sidebar:
    st.title("🏆 Project Parameters")
    fc = st.number_input("Concrete f'c (MPa)", 28.0)
    b_m = st.number_input("Section Width b (m)", 0.30)
    h_m = st.number_input("Section Height h (m)", 0.50)
    st.divider()
    n_spans = st.number_input("Total Spans", 1, 10, len(st.session_state.spans))
    if n_spans != len(st.session_state.spans):
        st.session_state.spans = [5.0] * n_spans
        st.session_state.supports = [{'id': i, 'type': 'Pin' if i==0 else 'Roller'} for i in range(n_spans+1)]
        st.rerun()

st.title("🏗️ Professional Beam Structural Lab")

# --- IMPROVED LOAD & GEOMETRY SECTION ---
t_geo, t_load = st.tabs(["⚓ Geometry & Supports", "📥 Advanced Load Manager"])

with t_geo:
    st.subheader("Span Lengths & Supports")
    cols = st.columns(n_spans)
    for i in range(n_spans):
        st.session_state.spans[i] = cols[i].number_input(f"L{i+1} (m)", 0.1, 25.0, float(st.session_state.spans[i]))
    
    df_sup = pd.DataFrame([{'id': i, 'type': next((s['type'] for s in st.session_state.supports if s['id'] == i), 'None')} for i in range(n_spans+1)])
    ed_sup = st.data_editor(df_sup, column_config={"type": st.column_config.SelectboxColumn("Support Type", options=["None", "Pin", "Roller", "Fixed"])}, hide_index=True, use_container_width=True)
    st.session_state.supports = ed_sup.to_dict('records')

with t_load:
    st.subheader("Load Definition")
    # Clean horizontal input for better UX
    l_col1, l_col2, l_col3, l_col4, l_col5 = st.columns([1, 1, 1, 1, 1])
    l_span = l_col1.selectbox("Span", range(n_spans), format_func=lambda x: f"Span {x+1}")
    l_type = l_col2.selectbox("Load Type", ["Point (P)", "Uniform (U)", "Moment (M)"])
    l_mag = l_col3.number_input("Magnitude (kN/m)", 15.0)
    l_x = l_col4.number_input("Start x (m)", 0.0)
    l_len = l_col5.number_input("End x / Length (m)", st.session_state.spans[l_span]) if "U" in l_type else 0.0
    
    if st.button("➕ Add Load to Span", use_container_width=True):
        st.session_state.loads.append({'span_index': l_span, 'type': l_type[0], 'mag': l_mag*1000, 'x': l_x, 'dist': l_len})
        st.rerun()

    if st.session_state.loads:
        st.write("**Active Loads Table**")
        for i, ld in enumerate(st.session_state.loads):
            c1, c2, c3 = st.columns([1, 4, 1])
            c1.markdown(f"**#{i+1}**")
            c2.info(f"Span {ld['span_index']+1} | {ld['type']} | {ld['mag']/1000} kN | x={ld['x']}m")
            if c3.button("🗑️", key=f"del_{i}"):
                st.session_state.loads.pop(i); st.rerun()

# --- EXECUTION & ANALYSIS ---
st.divider()
if st.button("🚀 EXECUTE FULL ANALYSIS", type="primary", use_container_width=True) or 'df_result' in st.session_state:
    sol = BeamSolver(st.session_state.spans, st.session_state.supports, st.session_state.loads, 2e11, b_m, h_m)
    df, reac, eq = sol.solve()
    st.session_state.df_result = df 

    if not df.empty:
        st.header("📊 PART I: Analysis Diagrams")
        design_view.draw_interactive_diagrams(df, reac, st.session_state.spans, pd.DataFrame(st.session_state.supports), st.session_state.loads)
        
        # --- RC DESIGN AREA (DYNAMIC INPUTS HERE) ---
        st.divider()
        st.header("🧱 PART II: Professional RC Design")
        
        with st.container(border=True):
            st.subheader("🛠️ Design Tuning")
            rc1, rc2, rc3, rc4, rc5 = st.columns(5)
            fy = rc1.number_input("fy Main (MPa)", 400)
            fyt = rc2.number_input("fy Stirrup (MPa)", 240)
            cover = rc3.slider("Concrete Cover (mm)", 20, 50, 30)
            db_m = rc4.selectbox("Main Bar Size", [12, 16, 20, 25])
            db_s = rc5.selectbox("Stirrup Size", [6, 9, 12], index=1)

        # Calculation per Span
        cum_dist = [0] + list(np.cumsum(st.session_state.spans))
        span_designs = []
        for i in range(len(st.session_state.spans)):
            span_df = df[(df['x'] >= cum_dist[i]) & (df['x'] <= cum_dist[i+1])]
            m_pos = span_df['moment'].max()/1000
            m_neg = span_df['moment'].min()/1000
            v_max = span_df['shear'].abs().max()/1000
            # PASS ALL PARAMETERS
            span_designs.append(rc_design.design_section(m_pos, m_neg, v_max, b_m, h_m, fc, fy, fyt, cover, db_m, db_s))

        # --- LONGITUDINAL SECTION ---
        st.subheader("Longitudinal Reinforcement Detail")
        
        fig_long = go.Figure()
        fig_long.add_shape(type="rect", x0=0, y0=0, x1=cum_dist[-1], y1=h_m, line=dict(color="Black", width=3), fillcolor="rgba(100,100,100,0.1)")
        
        c_off = cover/1000
        for i, des in enumerate(span_designs):
            # Top Rebar
            fig_long.add_trace(go.Scatter(x=[cum_dist[i]+0.05, cum_dist[i+1]-0.05], y=[h_m-c_off, h_m-c_off], 
                                          mode='lines+text', line=dict(color='Red', width=des['n_top']*2), 
                                          text=[f"{des['n_top']}-DB{db_m}"], textposition="top center"))
            # Bottom Rebar
            fig_long.add_trace(go.Scatter(x=[cum_dist[i]+0.05, cum_dist[i+1]-0.05], y=[c_off, c_off], 
                                          mode='lines+text', line=dict(color='Blue', width=des['n_bot']*2), 
                                          text=[f"{des['n_bot']}-DB{db_m}"], textposition="bottom center"))
            # Stirrups
            for sx in np.linspace(cum_dist[i], cum_dist[i+1], 15):
                fig_long.add_shape(type="line", x0=sx, y0=c_off, x1=sx, y1=h_m-c_off, line=dict(color="lightgray", width=1))

        fig_long.update_layout(showlegend=False, height=400, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_long, use_container_width=True)

        # --- OPTIMIZATION REPORT ---
        st.subheader("📋 Optimization Summary")
        opt_cols = st.columns(len(span_designs))
        for i, des in enumerate(span_designs):
            # FIXED: Accessing mu_pos/neg from the dict returned by the function
            mu_abs_max = max(abs(des['mu_pos']), abs(des['mu_neg']))
            k_val = (mu_abs_max * 1e6) / (0.9 * des['b'] * des['d']**2)
            k_limit = 0.15 * fc
            
            with opt_cols[i]:
                st.write(f"**Span {i+1}**")
                if k_val > k_limit:
                    st.error(f"⚠️ Section too small\n(k={k_val:.2f} > {k_limit:.2f})")
                else:
                    st.success(f"✅ Optimized\n(k={k_val:.2f} < {k_limit:.2f})")
                st.caption(f"Stirrups: RB{db_s} @ {des['spacing']} mm")
