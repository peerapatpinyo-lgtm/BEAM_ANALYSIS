import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from solver import BeamSolver
import rc_design
import design_view

st.set_page_config(page_title="Professional Structural Designer", layout="wide")

if 'loads' not in st.session_state: st.session_state.loads = []
if 'spans' not in st.session_state: st.session_state.spans = [5.0]
if 'supports' not in st.session_state: st.session_state.supports = [{'id': 0, 'type': 'Pin'}, {'id': 1, 'type': 'Roller'}]

# --- Sidebar: Global Specs ---
with st.sidebar:
    st.header("🏆 Project Specs")
    fc = st.number_input("Concrete f'c (MPa)", 28.0)
    b_m = st.number_input("Section Width b (m)", 0.30)
    h_m = st.number_input("Section Height h (m)", 0.50)
    st.divider()
    n_spans = st.number_input("Number of Spans", 1, 10, len(st.session_state.spans))
    if n_spans != len(st.session_state.spans):
        st.session_state.spans = [5.0] * n_spans
        st.session_state.supports = [{'id': i, 'type': 'Pin' if i==0 else 'Roller'} for i in range(n_spans+1)]
        st.rerun()

st.title("🏗️ Beam Structural Lab & Professional Report")

# --- Load & Geometry (Simplified Manager) ---
tab1, tab2 = st.tabs(["📏 Geometry & Supports", "📥 Load Manager"])
with tab1:
    cols = st.columns(n_spans)
    for i in range(n_spans):
        st.session_state.spans[i] = cols[i].number_input(f"L{i+1} (m)", 0.1, 25.0, float(st.session_state.spans[i]))
    df_sup = pd.DataFrame([{'id': i, 'type': next((s['type'] for s in st.session_state.supports if s['id'] == i), 'None')} for i in range(n_spans+1)])
    ed_sup = st.data_editor(df_sup, use_container_width=True)
    st.session_state.supports = ed_sup.to_dict('records')

with tab2:
    lc1, lc2, lc3, lc4 = st.columns([1,1,1,1])
    l_span = lc1.selectbox("Span", range(n_spans))
    l_type = lc2.selectbox("Type", ["P","U"])
    l_mag = lc3.number_input("Mag (kN)", 15.0)
    if lc4.button("➕ Add", use_container_width=True):
        st.session_state.loads.append({'span_index': l_span, 'type': l_type, 'mag': l_mag*1000, 'x': 0.0, 'dist': st.session_state.spans[l_span]})
        st.rerun()
    for i, ld in enumerate(st.session_state.loads):
        c1, c2 = st.columns([5, 1])
        c1.info(f"Span {ld['span_index']+1} | {ld['type']} | {ld['mag']/1000} kN")
        if c2.button("🗑️", key=f"del_{i}"): st.session_state.loads.pop(i); st.rerun()

# --- Execution ---
st.divider()
if st.button("🚀 EXECUTE ANALYSIS & DESIGN", type="primary", use_container_width=True) or 'df_result' in st.session_state:
    sol = BeamSolver(st.session_state.spans, st.session_state.supports, st.session_state.loads, 2e11, b_m, h_m)
    df, reac, eq = sol.solve()
    st.session_state.df_result = df 

    if not df.empty:
        # Part 1: Analysis
        st.header("📊 PART I: Analysis Results")
        design_view.draw_interactive_diagrams(df, reac, st.session_state.spans, pd.DataFrame(st.session_state.supports), st.session_state.loads)

        # Part 2: Design Tuning
        st.divider()
        st.header("🧱 PART II: Professional RC Design")
        with st.container(border=True):
            dc1, dc2, dc3, dc4, dc5 = st.columns(5)
            fy = dc1.number_input("fy (MPa)", 400)
            fyt = dc2.number_input("fyt (MPa)", 240)
            cover = dc3.number_input("Cover (mm)", 30) # เปลี่ยนเป็นแบบกรอกค่า
            db_m = dc4.selectbox("Main (DB)", [12, 16, 20, 25])
            db_s = dc5.selectbox("Stirrup (RB)", [6, 9, 12], index=1)

        cum_dist = [0] + list(np.cumsum(st.session_state.spans))
        span_results = []
        for i in range(len(st.session_state.spans)):
            s_df = df[(df['x'] >= cum_dist[i]) & (df['x'] <= cum_dist[i+1])]
            span_results.append(rc_design.design_section_detailed(s_df['moment'].max()/1000, s_df['moment'].min()/1000, s_df['shear'].abs().max()/1000, b_m, h_m, fc, fy, fyt, cover, db_m, db_s))

        # --- 2.1 Longitudinal Detail (High Quality) ---
        st.subheader("Longitudinal Reinforcement Detail")
        fig_long = go.Figure()
        # Beam Outline
        fig_long.add_shape(type="rect", x0=0, y0=0, x1=cum_dist[-1], y1=h_m, line=dict(color="Black", width=4), fillcolor="rgba(120,120,120,0.1)")
        
        c_m = cover/1000
        for i, res in enumerate(span_results):
            x0, x1 = cum_dist[i], cum_dist[i+1]
            # Bars
            fig_long.add_trace(go.Scatter(x=[x0+0.05, x1-0.05], y=[h_m-c_m, h_m-c_m], mode='lines+text', line=dict(color='Red', width=res['neg']['n']*1.5), text=[f"{res['neg']['n']}-DB{db_m}"], textposition="top center"))
            fig_long.add_trace(go.Scatter(x=[x0+0.05, x1-0.05], y=[c_m, c_m], mode='lines+text', line=dict(color='Blue', width=res['pos']['n']*1.5), text=[f"{res['pos']['n']}-DB{db_m}"], textposition="bottom center"))
            # Stirrups Lines
            for sx in np.linspace(x0, x1, 20):
                fig_long.add_shape(type="line", x0=sx, y0=c_m, x1=sx, y1=h_m-c_m, line=dict(color="lightgray", width=0.5))
        
        fig_long.update_layout(height=350, showlegend=False, margin=dict(l=0,r=0,t=40,b=0))
        st.plotly_chart(fig_long, use_container_width=True)

        # --- 2.2 Cross Sections & Detailed Calculation per Span ---
        st.header("📋 Detailed Design Report & Cross-Sections")
        for i, res in enumerate(span_results):
            with st.expander(f"SPAN {i+1} DETAILS (L={st.session_state.spans[i]}m)", expanded=(i==0)):
                col_calc, col_img = st.columns([2, 1])
                with col_calc:
                    st.write("**Calculation Step-by-Step:**")
                    st.latex(rf"M_{{u,max}} = {max(abs(res['mu_pos']), abs(res['mu_neg'])):.2f} \text{{ kNm}}")
                    st.latex(rf"k = \frac{{M_u}}{{\phi b d^2}} = {res['pos']['k']:.4f}")
                    st.latex(rf"\rho = {res['pos']['rho']:.5f} \rightarrow A_s = {res['pos']['as_req']:.1f} \text{{ mm}}^2")
                    st.info(f"Result: Top {res['neg']['n']}xDB{db_m} | Bot {res['pos']['n']}xDB{db_m} | Stirrup RB{db_s}@{res['spacing']}mm")
                
                with col_img:
                    # Draw Cross Section for this span
                    fig_cs = go.Figure()
                    fig_cs.add_shape(type="rect", x0=0, y0=0, x1=b_m*1000, y1=h_m*1000, line=dict(color="Black", width=3))
                    # Bars
                    for j in range(res['neg']['n']): 
                        fig_cs.add_trace(go.Scatter(x=[(b_m*1000/(res['neg']['n']+1))*(j+1)], y=[h_m*1000-cover], mode='markers', marker=dict(size=db_m, color='Red')))
                    for j in range(res['pos']['n']): 
                        fig_cs.add_trace(go.Scatter(x=[(b_m*1000/(res['pos']['n']+1))*(j+1)], y=[cover], mode='markers', marker=dict(size=db_m, color='Blue')))
                    fig_cs.update_layout(width=200, height=250, showlegend=False, xaxis=dict(visible=False), yaxis=dict(visible=False), margin=dict(l=5,r=5,t=5,b=5))
                    st.plotly_chart(fig_cs)
