import streamlit as st
import pandas as pd
import numpy as np
from solver import BeamSolver
import rc_design
import design_view
import plotly.graph_objects as go

st.set_page_config(page_title="World-Class Beam Studio", layout="wide")

if 'loads' not in st.session_state: st.session_state.loads = []
if 'spans' not in st.session_state: st.session_state.spans = [5.0]
if 'supports' not in st.session_state: st.session_state.supports = [{'id': 0, 'type': 'Pin'}, {'id': 1, 'type': 'Roller'}]

# --- Sidebar ---
with st.sidebar:
    st.header("🛠️ Parameters")
    fc = st.number_input("fc' (MPa)", 28.0)
    fy = st.number_input("fy (MPa)", 400.0)
    st.divider()
    b = st.number_input("Width b (m)", 0.3)
    h = st.number_input("Height h (m)", 0.5)
    I_val = st.number_input("I (m⁴)", value=(b*h**3)/12, format="%.6e")
    if st.button("🗑️ Clear Loads"): st.session_state.loads = []; st.rerun()

st.title("🚀 Beam Structural Master: Analysis & Design")

# --- Inputs ---
c_geo, c_load = st.columns(2)
with c_geo:
    st.subheader("📏 Geometry")
    n_spans = st.number_input("Spans", 1, 10, len(st.session_state.spans))
    if n_spans != len(st.session_state.spans):
        st.session_state.spans = [5.0] * n_spans
        st.session_state.supports = [{'id': i, 'type': 'Pin' if i==0 else 'None'} for i in range(n_spans+1)]
        st.rerun()
    for i in range(n_spans): st.session_state.spans[i] = st.number_input(f"L{i+1}", 0.1, 20.0, float(st.session_state.spans[i]))
    df_s = pd.DataFrame([{'id': i, 'type': next((s['type'] for s in st.session_state.supports if s['id'] == i), 'None')} for i in range(n_spans+1)])
    ed_s = st.data_editor(df_s, hide_index=True)
    st.session_state.supports = ed_s.to_dict('records')

with c_load:
    st.subheader("📥 Load List")
    with st.expander("➕ Add New Load"):
        l_idx = st.selectbox("Span", range(n_spans)); l_t = st.selectbox("Type", ["P","U","M"]); l_m = st.number_input("Mag (kN)", 10.0)
        lx = st.number_input("Pos x (m)", 0.0); ld = st.number_input("Dist (m)", 0.0)
        if st.button("Add"):
            st.session_state.loads.append({'span_index': l_idx, 'type': l_t[0], 'mag': l_m*1000, 'x': lx, 'dist': ld})
            st.rerun()
    for i, ld in enumerate(st.session_state.loads):
        cx, cd = st.columns([4, 1])
        cx.write(f"{i+1}. {ld['type']} | {ld['mag']/1000}kN | Span {ld['span_index']+1}")
        if cd.button("🗑️", key=f"d_{i}"): st.session_state.loads.pop(i); st.rerun()

# --- Execution ---
st.divider()
if st.button("🚀 EXECUTE ANALYSIS", type="primary", use_container_width=True):
    sol = BeamSolver(st.session_state.spans, st.session_state.supports, st.session_state.loads, 2e11, b, h, I_val)
    df, reac, eq = sol.solve()
    
    if not df.empty:
        # 1. ANALYSIS SECTION (ห้ามหาย)
        st.header("📊 Structural Analysis")
        design_view.draw_interactive_diagrams(df, reac, st.session_state.spans, pd.DataFrame(st.session_state.supports), st.session_state.loads)
        st.subheader("⚓ Reactions")
        st.dataframe(reac, use_container_width=True, hide_index=True)

        # 2. DESIGN SECTION
        st.divider()
        st.header("🧱 Professional RC Design")
        mu_pos, mu_neg = df['moment'].max()/1000, df['moment'].min()/1000
        rc = rc_design.calculate_advanced_rc(mu_pos, mu_neg, df['shear'].abs().max()/1000, b, h, fc, fy)

        c_opt, c_drawing = st.columns([1, 1.2])
        
        with c_opt:
            st.subheader("🎯 Optimization Report")
            st.title(rc['opt_status'])
            st.write(f"**เหตุผล:** {rc['opt_desc']}")
            st.info(f"**คำแนะนำ:** {rc['suggestion']}")
            
            st.subheader("📝 Details")
            st.write(f"- Required As (Top): {rc['as_top']:.0f} mm²")
            st.write(f"- Required As (Bot): {rc['as_bot']:.0f} mm²")
            st.write(f"- Stirrups: RB9 @ {int(rc['spacing'])} mm")

        with c_drawing:
            st.subheader("🎨 Section Detail (Typical)")
            # วาดรูป Section ด้วย Plotly
            fig = go.Figure()
            # คาน
            fig.add_shape(type="rect", x0=0, y0=0, x1=rc['b'], y1=rc['h'], line=dict(color="Black", width=3))
            # เหล็กบน
            for i in range(rc['n_top']):
                fig.add_trace(go.Scatter(x=[(rc['b']/(rc['n_top']+1))*(i+1)], y=[rc['h']-40], mode='markers', marker=dict(size=12, color='Red'), name='Top Bar'))
            # เหล็กล่าง
            for i in range(rc['n_bot']):
                fig.add_trace(go.Scatter(x=[(rc['b']/(rc['n_bot']+1))*(i+1)], y=[40], mode='markers', marker=dict(size=12, color='Blue'), name='Bottom Bar'))
            
            fig.update_layout(width=400, height=400, xaxis=dict(range=[-0.1, rc['b']+0.1], visible=False), yaxis=dict(range=[-0.1, rc['h']+0.1], visible=False), showlegend=False)
            st.plotly_chart(fig)
            st.caption(f"Drawing: {rc['n_top']}xDB20 (Top) & {rc['n_bot']}xDB20 (Bottom)")
