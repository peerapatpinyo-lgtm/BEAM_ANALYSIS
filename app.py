import streamlit as st
import pandas as pd
from solver import BeamSolver
import design_view

st.set_page_config(page_title="Expert Beam System", layout="wide")

if 'spans' not in st.session_state: st.session_state.spans = [5.0]
if 'supports' not in st.session_state: st.session_state.supports = [{'id': 0, 'type': 'Pin'}, {'id': 1, 'type': 'Roller'}]
if 'loads' not in st.session_state: st.session_state.loads = []

st.sidebar.header("Section & Materials")
fc = st.sidebar.number_input("fc' (MPa)", 25.0)
fy = st.sidebar.number_input("fy (MPa)", 400.0)
b = st.sidebar.number_input("b (m)", 0.3)
h = st.sidebar.number_input("h (m)", 0.5)
use_custom_i = st.sidebar.checkbox("Custom I")
I_val = st.sidebar.number_input("I (m4)", value=(b*h**3)/12, format="%.6e") if use_custom_i else (b*h**3)/12

st.header("1. Structure Configuration")
c1, c2 = st.columns([1, 2])
with c1:
    n_spans = st.number_input("Spans Count", 1, 10, len(st.session_state.spans))
    if n_spans != len(st.session_state.spans):
        st.session_state.spans = [5.0] * n_spans
        st.session_state.supports = [{'id': i, 'type': 'Pin' if i==0 else 'None'} for i in range(n_spans+1)]
        st.rerun()
    for i in range(n_spans):
        st.session_state.spans[i] = st.number_input(f"L{i+1} (m)", 0.1, 25.0, float(st.session_state.spans[i]))
with c2:
    sup_df = pd.DataFrame([{'Node': i, 'Type': next((s['type'] for s in st.session_state.supports if s['id'] == i), 'None')} for i in range(n_spans+1)])
    ed_sup = st.data_editor(sup_df, column_config={"Type": st.column_config.SelectboxColumn("Type", options=["None", "Pin", "Roller", "Fixed"])}, hide_index=True)
    st.session_state.supports = [{'id': r['Node'], 'type': r['Type']} for _, r in ed_sup.iterrows()]

st.header("2. Loads Management")
with st.expander("Add Load", expanded=True):
    cl1, cl2, cl3 = st.columns(3)
    l_span = cl1.selectbox("Span Index", range(n_spans))
    l_type = cl2.selectbox("Type", ["P", "U", "M"])
    l_mag = cl3.number_input("Mag", value=10.0)
    x_pos = st.number_input("Pos x (m)", 0.0, float(st.session_state.spans[l_span]), 0.0)
    l_dst = st.number_input("Dist (m)", 0.0, float(st.session_state.spans[l_span]), 0.0) if l_type == "U" else 0.0
    if st.button("Add"):
        st.session_state.loads.append({'span_index': l_span, 'type': l_type, 'mag': l_mag*1000, 'x': x_pos, 'dist': l_dst})
        st.rerun()

for i, ld in enumerate(st.session_state.loads):
    cc1, cc2 = st.columns([5, 1])
    cc1.info(f"#{i+1}: Span {ld['span_index']+1} | {ld['type']} | {ld['mag']/1000} kN")
    if cc2.button("🗑️", key=f"del_{i}"):
        st.session_state.loads.pop(i); st.rerun()

if st.button("RUN ANALYSIS", type="primary", use_container_width=True):
    solver = BeamSolver(st.session_state.spans, st.session_state.supports, st.session_state.loads, 2e11, b, h, I_val)
    df, reac, eq = solver.solve()
    if not df.empty:
        design_view.draw_interactive_diagrams(df, None, st.session_state.spans, st.session_state.supports, st.session_state.loads)
        st.subheader("Statics Check (Equation Check)")
        k1, k2 = st.columns(2)
        k1.metric("Fy Error (N)", f"{abs(eq['load_fy']-eq['reac_fy']):.4f}")
        k2.metric("M0 Error (Nm)", f"{abs(eq['load_m0']-eq['reac_m0']):.4f}")
        st.subheader("Reactions")
        st.table(reac)
        
        max_d = df['deflection'].abs().max() * 1000
        st.subheader("Deflection (Timoshenko)")
        st.write(f"Max: {max_d:.2f} mm | Limit L/240: {(sum(st.session_state.spans)*1000/240):.2f} mm")
