import streamlit as st
import pandas as pd
import numpy as np
from solver import BeamSolver
import design_view

st.set_page_config(page_title="Professional Beam Structural System", layout="wide")

# Session State Initialization
if 'spans' not in st.session_state: st.session_state.spans = [5.0]
if 'supports' not in st.session_state: st.session_state.supports = [{'id': 0, 'type': 'Pin'}, {'id': 1, 'type': 'Roller'}]
if 'loads' not in st.session_state: st.session_state.loads = []

st.title("🏗️ Beam Analysis & Verification Master")

# --- 1. Materials & Section (fc', fy, I กลับมาแล้ว) ---
st.sidebar.header("Section & Materials")
fc = st.sidebar.number_input("Concrete Strength (fc') [MPa]", 20.0, 50.0, 25.0)
fy = st.sidebar.number_input("Steel Strength (fy) [MPa]", 240.0, 500.0, 400.0)
b = st.sidebar.number_input("Width (b) [m]", 0.1, 1.0, 0.3)
h = st.sidebar.number_input("Height (h) [m]", 0.1, 2.0, 0.5)

use_custom_i = st.sidebar.checkbox("Custom Moment of Inertia (I)")
if use_custom_i:
    I_val = st.sidebar.number_input("Custom I (m4)", value=(b*h**3)/12, format="%.6e")
else:
    I_val = (b * h**3) / 12
    st.sidebar.info(f"Auto I: {I_val:.6e} m⁴")

# --- 2 & 6. Spans & Supports Input (Node Data) ---
st.header("1. Structure Configuration")
c_geo1, c_geo2 = st.columns([1, 2])

with c_geo1:
    n_spans = st.number_input("Number of Spans", 1, 10, len(st.session_state.spans))
    if n_spans != len(st.session_state.spans):
        st.session_state.spans = [5.0] * n_spans
        st.session_state.supports = [{'id': i, 'type': 'Pin' if i==0 else 'None'} for i in range(n_spans+1)]
        st.rerun()
    for i in range(n_spans):
        st.session_state.spans[i] = st.number_input(f"Span {i+1} (m)", 0.1, 25.0, float(st.session_state.spans[i]))

with c_geo2:
    st.write("⚓ Support Settings")
    n_nodes = n_spans + 1
    sup_init = pd.DataFrame([
        {'Node': i, 'Type': next((s['type'] for s in st.session_state.supports if s['id'] == i), 'None')}
        for i in range(n_nodes)
    ])
    edited_sup = st.data_editor(
        sup_init,
        column_config={"Type": st.column_config.SelectboxColumn("Type", options=["None", "Pin", "Roller", "Fixed"])},
        hide_index=True, use_container_width=True
    )
    st.session_state.supports = [{'id': r['Node'], 'type': r['Type']} for _, r in edited_sup.iterrows()]

# --- 3. Loads Management (Delete Row) ---
st.header("2. Loads Management")
with st.expander("Add New Load Item", expanded=True):
    cl1, cl2, cl3, cl4 = st.columns([1, 1, 1, 2])
    l_span = cl1.selectbox("Span", range(n_spans), format_func=lambda x: f"Span {x+1}")
    l_type = cl2.selectbox("Type", ["Point (P)", "Uniform (U)", "Moment (M)"])
    l_case = cl3.selectbox("Case", ["DL", "LL"])
    l_mag = cl4.number_input("Mag (kN or kNm)", value=10.0)
    
    x_pos = st.number_input("X from Left (m)", 0.0, float(st.session_state.spans[l_span]), 0.0)
    l_dist = st.number_input("Load Length (m)", 0.0, float(st.session_state.spans[l_span]), 0.0) if "U" in l_type else 0.0

    if st.button("➕ Add Load"):
        t_code = "P" if "P" in l_type else ("U" if "U" in l_type else "M")
        st.session_state.loads.append({'span_index': l_span, 'type': t_code, 'mag': l_mag*1000, 'x': x_pos, 'dist': l_dist, 'case': l_case})
        st.rerun()

if st.session_state.loads:
    for i, ld in enumerate(st.session_state.loads):
        cl1, cl2 = st.columns([6, 1])
        cl1.info(f"Load #{i+1}: {ld['type']} | Span {ld['span_index']+1} | {ld['mag']/1000} kN/kNm")
        if cl2.button("🗑️", key=f"del_{i}"):
            st.session_state.loads.pop(i)
            st.rerun()

# --- 4 & 5. Analysis, Summary, Reactions & Eq Check ---
if st.button("🚀 EXECUTE ANALYSIS", type="primary", use_container_width=True):
    solver = BeamSolver(st.session_state.spans, st.session_state.supports, st.session_state.loads, 2e11, b, h, I_val)
    df, reac_df, eq_check = solver.solve()
    
    if not df.empty:
        # Diagrams
        design_view.draw_interactive_diagrams(df, None, st.session_state.spans, st.session_state.supports, st.session_state.loads)

        # 4. Reaction Table
        st.subheader("⚓ Support Reactions")
        st.dataframe(reac_df, use_container_width=True, hide_index=True)

        # 3. Equation Check Table (Statics Verification)
        st.subheader("⚖️ Equation Check (ΣFy = 0)")
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Load (kN)", f"{eq_check['total_load']/1000:.2f}")
        c2.metric("Total Reaction (kN)", f"{eq_check['total_reac']/1000:.2f}")
        c3.metric("Static Error (N)", f"{eq_check['error']:.4f}", delta="Equilibrium" if eq_check['error'] < 1 else "Error")

        # 5. Deep Beam Check
        d_eff = h - 0.05
        is_deep = any((L / d_eff) < 2.0 for L in st.session_state.spans)
        if is_deep:
            st.warning("⚠️ **DEEP BEAM WARNING:** L/d < 2.0 detected. Simple beam theory may not apply.")
            
        else:
            st.success("✅ Standard Slender Beam (L/d ≥ 2.0)")

        st.success(f"Verified for fc'={fc} MPa, fy={fy} MPa")
