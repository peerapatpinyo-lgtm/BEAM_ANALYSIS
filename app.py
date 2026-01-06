import streamlit as st
import pandas as pd
import numpy as np
from solver import BeamSolver
import design_view

st.set_page_config(page_title="Beam Analysis Expert", layout="wide")

# Initialize Session State
if 'spans' not in st.session_state: st.session_state.spans = [5.0]
if 'supports' not in st.session_state: st.session_state.supports = [{'id': 0, 'type': 'Pin'}, {'id': 1, 'type': 'Roller'}]
if 'loads' not in st.session_state: st.session_state.loads = []

st.title("⚖️ Beam Structural Analysis (Standard & Deep Beam)")

# --- 1. Section & Materials (การกรอกค่า I กลับมาแล้ว) ---
st.sidebar.header("1. Section Properties")
b = st.sidebar.number_input("Width (b) [m]", 0.1, 1.0, 0.3)
h = st.sidebar.number_input("Height (h) [m]", 0.1, 2.0, 0.5)

use_custom_i = st.sidebar.checkbox("Input Custom Moment of Inertia (I)")
if use_custom_i:
    I_val = st.sidebar.number_input("Custom I (m^4)", value=(b*h**3)/12, format="%.6e")
else:
    I_val = (b * h**3) / 12
    st.sidebar.info(f"Calculated I: {I_val:.6e} m⁴")

fc = st.sidebar.number_input("Concrete Strength (f'c) [MPa]", 20.0, 50.0, 25.0)
fy = st.sidebar.number_input("Steel Strength (fy) [MPa]", 240.0, 500.0, 400.0)

# --- 2 & 3. Loads (Moment & Better Clear) ---
st.header("2. Input Spans, Supports & Loads")
t1, t2, t3 = st.tabs(["📏 Spans", "⚓ Supports", "⚖️ Loads"])

with t1:
    n_spans = st.number_input("Number of Spans", 1, 10, len(st.session_state.spans))
    if n_spans != len(st.session_state.spans):
        st.session_state.spans = [5.0] * n_spans
        st.rerun()
    for i in range(n_spans):
        st.session_state.spans[i] = st.number_input(f"Span {i+1} Length (m)", 0.1, 20.0, float(st.session_state.spans[i]))

with t2:
    n_nodes = len(st.session_state.spans) + 1
    st.write(f"Nodes: {n_nodes}")
    # Support Selection Table
    sup_df = pd.DataFrame([{"Node": i, "Type": "None"} for i in range(n_nodes)])
    # (Simplified support logic for brevity in this snippet - same as your working version)

with t3:
    # 2. Moment Load Type Included
    col1, col2, col3, col4 = st.columns([2, 1, 1, 2])
    s_idx = col1.selectbox("Span", range(len(st.session_state.spans)))
    l_type = col2.selectbox("Type", ["P", "U", "M"]) # P=Point, U=Uniform, M=Moment
    l_case = col3.selectbox("Case", ["DL", "LL"])
    l_mag = col4.number_input("Mag (kN or kNm)", value=10.0)
    
    x_pos = st.number_input("Pos x (from left of span)", 0.0, float(st.session_state.spans[s_idx]), 0.0)
    u_dist = st.number_input("Dist (for U-Load only)", 0.0, float(st.session_state.spans[s_idx]), 0.0) if l_type == "U" else 0.0

    if st.button("➕ Add Load"):
        st.session_state.loads.append({
            'span_index': s_idx, 'type': l_type, 'mag': l_mag * 1000, 
            'x': x_pos, 'dist': u_dist, 'case': l_case
        })

    # 3. BETTER CLEAR LOAD (ลบรายชิ้น)
    if st.session_state.loads:
        st.markdown("---")
        for i, ld in enumerate(st.session_state.loads):
            c1, c2 = st.columns([5, 1])
            c1.write(f"#{i+1}: {ld['type']} | {ld['mag']/1000} kN/kNm | Span {ld['span_index']+1} @ {ld['x']}m")
            if c2.button("🗑️", key=f"del_{i}"):
                st.session_state.loads.pop(i)
                st.rerun()

# --- 4 & 5. Analysis & Summary (Deep Beam Check) ---
if st.button("🚀 RUN ANALYSIS", type="primary", use_container_width=True):
    solver = BeamSolver(st.session_state.spans, st.session_state.supports, st.session_state.loads, 2e11, b, h, I_val)
    df, reactions, summ = solver.solve()
    
    if not df.empty:
        # Visualization
        design_view.draw_interactive_diagrams(df, reactions, st.session_state.spans, st.session_state.supports, st.session_state.loads)

        # 4. Analysis Summary (ครบถ้วน)
        st.subheader("📊 Analysis Summary")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Max Shear (Vu)", f"{summ['V_max']/1000:.2f} kN")
        m2.metric("Max Moment (+)", f"{summ['M_pos']/1000:.2f} kNm")
        m3.metric("Max Moment (-)", f"{summ['M_neg']/1000:.2f} kNm")
        m4.metric("Max Deflection", f"{summ['D_max']*1000:.2f} mm")

        # 5. DEEP BEAM CHECK
        if summ['is_deep']:
            st.warning("⚠️ **Deep Beam Condition!** (One or more spans have L/d < 2.0)")
            st.info("Additional shear reinforcement or Strut-and-Tie model should be verified.")
            
        else:
            st.success("✅ Standard Beam Condition (L/d ≥ 2.0)")

        # 6. RE-CHECK (Deflection Comparison)
        l_total = sum(st.session_state.spans)
        d_allow = (l_total * 1000) / 240
        st.write(f"**Deflection Re-check:** Actual ({summ['D_max']*1000:.2f} mm) vs Allowable L/240 ({d_allow:.2f} mm)")

    else:
        st.error("Analysis Error. Check loads and supports.")
