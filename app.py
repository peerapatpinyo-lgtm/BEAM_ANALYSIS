import streamlit as st
import pandas as pd
from solver import BeamSolver
import design_view

st.set_page_config(page_title="Beam Analysis Master", layout="wide")

# Session State Persistence
if 'spans' not in st.session_state: st.session_state.spans = [5.0]
if 'supports' not in st.session_state: st.session_state.supports = [{'id': 0, 'type': 'Pin'}, {'id': 1, 'type': 'Roller'}]
if 'loads' not in st.session_state: st.session_state.loads = []

st.title("🏗️ Beam Structural Analysis System")

# --- 1. Section & Materials (I-Value Handling) ---
st.sidebar.header("Section Properties")
b = st.sidebar.number_input("Width (b) [m]", 0.1, 1.0, 0.3)
h = st.sidebar.number_input("Height (h) [m]", 0.1, 2.0, 0.5)

use_custom_i = st.sidebar.checkbox("Custom Moment of Inertia (I)")
if use_custom_i:
    I_val = st.sidebar.number_input("I (m^4)", value=(b*h**3)/12, format="%.6e")
else:
    I_val = (b * h**3) / 12
    st.sidebar.info(f"Calculated I: {I_val:.6e} m⁴")

# --- 2 & 6. Spans & Supports Input (Node Based) ---
st.header("1. Geometry & Supports")
col_s1, col_s2 = st.columns([1, 2])

with col_s1:
    n_spans = st.number_input("Number of Spans", 1, 10, len(st.session_state.spans))
    if n_spans != len(st.session_state.spans):
        st.session_state.spans = [5.0] * n_spans
        # Reset supports to match new nodes
        st.session_state.supports = [{'id': i, 'type': 'Pin' if i==0 else 'None'} for i in range(n_spans+1)]
        st.rerun()

    for i in range(n_spans):
        st.session_state.spans[i] = st.number_input(f"Span {i+1} (m)", 0.1, 20.0, float(st.session_state.spans[i]))

with col_s2:
    st.write("⚓ Support Configuration")
    n_nodes = len(st.session_state.spans) + 1
    # Create editable table for Supports
    sup_init_df = pd.DataFrame([
        {'Node': i, 'Type': next((s['type'] for s in st.session_state.supports if s['id'] == i), 'None')}
        for i in range(n_nodes)
    ])
    edited_sup = st.data_editor(
        sup_init_df,
        column_config={"Type": st.column_config.SelectboxColumn("Type", options=["None", "Pin", "Roller", "Fixed"])},
        hide_index=True, use_container_width=True
    )
    st.session_state.supports = [{'id': r['Node'], 'type': r['Type']} for _, r in edited_sup.iterrows()]

# --- 3. Loads Management (Delete per item) ---
st.header("2. Applied Loads")
with st.expander("Add New Load (Point, Uniform, or Moment)", expanded=True):
    c1, c2, c3, c4 = st.columns([1, 1, 1, 2])
    l_span = c1.selectbox("Span", range(n_spans), format_func=lambda x: f"Span {x+1}")
    l_type = c2.selectbox("Type", ["Point (P)", "Uniform (U)", "Moment (M)"])
    l_case = c3.selectbox("Case", ["DL", "LL"])
    l_mag = c4.number_input("Magnitude (kN or kNm)", value=10.0)
    
    x_pos = st.number_input("Position x (m from span left)", 0.0, float(st.session_state.spans[l_span]), 0.0)
    l_dist = st.number_input("Load Width (m)", 0.0, float(st.session_state.spans[l_span]), 0.0) if "U" in l_type else 0.0

    if st.button("➕ Add Load"):
        t_code = "P" if "P" in l_type else ("U" if "U" in l_type else "M")
        st.session_state.loads.append({'span_index': l_span, 'type': t_code, 'mag': l_mag*1000, 'x': x_pos, 'dist': l_dist, 'case': l_case})
        st.rerun()

if st.session_state.loads:
    st.subheader("Current Loads List")
    for i, ld in enumerate(st.session_state.loads):
        cl1, cl2 = st.columns([5, 1])
        cl1.info(f"#{i+1}: Span {ld['span_index']+1} | {ld['type']} Load: {ld['mag']/1000} | @ x={ld['x']}m")
        if cl2.button("🗑️", key=f"del_{i}"):
            st.session_state.loads.pop(i)
            st.rerun()

# --- 4 & 5. Analysis & Summary (Deep Beam Check) ---
if st.button("🚀 EXECUTE ANALYSIS", type="primary", use_container_width=True):
    solver = BeamSolver(st.session_state.spans, st.session_state.supports, st.session_state.loads, 2e11, b, h, I_val)
    df, reactions, summ = solver.solve()
    
    if not df.empty:
        design_view.draw_interactive_diagrams(df, reactions, st.session_state.spans, st.session_state.supports, st.session_state.loads)

        # Summary Display
        st.subheader("📊 Analysis Summary")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Max Shear (Vu)", f"{summ['V_max']/1000:.2f} kN")
        m2.metric("Max Moment (+)", f"{summ['M_pos']/1000:.2f} kNm")
        m3.metric("Max Moment (-)", f"{summ['M_neg']/1000:.2f} kNm")
        m4.metric("Max Deflection", f"{summ['D_max']*1000:.2f} mm")

        # 5. Deep Beam Check
        if summ['is_deep']:
            st.warning(f"⚠️ **Deep Beam Condition Detected!** (L/d < 2.0)")
            st.write("The shear behavior might be dominated by arch action. Check reinforcement accordingly.")
            
        else:
            st.success("✅ Slender Beam Condition (L/d ≥ 2.0)")

        # Additional Check: Deflection L/240
        L_total = sum(st.session_state.spans)
        st.info(f"Allowable Deflection (L/240): {(L_total*1000/240):.2f} mm")
    else:
        st.error("Analysis failed. Please verify support stability (at least 2 supports).")
