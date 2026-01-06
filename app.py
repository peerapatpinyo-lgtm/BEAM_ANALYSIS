import streamlit as st
import pandas as pd
from solver import BeamSolver
import design_view

st.set_page_config(page_title="Pro Structural Analysis", layout="wide")

if 'spans' not in st.session_state: st.session_state.spans = [5.0]
if 'supports' not in st.session_state: st.session_state.supports = [{'id': 0, 'type': 'Pin'}, {'id': 1, 'type': 'Roller'}]
if 'loads' not in st.session_state: st.session_state.loads = []

with st.sidebar:
    st.header("Section & Materials")
    fc = st.number_input("fc' (MPa)", 25.0)
    b = st.number_input("Width (m)", 0.3)
    h = st.number_input("Height (m)", 0.5)
    use_i = st.checkbox("Custom I")
    I_val = st.number_input("I (m4)", value=(b*h**3)/12, format="%.6e") if use_i else (b*h**3)/12

st.title("🏗️ Beam Analysis Master (Timoshenko)")

# Configuration
st.header("1. Structure & Supports")
c1, c2 = st.columns([1, 2])
with c1:
    ns = st.number_input("Number of Spans", 1, 10, len(st.session_state.spans))
    if ns != len(st.session_state.spans):
        st.session_state.spans = [5.0] * ns
        st.session_state.supports = [{'id': i, 'type': 'Pin' if i==0 else 'None'} for i in range(ns+1)]
        st.rerun()
    for i in range(ns):
        st.session_state.spans[i] = st.number_input(f"L{i+1} (m)", 0.1, 30.0, float(st.session_state.spans[i]))
with c2:
    st.write("⚓ Supports")
    sdf = pd.DataFrame([{'Node': i, 'Type': next((s['type'] for s in st.session_state.supports if s['id'] == i), 'None')} for i in range(ns+1)])
    eds = st.data_editor(sdf, column_config={"Type": st.column_config.SelectboxColumn("Type", options=["None", "Pin", "Roller", "Fixed"])}, hide_index=True)
    st.session_state.supports = [{'id': r['Node'], 'type': r['Type']} for _, r in eds.iterrows()]

# Loading
st.header("2. Applied Loads")
with st.expander("➕ Add New Load", expanded=True):
    ca, cb, cc = st.columns(3)
    l_idx = ca.selectbox("Span", range(ns))
    l_t = cb.selectbox("Type", ["P (Point)", "U (Uniform)", "M (Moment)"])
    l_m = cc.number_input("Mag (kN/kNm)", 10.0)
    pos_x = st.number_input("x (m from left)", 0.0, float(st.session_state.spans[l_idx]), 0.0)
    dist_l = st.number_input("Length (m)", 0.0, float(st.session_state.spans[l_idx])) if "U" in l_t else 0.0
    if st.button("Add Load"):
        st.session_state.loads.append({'span_index': l_idx, 'type': l_t[0], 'mag': l_m*1000, 'x': pos_x, 'dist': dist_l})
        st.rerun()

for i, ld in enumerate(st.session_state.loads):
    cc1, cc2 = st.columns([5, 1])
    cc1.info(f"#{i+1}: Span {ld['span_index']+1} | {ld['type']} | {ld['mag']/1000} kN/kNm")
    if cc2.button("🗑️", key=f"d_{i}"):
        st.session_state.loads.pop(i); st.rerun()

# Run
if st.button("🚀 EXECUTE ANALYSIS", type="primary", use_container_width=True):
    sol = BeamSolver(st.session_state.spans, st.session_state.supports, st.session_state.loads, 2e11, b, h, I_val)
    df, reac, eq = sol.solve()
    if not df.empty:
        design_view.draw_interactive_diagrams(df, None, st.session_state.spans, st.session_state.supports, st.session_state.loads)
        
        # ปรับปรุงส่วน Equation Check ให้สวยงาม
        st.header("⚖️ Statics Verification")
        v1, v2 = st.columns(2)
        with v1:
            st.write("**Vertical Forces (ΣFy)**")
            st.metric("Total Load", f"{eq['l_fy']/1000:.3f} kN")
            st.metric("Total Reaction", f"{eq['r_fy']/1000:.3f} kN")
            err_f = abs(eq['l_fy'] - eq['r_fy'])
            if err_f < 1e-7:
                st.success("✅ **Balanced** (Error ≈ 0)")
            else:
                st.warning(f"⚠️ **Error:** {err_f:.4f} N")

        with v2:
            st.write("**Moments @ Node 0 (ΣM)**")
            st.metric("Total Load Moment", f"{eq['l_m0']/1000:.3f} kNm")
            st.metric("Total Reaction Moment", f"{eq['r_m0']/1000:.3f} kNm")
            err_m = abs(eq['l_m0'] - eq['r_m0'])
            if err_m < 1e-7:
                st.success("✅ **Balanced** (Error ≈ 0)")
            else:
                st.warning(f"⚠️ **Error:** {err_m:.4f} Nm")

        st.header("⚓ Reaction Table")
        st.dataframe(reac, use_container_width=True, hide_index=True)

        st.header("📉 Serviceability (Deflection)")
        max_d = df['deflection'].abs().max() * 1000
        total_L = sum(st.session_state.spans)
        st.write(f"Max Displacement: **{max_d:.3f} mm**")
        st.write(f"Limit L/240: **{total_L*1000/240:.2f} mm**")
        if max_d > total_L*1000/240:
            st.error("⚠️ Deflection exceeds limit!")
        else:
            st.success("✅ Deflection is within limit.")
