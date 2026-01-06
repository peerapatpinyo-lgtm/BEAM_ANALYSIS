import streamlit as st
import pandas as pd
from solver import BeamSolver
import design_view

st.set_page_config(page_title="Professional Beam Structural Analyzer", layout="wide")

# --- Persistent Data ---
if 'spans' not in st.session_state: st.session_state.spans = [5.0]
if 'supports' not in st.session_state: st.session_state.supports = [{'id': 0, 'type': 'Pin'}, {'id': 1, 'type': 'Roller'}]
if 'loads' not in st.session_state: st.session_state.loads = []

# --- Sidebar: Section & Material ---
with st.sidebar:
    st.title("🛡️ Engineering Settings")
    st.header("Section Properties")
    fc = st.number_input("Concrete fc' (MPa)", 25.0)
    b = st.number_input("Width b (m)", 0.3)
    h = st.number_input("Height h (m)", 0.5)
    st.divider()
    use_custom_i = st.checkbox("Specify Custom I")
    I_val = st.number_input("I (m4)", value=(b*h**3)/12, format="%.6e") if use_custom_i else (b*h**3)/12
    st.divider()
    if st.button("🗑️ Clear All Loads", use_container_width=True):
        st.session_state.loads = []; st.rerun()

# --- Header ---
st.title("🏗️ Beam Master Analysis")
st.caption("Advanced Finite Element Analysis using Timoshenko-Modified Beam Theory")

# --- Step 1 & 2: Structure and Loads ---
tab1, tab2 = st.tabs(["📏 Geometry & Supports", "⚖️ Loads & Forces"])

with tab1:
    c1, c2 = st.columns([1, 2])
    with c1:
        n_spans = st.number_input("Total Spans", 1, 10, len(st.session_state.spans))
        if n_spans != len(st.session_state.spans):
            st.session_state.spans = [5.0] * n_spans
            st.session_state.supports = [{'id': i, 'type': 'Pin' if i==0 else 'None'} for i in range(n_spans+1)]
            st.rerun()
        for i in range(n_spans):
            st.session_state.spans[i] = st.number_input(f"L{i+1} (m)", 0.1, 30.0, float(st.session_state.spans[i]))
    with c2:
        st.write("⚓ Support Conditions")
        df_sup = pd.DataFrame([{'Node': i, 'Type': next((s['type'] for s in st.session_state.supports if s['id'] == i), 'None')} for i in range(n_spans+1)])
        ed_sup = st.data_editor(df_sup, column_config={"Type": st.column_config.SelectboxColumn("Type", options=["None", "Pin", "Roller", "Fixed"])}, hide_index=True, use_container_width=True)
        st.session_state.supports = [{'id': r['Node'], 'type': r['Type']} for _, r in ed_sup.iterrows()]

with tab2:
    with st.expander("➕ Add New Load Case", expanded=True):
        la, lb, lc = st.columns(3)
        l_idx = la.selectbox("Select Span", range(n_spans), format_func=lambda x: f"Span {x+1}")
        l_type = lb.selectbox("Load Type", ["P (Point Load)", "U (Uniform Load)", "M (Moment)"])
        l_mag = lc.number_input("Magnitude (kN / kNm)", 10.0)
        
        lx, ld = st.columns(2)
        x_pos = lx.number_input("x-Position (m from left of span)", 0.0, float(st.session_state.spans[l_idx]), 0.0)
        dist = ld.number_input("Distribution Length (m)", 0.0, float(st.session_state.spans[l_idx])) if "U" in l_type else 0.0
        
        if st.button("Apply Load", use_container_width=True):
            st.session_state.loads.append({'span_index': l_idx, 'type': l_type[0], 'mag': l_mag*1000, 'x': x_pos, 'dist': dist})
            st.rerun()

    if st.session_state.loads:
        st.subheader("Current Load Inventory")
        for i, ld in enumerate(st.session_state.loads):
            col_info, col_del = st.columns([6, 1])
            col_info.info(f"Load #{i+1}: {ld['type']} | {ld['mag']/1000} kN | Span {ld['span_index']+1} @ {ld['x']}m")
            if col_del.button("🗑️", key=f"del_{i}"):
                st.session_state.loads.pop(i); st.rerun()

# --- Step 3: Analysis & Results ---
st.divider()
if st.button("🚀 EXECUTE FULL ANALYSIS", type="primary", use_container_width=True):
    solver = BeamSolver(st.session_state.spans, st.session_state.supports, st.session_state.loads, 2e11, b, h, I_val)
    df, reac, eq = solver.solve()
    
    if not df.empty:
        # Diagrams
        design_view.draw_interactive_diagrams(df, None, st.session_state.spans, st.session_state.supports, st.session_state.loads)

        # Statics Balance Report
        st.header("⚖️ Statics Verification (Equilibrium)")
        v1, v2 = st.columns(2)
        with v1:
            st.metric("Total Vertical Load", f"{eq['l_fy']/1000:.3f} kN")
            st.metric("Total Reaction Fy", f"{eq['r_fy']/1000:.3f} kN")
            err_f = abs(eq['l_fy'] - eq['r_fy'])
            if err_f < 1e-7: st.success("✅ **Balanced** (Error ≈ 0.000 N)")
            elif err_f < 0.1: st.warning(f"⚠️ **Near Balance** (Error: {err_f:.4f} N)")
            else: st.error(f"❌ **Unbalanced** (Error: {err_f:.2f} N)")
        
        with v2:
            st.metric("Total Load Moment", f"{eq['l_m0']/1000:.3f} kNm")
            st.metric("Total Reaction Moment", f"{eq['r_m0']/1000:.3f} kNm")
            err_m = abs(eq['l_m0'] - eq['r_m0'])
            if err_m < 1e-7: st.success("✅ **Balanced** (Error ≈ 0.000 Nm)")
            elif err_m < 0.1: st.warning(f"⚠️ **Near Balance** (Error: {err_m:.4f} Nm)")
            else: st.error(f"❌ **Unbalanced** (Error: {err_m:.2f} Nm)")

        # Result Tables
        r_col, d_col = st.columns([1, 1])
        with r_col:
            st.subheader("⚓ Support Reactions")
            st.dataframe(reac, use_container_width=True, hide_index=True)
        with d_col:
            st.subheader("📉 Serviceability Check")
            max_d = df['deflection'].abs().max() * 1000
            total_L = sum(st.session_state.spans)
            limit = (total_L * 1000) / 240
            st.write(f"**Max Deflection:** {max_d:.3f} mm")
            st.write(f"**Standard Limit (L/240):** {limit:.2f} mm")
            if max_d > limit: st.error("❌ Deflection limit exceeded.")
            else: st.success("✅ Deflection is within safe limits.")
    else:
        st.error("Analysis failed. Please check if the structure is stable.")
