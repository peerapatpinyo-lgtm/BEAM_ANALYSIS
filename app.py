import streamlit as st
import pandas as pd
from solver import BeamSolver
import rc_design
import design_view 

st.set_page_config(page_title="Professional Beam Structural Analyzer", layout="wide")

# --- Initialize Session State ---
if 'spans' not in st.session_state: st.session_state.spans = [5.0]
if 'supports' not in st.session_state: st.session_state.supports = [{'id': 0, 'type': 'Pin'}, {'id': 1, 'type': 'Roller'}]
if 'loads' not in st.session_state: st.session_state.loads = []

# --- Sidebar: Section & Material ---
with st.sidebar:
    st.header("⚙️ Section & Material")
    fc = st.number_input("fc' (Concrete Strength, MPa)", 25.0)
    b = st.number_input("Width b (m)", 0.3)
    h = st.number_input("Height h (m)", 0.5)
    
    st.divider()
    st.write("**Moment of Inertia (I)**")
    i_mode = st.radio("I Calculation", ["Auto (bh³/12)", "Manual Input"], horizontal=True)
    
    if i_mode == "Manual Input":
        # ให้กรอกหน่วย m4 แบบวิทยาศาสตร์เพื่อให้วิศวกรทำงานง่ายขึ้น
        I_val = st.number_input("I value (m⁴)", value=(b*h**3)/12, format="%.6e")
    else:
        I_val = (b*h**3)/12
        st.info(f"Calculated I: {I_val:.6e} m⁴")
    
    st.divider()
    if st.button("🗑️ Reset All Loads", use_container_width=True):
        st.session_state.loads = []
        st.rerun()

st.title("🏗️ Beam Structural Master")

# --- Step 1: Geometry ---
st.header("1. Structure & Supports")
tab_geo, tab_load = st.tabs(["📏 Geometry & Nodes", "📥 Applied Loads"])

with tab_geo:
    c1, c2 = st.columns([1, 2])
    with c1:
        n_spans = st.number_input("Number of Spans", 1, 10, len(st.session_state.spans))
        if n_spans != len(st.session_state.spans):
            st.session_state.spans = [5.0] * n_spans
            st.session_state.supports = [{'id': i, 'type': 'Pin' if i==0 else 'None'} for i in range(n_spans+1)]
            st.rerun()
        for i in range(n_spans):
            st.session_state.spans[i] = st.number_input(f"L{i+1} (m)", 0.1, 30.0, float(st.session_state.spans[i]))
    with c2:
        st.write("⚓ Support Conditions")
        df_sup = pd.DataFrame([{'id': i, 'type': next((s['type'] for s in st.session_state.supports if s['id'] == i), 'None')} for i in range(n_spans+1)])
        ed_sup = st.data_editor(df_sup, column_config={"type": st.column_config.SelectboxColumn("Type", options=["None", "Pin", "Roller", "Fixed"])}, hide_index=True, use_container_width=True)
        st.session_state.supports = ed_sup.to_dict('records')

with tab_load:
    with st.form("load_form", clear_on_submit=True):
        la, lb, lc = st.columns(3)
        l_idx = la.selectbox("Select Span", range(n_spans), format_func=lambda x: f"Span {x+1}")
        l_type = lb.selectbox("Load Type", ["P (Point)", "U (Uniform)", "M (Moment)"])
        l_mag = lc.number_input("Magnitude (kN or kNm)", 10.0)
        
        pos_col, dist_col = st.columns(2)
        x_pos = pos_col.number_input("Position x (m from left)", 0.0, float(st.session_state.spans[l_idx]), 0.0)
        dist = dist_col.number_input("Distance (m)", 0.0, float(st.session_state.spans[l_idx])) if "U" in l_type else 0.0
        
        if st.form_submit_button("➕ Add Load"):
            st.session_state.loads.append({
                'span_index': l_idx, 
                'type': l_type[0], 
                'mag': l_mag * 1000, # Convert to N
                'x': x_pos, 
                'dist': dist,
                'case': 'DL' # Default to Dead Load
            })
            st.rerun()

    if st.session_state.loads:
        st.subheader("Current Load Inventory")
        for i, ld in enumerate(st.session_state.loads):
            col_info, col_del = st.columns([6, 1])
            col_info.info(f"Load #{i+1}: {ld['type']} | {ld['mag']/1000} kN | Span {ld['span_index']+1} @ {ld['x']}m")
            if col_del.button("🗑️", key=f"del_{i}"):
                st.session_state.loads.pop(i)
                st.rerun()

# --- Step 2: Analysis & Display ---
st.divider()
if st.button("🚀 EXECUTE ANALYSIS", type="primary", use_container_width=True):
    sol = BeamSolver(st.session_state.spans, st.session_state.supports, st.session_state.loads, 2e11, b, h, I_val)
    df, reac, eq = sol.solve()
    
    if not df.empty:
        # 1. Visualization (จากไฟล์ design_view.py เดิม)
        design_view.draw_interactive_diagrams(df, reac, st.session_state.spans, pd.DataFrame(st.session_state.supports), st.session_state.loads)

        # 2. Statics Balance Check
        st.header("⚖️ Statics Verification (Equilibrium)")
        v1, v2 = st.columns(2)
        with v1:
            st.metric("Total Vertical Load", f"{eq['l_fy']/1000:.3f} kN")
            st.metric("Total Reaction Fy", f"{eq['r_fy']/1000:.3f} kN")
            err_f = abs(eq['l_fy'] - eq['r_fy'])
            if err_f < 1e-7: st.success("✅ **Balanced** (Error ≈ 0.000 N)")
            else: st.warning(f"⚠️ Error: {err_f:.4f} N")
        
        with v2:
            st.metric("Total Load Moment @0", f"{eq['l_m0']/1000:.3f} kNm")
            st.metric("Total Reaction Moment @0", f"{eq['r_m0']/1000:.3f} kNm")
            err_m = abs(eq['l_m0'] - eq['r_m0'])
            if err_m < 1e-7: st.success("✅ **Balanced** (Error ≈ 0.000 Nm)")
            else: st.warning(f"⚠️ Error: {err_m:.4f} Nm")

        # 3. Reactions Table
        st.header("⚓ Support Reactions")
        st.dataframe(reac, use_container_width=True, hide_index=True)

        # 4. RC Design Summary (เชื่อมโยง rc_design.py)
        st.header("🧱 Preliminary Reinforcement Design")
        mu_max = df['moment'].abs().max() / 1000
        vu_max = df['shear'].abs().max() / 1000
        rc_res = rc_design.calculate_rc_details(mu_max, vu_max, b, h, fc)
        
        rc1, rc2, rc3 = st.columns(3)
        rc1.metric("Required As", f"{rc_res['as_mm2']:.0f} mm²")
        rc2.metric("Steel Ratio (ρ)", f"{rc_res['rho']:.5f}")
        rc3.info(f"Suggested Steel:\n\n**{max(2, int(rc_res['as_mm2']/314.16+1))}xDB20**")
    else:
        st.error("Check your supports and loads. The beam might be unstable.")
