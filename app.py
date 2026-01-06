import streamlit as st
import pandas as pd
from solver import BeamSolver
import design_view

st.set_page_config(page_title="Expert Structural Analysis", layout="wide", initial_sidebar_state="expanded")

# Initialize Session States
if 'spans' not in st.session_state: st.session_state.spans = [5.0]
if 'supports' not in st.session_state: st.session_state.supports = [{'id': 0, 'type': 'Pin'}, {'id': 1, 'type': 'Roller'}]
if 'loads' not in st.session_state: st.session_state.loads = []

# --- Sidebar: Section & Materials ---
with st.sidebar:
    st.title("⚙️ Beam Settings")
    st.subheader("Materials")
    fc = st.number_input("fc' (MPa)", 20.0, 50.0, 25.0)
    fy = st.number_input("fy (MPa)", 240.0, 500.0, 400.0)
    
    st.subheader("Section Geometry")
    b = st.number_input("Width b (m)", 0.1, 1.0, 0.3)
    h = st.number_input("Height h (m)", 0.1, 2.0, 0.5)
    
    use_custom_i = st.checkbox("Define Custom I (m⁴)")
    I_val = st.number_input("Moment of Inertia", value=(b*h**3)/12, format="%.6e") if use_custom_i else (b*h**3)/12
    
    st.divider()
    if st.button("🔄 Reset All Data", type="secondary"):
        st.session_state.loads = []
        st.rerun()

# --- Main UI ---
st.title("🏗️ Professional Beam Structural System")
st.caption("Timoshenko Beam Theory | Shear Deformation Included | Serviceability Check")

# 1. Geometry Section
t1, t2 = st.tabs(["📏 Geometry & Supports", "⚖️ Applied Loads"])

with t1:
    st.header("Structure Configuration")
    col1, col2 = st.columns([1, 2])
    with col1:
        n_spans = st.number_input("Number of Spans", 1, 10, len(st.session_state.spans))
        if n_spans != len(st.session_state.spans):
            st.session_state.spans = [5.0] * n_spans
            st.session_state.supports = [{'id': i, 'type': 'Pin' if i==0 else 'None'} for i in range(n_spans+1)]
            st.rerun()
        for i in range(n_spans):
            st.session_state.spans[i] = st.number_input(f"Span {i+1} Length (m)", 0.1, 30.0, float(st.session_state.spans[i]))
    
    with col2:
        st.write("⚓ Support Conditions")
        sup_df_init = pd.DataFrame([{'Node': i, 'Type': next((s['type'] for s in st.session_state.supports if s['id'] == i), 'None')} for i in range(n_spans+1)])
        edited_sup = st.data_editor(sup_df_init, column_config={"Type": st.column_config.SelectboxColumn("Type", options=["None", "Pin", "Roller", "Fixed"])}, hide_index=True, use_container_width=True)
        st.session_state.supports = [{'id': r['Node'], 'type': r['Type']} for _, r in edited_sup.iterrows()]

with t2:
    st.header("Load Management")
    with st.expander("➕ Add New Load Item", expanded=True):
        la1, la2, la3 = st.columns(3)
        l_span = la1.selectbox("Target Span", range(n_spans), format_func=lambda x: f"Span {x+1}")
        l_type = la2.selectbox("Load Type", ["Point (P)", "Uniform (U)", "Moment (M)"])
        l_mag = la3.number_input("Magnitude (kN or kNm)", value=10.0)
        
        lb1, lb2 = st.columns(2)
        x_pos = lb1.number_input("Position x (m from left)", 0.0, float(st.session_state.spans[l_span]), 0.0)
        l_dist = lb2.number_input("Load Length (m)", 0.0, float(st.session_state.spans[l_span]), 0.0) if "U" in l_type else 0.0
        
        if st.button("Add Load to System", use_container_width=True):
            st.session_state.loads.append({'span_index': l_span, 'type': l_type[0], 'mag': l_mag*1000, 'x': x_pos, 'dist': l_dist})
            st.rerun()

    if st.session_state.loads:
        st.subheader("Current Load List")
        for i, ld in enumerate(st.session_state.loads):
            lc1, lc2 = st.columns([6, 1])
            lc1.info(f"Load #{i+1}: {ld['type']} | {ld['mag']/1000} kN/kNm | Span {ld['span_index']+1} @ {ld['x']}m")
            if lc2.button("🗑️", key=f"del_{i}"):
                st.session_state.loads.pop(i)
                st.rerun()

# --- Analysis Execution ---
st.divider()
if st.button("🚀 EXECUTE FULL ANALYSIS", type="primary", use_container_width=True):
    solver = BeamSolver(st.session_state.spans, st.session_state.supports, st.session_state.loads, 2e11, b, h, I_val)
    df, reac_df, eq = solver.solve()
    
    if not df.empty:
        # Diagrams
        design_view.draw_interactive_diagrams(df, None, st.session_state.spans, st.session_state.supports, st.session_state.loads)
        
        # 1. Summary Metrics
        st.subheader("📊 Analysis Summary")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Max Shear (Vu)", f"{df['shear'].abs().max()/1000:.2f} kN")
        m2.metric("Max Moment (Mu)", f"{df['moment'].abs().max()/1000:.2f} kNm")
        m3.metric("Max Deflection", f"{df['deflection'].abs().max()*1000:.2f} mm")
        
        d_eff = h - 0.05
        is_deep = any(L/d_eff < 2.0 for L in st.session_state.spans)
        m4.metric("Deep Beam Status", "⚠️ YES" if is_deep else "✅ NO")

        # 2. Equation Check (The Master Key)
        st.subheader("⚖️ Equation Check (Statics Verification)")
        with st.expander("View Mathematical Balance Details", expanded=True):
            ec1, ec2 = st.columns(2)
            with ec1:
                st.write("**Vertical Equilibrium (ΣFy)**")
                diff_fy = abs(eq['load_fy'] - eq['reac_fy'])
                st.write(f"Load: {eq['load_fy']/1000:.4f} kN")
                st.write(f"Reaction: {eq['reac_fy']/1000:.4f} kN")
                st.success(f"Error: {diff_fy:.6f} N") if diff_fy < 0.1 else st.error(f"Error: {diff_fy:.6f} N")
            with ec2:
                st.write("**Moment Equilibrium (ΣM @ Node 0)**")
                diff_m = abs(eq['load_m0'] - eq['reac_m0'])
                st.write(f"Load Moment: {eq['load_m0']/1000:.4f} kNm")
                st.write(f"Reaction Moment: {eq['reac_m0']/1000:.4f} kNm")
                st.success(f"Error: {diff_m:.6f} Nm") if diff_m < 0.1 else st.error(f"Error: {diff_m:.6f} Nm")

        # 3. Detailed Results
        col_res1, col_res2 = st.columns([1, 1])
        with col_res1:
            st.subheader("⚓ Reactions")
            st.dataframe(reac_df, use_container_width=True, hide_index=True)
        with col_res2:
            st.subheader("📉 Deflection Check")
            total_L = sum(st.session_state.spans)
            st.write(f"Span Limit L/240: **{total_L*1000/240:.2f} mm**")
            st.write(f"Span Limit L/360: **{total_L*1000/360:.2f} mm**")
            actual_d = df['deflection'].abs().max()*1000
            if actual_d > total_L*1000/240:
                st.error(f"Critical: Deflection ({actual_d:.2f} mm) exceeds L/240!")
            else:
                st.success(f"Serviceability: Deflection ({actual_d:.2f} mm) is within limits.")

    else:
        st.error("❌ Analysis Failed: Please check your supports and loads.")
