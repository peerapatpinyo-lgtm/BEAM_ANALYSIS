import streamlit as st
import pandas as pd
import numpy as np
from solver import BeamSolver
import design_view

# --- Page Config ---
st.set_page_config(page_title="Beam Analysis Pro", layout="wide", page_icon="🏗️")

# --- Session State ---
if 'spans' not in st.session_state: st.session_state['spans'] = [5.0, 5.0]
if 'supports' not in st.session_state: 
    st.session_state['supports'] = [{'id': 0, 'type': 'Pin'}, {'id': 1, 'type': 'Roller'}, {'id': 2, 'type': 'Roller'}]
if 'loads' not in st.session_state: st.session_state['loads'] = []

# --- Sidebar ---
st.sidebar.title("🏗️ Beam Settings")
st.sidebar.markdown("---")
if st.sidebar.button("Reset Project", type="primary"):
    st.session_state['spans'] = [5.0]
    st.session_state['supports'] = [{'id': 0, 'type': 'Pin'}, {'id': 1, 'type': 'Roller'}]
    st.session_state['loads'] = []
    st.rerun()

st.sidebar.markdown("### 1. Section Properties")
E = st.sidebar.number_input("Elastic Modulus (E) [Pa]", value=2e11, format="%.2e")

input_method = st.sidebar.radio("Input Method", ["Rectangular Size (b x h)", "Custom Properties (I, A)"])
if input_method == "Rectangular Size (b x h)":
    c1, c2 = st.sidebar.columns(2)
    b = c1.number_input("Width (b) [m]", 0.30)
    h = c2.number_input("Depth (h) [m]", 0.50)
    I = (b * h**3) / 12
    A = b * h
    st.sidebar.info(f"I = {I:.2e} m⁴ | A = {A:.2f} m²")
else:
    I = st.sidebar.number_input("Inertia (I) [m^4]", 5e-5, format="%.2e")
    A = st.sidebar.number_input("Area (A) [m^2]", 0.01, format="%.4f")

use_timoshenko = st.sidebar.checkbox("Advanced: Timoshenko (Shear Deform.)", False)
G = st.sidebar.number_input("Shear Modulus (G)", 7.7e10, format="%.2e") if use_timoshenko else None

st.sidebar.markdown("---")
st.sidebar.markdown("### 2. Load Factors (ULS)")
dl_factor = st.sidebar.number_input("Dead Load Factor", 1.4)
ll_factor = st.sidebar.number_input("Live Load Factor", 1.7)

# --- Main Interface ---
st.title("🏗️ Structural Beam Analysis")
st.info("ℹ️ **Analysis Method:** Direct Stiffness Method (FEM) combined with Exact Integration for internal forces. Results shown are Factored Loads (ULS).")

tab1, tab2, tab3 = st.tabs(["1️⃣ Spans", "2️⃣ Supports", "3️⃣ Loads"])

with tab1:
    col_s1, _ = st.columns([2, 1])
    with col_s1:
        n = st.number_input("Number of Spans", 1, 10, len(st.session_state['spans']))
        current = st.session_state['spans']
        if len(current) < n: current.extend([5.0]*(n-len(current)))
        else: current = current[:n]
        
        new_spans = []
        cols = st.columns(min(n, 4))
        for i in range(n):
            new_spans.append(cols[i%4].number_input(f"Span {i+1} (m)", value=float(current[i]), min_value=0.1, key=f"s_{i}"))
        st.session_state['spans'] = new_spans
    st.caption(f"Total Length: {sum(new_spans):.2f} m")

with tab2:
    sup_data = []
    nodes_count = len(st.session_state['spans']) + 1
    current_sups = {int(s.get('id', -1)): s.get('type') for s in st.session_state['supports'] if 'id' in s}
    
    for i in range(nodes_count):
        stype = current_sups.get(i, "None")
        sup_data.append({"Node ID": i+1, "Support Type": stype}) 

    edited = st.data_editor(pd.DataFrame(sup_data), column_config={
        "Node ID": st.column_config.NumberColumn(format="%d", disabled=True), 
        "Support Type": st.column_config.SelectboxColumn(options=["None","Pin","Roller","Fixed"], required=True)
    }, hide_index=True, use_container_width=True)
    
    st.session_state['supports'] = [{'id': r['Node ID']-1, 'type': r['Support Type']} for _, r in edited.iterrows() if r['Support Type'] != "None"]

with tab3:
    c1, c2, c3 = st.columns([1,1,2])
    span_idx = c1.selectbox("Select Span", range(len(st.session_state['spans'])), format_func=lambda x: f"Span {x+1}")
    l_type = c2.selectbox("Load Type", ["Point Load (P)", "Uniform Load (U)", "Moment (M)"])
    l_case = c2.selectbox("Case", ["DL", "LL"])
    mag = c3.number_input("Magnitude (kg, kg/m, kg-m)", value=1000.0)
    
    sl = st.session_state['spans'][span_idx]
    if "Uniform" in l_type:
        cp = st.columns(2)
        x1 = cp[0].number_input("Start (m)", 0.0, float(sl), 0.0)
        x2 = cp[1].number_input("End (m)", 0.0, float(sl), float(sl))
        x_loc, dist = x1, x2-x1
    else:
        x_loc = st.number_input("Position x (m)", 0.0, float(sl), float(sl)/2)
        dist = 0
        
    if st.button("➕ Add Load", type="primary"):
        code = 'P' if 'Point' in l_type else ('U' if 'Uniform' in l_type else 'M')
        st.session_state['loads'].append({'span_index': span_idx, 'type': code, 'mag': mag, 'x': x_loc, 'dist': dist, 'case': l_case})
        st.rerun()
        
    if st.session_state['loads']:
        disp_data = []
        for l in st.session_state['loads']:
            s_idx = l.get('span_index', 0)
            disp_data.append({
                "Span": s_idx + 1, "Type": l['type'], "Mag": l['mag'],
                "Pos": f"x={l['x']}" + (f" to {l['x']+l['dist']}" if l['type']=='U' else ""), "Case": l['case']
            })
        st.dataframe(pd.DataFrame(disp_data), use_container_width=True)
        if st.button("Clear Last Load"): 
            st.session_state['loads'].pop()
            st.rerun()

# --- RUN ---
st.markdown("---")
if st.button("🚀 Run Analysis", type="primary", use_container_width=True):
    if len(st.session_state['supports']) < 2:
        st.error("Error: Unstable Structure. Please assign at least 2 supports.")
    else:
        g = 9.81
        calc_loads = []
        
        # --- 1. GEOMETRY SETUP ---
        spans = st.session_state['spans']
        cum_spans = [0.0] + list(np.cumsum(spans))
        total_beam_length = cum_spans[-1]
        
        # Variables for Checking
        check_applied_force_y_clipped = 0.0  # แรงที่ลงบนคานจริง (Physics Correct)
        phantom_force_y = 0.0                # แรงที่ยื่นเกินคาน (The Overhang)
        
        overhang_warnings = [] # เก็บข้อความแจ้งเตือน

        # --- 2. PROCESS LOADS & DETECT OVERHANG ---
        for i, l in enumerate(st.session_state['loads']):
            fac = dl_factor if l['case']=='DL' else ll_factor
            factored_mag = l['mag'] * fac * g
            
            # Send raw load to solver
            new_l = l.copy()
            new_l['mag'] = factored_mag
            calc_loads.append(new_l)
            
            # --- INTELLIGENT CHECK LOGIC ---
            span_idx = int(l.get('span_index', 0))
            if span_idx < len(spans):
                start_node_x = cum_spans[span_idx]
                local_x = float(l['x'])
                global_start = start_node_x + local_x
                
                # Point Load Logic
                if l['type'] == 'P':
                    if 0 <= global_start <= total_beam_length + 1e-4:
                        check_applied_force_y_clipped += factored_mag
                    elif global_start > total_beam_length:
                         phantom_force_y += factored_mag
                         overhang_warnings.append(f"• Point Load at x={global_start:.2f}m is outside beam (L={total_beam_length:.2f}m)")

                # Uniform Load Logic
                elif l['type'] == 'U':
                    dist = float(l['dist'])
                    global_end = global_start + dist
                    
                    # Calculate Intersection with Beam
                    overlap_start = max(0.0, global_start)
                    overlap_end = min(total_beam_length, global_end)
                    
                    # 1. Valid Force (On Beam)
                    if overlap_end > overlap_start:
                        effective_len = overlap_end - overlap_start
                        check_applied_force_y_clipped += factored_mag * effective_len
                    
                    # 2. Phantom Force (Off Beam) - The culprit of your error!
                    # Case A: Overhang at the end
                    if global_end > total_beam_length + 1e-4:
                        overhang_len = global_end - total_beam_length
                        force_excess = factored_mag * overhang_len
                        phantom_force_y += force_excess
                        overhang_warnings.append(f"• UDL on Span {span_idx+1} extends **{overhang_len:.4f}m** beyond beam end.")
                    
                    # Case B: Starts before beam (rare)
                    if global_start < -1e-4:
                        overhang_len = abs(global_start)
                        phantom_force_y += factored_mag * overhang_len
                        overhang_warnings.append(f"• UDL on Span {span_idx+1} starts **{overhang_len:.4f}m** before beam start.")

        # --- 3. SOLVE ---
        solver = BeamSolver(st.session_state['spans'], st.session_state['supports'], calc_loads, E, I, A, G)
        df, r, summ = solver.solve()
        
        if not df.empty:
            design_view.draw_interactive_diagrams(df, r, st.session_state['spans'], st.session_state['supports'], st.session_state['loads'], dl_factor, ll_factor)
            
            with st.expander("📊 View Critical Values, Reactions & Equilibrium Check", expanded=True):
                # --- CRITICAL VALUES ---
                st.markdown("#### 1. Critical Design Forces (Factored)")
                c1, c2, c3, c4 = st.columns(4)
                v_max_kn = summ['V_max']['value'] / 1000
                m_pos_knm = summ['M_pos']['value'] / 1000
                m_neg_knm = summ['M_neg']['value'] / 1000
                d_mm = summ['D_max']['value'] * 1000
                c1.metric("Max Shear (V_u)", f"{v_max_kn:.2f} kN", f"@ {summ['V_max']['x']:.2f} m")
                c2.metric("Max Moment (+M_u)", f"{m_pos_knm:.2f} kNm", f"@ {summ['M_pos']['x']:.2f} m")
                c3.metric("Max Moment (-M_u)", f"{m_neg_knm:.2f} kNm", f"@ {summ['M_neg']['x']:.2f} m", delta_color="inverse")
                
                limit_val = (total_beam_length / 360) * 1000
                c4.metric("Max Deflection", f"{d_mm:.4f} mm", f"{'✅' if abs(d_mm)<limit_val else '⚠️'} < L/360")

                st.markdown("---")
                
                # --- REACTIONS ---
                st.markdown("#### 2. Support Reactions")
                n_nodes = len(st.session_state['spans']) + 1
                cols = st.columns(n_nodes)
                sup_map = {int(s['id']): s['type'] for s in st.session_state['supports'] if 'id' in s}
                total_react_y = 0.0
                
                for i in range(n_nodes):
                    with cols[i]:
                        Ry = r[2*i]
                        Mz = r[2*i+1]
                        total_react_y += Ry
                        
                        has_sup = i in sup_map
                        has_force = abs(Ry) > 1.0 or abs(Mz) > 1.0
                        
                        if has_sup or has_force:
                            lbl = sup_map.get(i, "Free Node")
                            st.markdown(f"**Node {i+1}** : `{lbl}`")
                            if abs(Ry) > 0.1: st.write(f"Fy: **{Ry/1000:.2f}** kN")
                            elif has_sup: st.write(f"Fy: 0.00 kN")
                            if abs(Mz) > 0.1: st.write(f"Mz: **{Mz/1000:.2f}** kNm")

                # --- ADVANCED EQUILIBRIUM DIAGNOSTICS ---
                st.markdown("---")
                st.markdown("#### 3. Equilibrium Diagnostics")
                
                # Calculate Error
                # Error = Reaction - (Force on Beam)
                raw_diff = total_react_y - check_applied_force_y_clipped
                residual = raw_diff - phantom_force_y # Should be ~0 if Phantom Force explains the error
                
                col_e1, col_e2 = st.columns(2)
                col_e1.write(f"Total Applied Load (On Beam): **{check_applied_force_y_clipped/1000:.2f} kN** (Down)")
                col_e2.write(f"Total Reaction Sum: **{total_react_y/1000:.2f} kN** (Up)")
                
                if abs(raw_diff) < 10.0:
                    st.success(f"✅ Equilibrium Perfect! (Diff: {raw_diff:.2f} N)")
                elif abs(residual) < 10.0:
                    # Case: Error is exactly explained by the Overhang
                    st.warning(f"⚠️ **Warning:** Equilibrium matches if we include **{phantom_force_y/1000:.2f} kN** of load that is hanging off the beam!")
                    st.error(f"❌ **Action Required:** You have load(s) extending beyond the beam length:")
                    for msg in overhang_warnings:
                        st.markdown(msg)
                else:
                    # Case: Real math error
                    st.error(f"❌ Unknown Equilibrium Error: {raw_diff:.2f} N. (Overhang explains {phantom_force_y:.2f} N)")

                st.markdown("---")
                design_view.render_result_tables(df, r, st.session_state['spans'])
