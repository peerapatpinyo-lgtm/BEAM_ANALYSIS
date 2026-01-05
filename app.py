import streamlit as st
import pandas as pd
import numpy as np

# Import modules (ตรวจสอบว่าไฟล์ solver.py และ design_view.py อยู่ในโฟลเดอร์เดียวกัน)
from solver import BeamSolver
import design_view

# --- Page Config ---
st.set_page_config(page_title="Beam Analysis Pro", layout="wide", page_icon="🏗️")

# --- Session State Init ---
if 'spans' not in st.session_state:
    st.session_state['spans'] = [5.0, 5.0] # Default 2 spans
if 'supports' not in st.session_state:
    # Default: Pin at start, Roller at ends
    st.session_state['supports'] = [
        {'id': 0, 'type': 'Pin'},
        {'id': 1, 'type': 'Roller'},
        {'id': 2, 'type': 'Roller'}
    ]
if 'loads' not in st.session_state:
    st.session_state['loads'] = []

# --- Sidebar ---
st.sidebar.title("🏗️ Beam Settings")
st.sidebar.markdown("---")

# Reset Button
if st.sidebar.button("Reset Project", type="primary"):
    st.session_state['spans'] = [5.0]
    st.session_state['supports'] = [{'id': 0, 'type': 'Pin'}, {'id': 1, 'type': 'Roller'}]
    st.session_state['loads'] = []
    st.rerun()

st.sidebar.markdown("### Design Parameters")
E = st.sidebar.number_input("Elastic Modulus (E) [Pa]", value=2e11, format="%.2e")
I = st.sidebar.number_input("Moment of Inertia (I) [m^4]", value=5e-5, format="%.2e")

# --- Timoshenko Inputs (Optional) ---
use_timoshenko = st.sidebar.checkbox("Advanced: Timoshenko Beam", value=False, help="Enable for deep beams (accounts for shear deformation)")
if use_timoshenko:
    st.sidebar.caption("Required for Timoshenko:")
    A = st.sidebar.number_input("Cross-sectional Area (A) [m^2]", value=0.01, format="%.4f")
    G = st.sidebar.number_input("Shear Modulus (G) [Pa]", value=7.7e10, format="%.2e")
else:
    A, G = None, None # ให้ Solver คำนวณ Default เอง

st.sidebar.markdown("---")
st.sidebar.markdown("### Load Factors")
dl_factor = st.sidebar.number_input("Dead Load Factor", value=1.4, step=0.1)
ll_factor = st.sidebar.number_input("Live Load Factor", value=1.7, step=0.1)

# --- Main Interface ---
st.title("🏗️ Structural Beam Analysis (Exact FEM)")

# Tabs for input steps
tab1, tab2, tab3 = st.tabs(["1️⃣ Geometry (Spans)", "2️⃣ Supports", "3️⃣ Applied Loads"])

# --- TAB 1: SPANS ---
with tab1:
    st.subheader("Define Beam Spans")
    col_s1, col_s2 = st.columns([2, 1])
    with col_s1:
        num_spans = st.number_input("Number of Spans", min_value=1, max_value=10, value=len(st.session_state['spans']))
        
        # Adjust list size
        current_spans = st.session_state['spans']
        if len(current_spans) < num_spans:
            current_spans.extend([5.0] * (num_spans - len(current_spans)))
        elif len(current_spans) > num_spans:
            st.session_state['spans'] = current_spans[:num_spans]
            
        # Inputs for each span
        new_spans = []
        cols = st.columns(min(num_spans, 4))
        for i in range(num_spans):
            with cols[i % 4]:
                val = st.number_input(f"Span {i+1} Length (m)", value=float(current_spans[i]), min_value=0.1, key=f"span_{i}")
                new_spans.append(val)
        st.session_state['spans'] = new_spans
        
    st.info(f"Total Length: {sum(new_spans):.2f} m")

# --- TAB 2: SUPPORTS ---
with tab2:
    st.subheader("Define Supports")
    num_nodes = len(st.session_state['spans']) + 1
    
    # Create a DataFrame for editing
    sup_data = []
    existing_sups = {s['id']: s['type'] for s in st.session_state['supports']}
    
    for i in range(num_nodes):
        stype = existing_sups.get(i, "None")
        sup_data.append({"Node ID": i, "Support Type": stype})
    
    df_sup = pd.DataFrame(sup_data)
    
    edited_df = st.data_editor(
        df_sup,
        column_config={
            "Node ID": st.column_config.NumberColumn(disabled=True),
            "Support Type": st.column_config.SelectboxColumn(
                "Type", options=["None", "Pin", "Roller", "Fixed"], required=True
            )
        },
        hide_index=True,
        use_container_width=True
    )
    
    # Save back to session state
    new_sups = []
    for index, row in edited_df.iterrows():
        if row['Support Type'] != "None":
            new_sups.append({'id': int(row['Node ID']), 'type': row['Support Type']})
    st.session_state['supports'] = new_sups

# --- TAB 3: LOADS ---
with tab3:
    st.subheader("Add Applied Loads")
    
    c1, c2, c3 = st.columns([1, 1, 2])
    
    with c1:
        span_idx_load = st.selectbox("Select Span", options=list(range(len(st.session_state['spans']))), format_func=lambda x: f"Span {x+1}")
        current_span_len = st.session_state['spans'][span_idx_load]
        st.caption(f"Span Length: {current_span_len} m")
        
    with c2:
        load_type = st.selectbox("Load Type", ["Point Load (P)", "Uniform Load (U)", "Moment (M)"])
        load_case = st.selectbox("Load Case", ["DL", "LL"])

    with c3:
        mag = st.number_input("Magnitude (kg, kg/m, kg-m)", value=1000.0)
        
        # --- UI LOGIC FOR LOAD POSITION ---
        if "Uniform" in load_type:
            # Inputs for Uniform Load Start/End
            cols_pos = st.columns(2)
            with cols_pos[0]:
                x_start = st.number_input("Start Position (x1) [m]", 
                                          min_value=0.0, max_value=float(current_span_len), value=0.0)
            with cols_pos[1]:
                x_end = st.number_input("End Position (x2) [m]", 
                                        min_value=0.0, max_value=float(current_span_len), value=float(current_span_len))
            
            # Calculation for internal logic
            dist_val = x_end - x_start
            x_loc = x_start
            
            if x_end < x_start:
                st.warning("⚠️ End position must be greater than Start position.")
        
        else:
            # Point Load or Moment
            x_loc = st.number_input("Position x (m) from left of span", 
                                    min_value=0.0, max_value=float(current_span_len), value=float(current_span_len)/2)
            dist_val = 0 # Not used for Point/Moment
            
    if st.button("➕ Add Load", type="primary"):
        # Validate Uniform Load
        valid = True
        if "Uniform" in load_type:
            if dist_val <= 0:
                st.error("Error: Uniform load length must be greater than 0.")
                valid = False
        
        if valid:
            l_type_code = 'P'
            if 'Uniform' in load_type: l_type_code = 'U'
            elif 'Moment' in load_type: l_type_code = 'M'
            
            new_load = {
                'span_idx': span_idx_load,
                'type': l_type_code,
                'mag': mag,
                'x': x_loc,
                'dist': dist_val, # Save the calculated distance
                'case': load_case
            }
            st.session_state['loads'].append(new_load)
            st.success("Load added!")
            st.rerun()

    # Display Loads Table
    if st.session_state['loads']:
        st.markdown("##### Current Loads List")
        # Process data for display
        display_data = []
        for i, l in enumerate(st.session_state['loads']):
            s_num = l['span_idx'] + 1
            l_t = l['type']
            
            pos_desc = f"x={l['x']:.2f} m"
            if l_t == 'U':
                end_pos = l['x'] + l.get('dist', 0)
                pos_desc = f"x={l['x']:.2f} to {end_pos:.2f} m"
                
            display_data.append({
                "Index": i,
                "Span": s_num,
                "Type": l_t,
                "Mag": l['mag'],
                "Case": l['case'],
                "Position": pos_desc
            })
            
        df_loads = pd.DataFrame(display_data)
        st.dataframe(df_loads, use_container_width=True, hide_index=True)
        
        # Remove Load
        col_del, _ = st.columns([1, 3])
        with col_del:
            idx_to_del = st.number_input("Remove Load Index", min_value=0, max_value=max(0, len(st.session_state['loads'])-1), step=1)
            if st.button("🗑️ Remove Load"):
                if 0 <= idx_to_del < len(st.session_state['loads']):
                    st.session_state['loads'].pop(idx_to_del)
                    st.rerun()

# --- CALCULATION & RESULTS ---
st.markdown("---")
if st.button("🚀 Run Analysis", type="primary", use_container_width=True):
    
    # Prepare Data
    spans = st.session_state['spans']
    supports_df = pd.DataFrame(st.session_state['supports'])
    loads_df = pd.DataFrame(st.session_state['loads'])
    
    # Check minimum stability (Basic check)
    if len(supports_df) < 2:
        st.error("Structure unstable: Need at least 2 supports.")
    else:
        try:
            # FACTORED LOADS CALCULATION
            # Create a copy of loads to apply factors before sending to solver
            calc_loads = loads_df.copy()
            if not calc_loads.empty:
                # Apply factors based on 'case'
                def apply_factor(row):
                    f = dl_factor if row['case'] == 'DL' else ll_factor
                    return row['mag'] * f
                
                calc_loads['mag'] = calc_loads.apply(apply_factor, axis=1)
            
            # Initialize Solver (With A and G for Timoshenko)
            solver = BeamSolver(spans, supports_df, calc_loads, E, I, A, G)
            
            # Solve (รับค่า 3 ตัว: df, reactions, summary)
            df_res, reactions, summary = solver.solve()
            
            # --- 1. แสดง Dashboard สรุปค่า Critical ---
            if summary:
                st.markdown("### 📊 Critical Design Values (Envelope)")
                col1, col2, col3, col4 = st.columns(4)
                
                # Max Shear
                col1.metric("Max Shear", 
                            f"{summary['V_max']['value']:.2f}", 
                            f"@ x = {summary['V_max']['x']:.2f} m")
                
                # Max Moment (+)
                col2.metric("Max Moment (+)", 
                            f"{summary['M_pos']['value']:.2f}", 
                            f"@ x = {summary['M_pos']['x']:.2f} m")
                
                # Max Moment (-)
                col3.metric("Max Moment (-)", 
                            f"{summary['M_neg']['value']:.2f}", 
                            f"@ x = {summary['M_neg']['x']:.2f} m", 
                            delta_color="inverse")
                
                # Max Deflection
                col4.metric("Max Deflection", 
                            f"{summary['D_max']['value']*1000:.4f} mm", # แปลงเป็น mm
                            f"@ x = {summary['D_max']['x']:.2f} m")
                
                st.divider()

            # --- 2. เรียกใช้การวาดกราฟแบบเดิม (design_view) ---
            design_view.draw_interactive_diagrams(
                df_res, reactions, spans, supports_df, 
                st.session_state['loads'], # Original inputs
                dl_factor=dl_factor, ll_factor=ll_factor
            )
            
            # --- 3. ตารางสรุป Reaction ---
            design_view.render_result_tables(df_res, reactions, spans)
            
        except Exception as e:
            st.error(f"Analysis Failed: {str(e)}")
            st.code(e)
