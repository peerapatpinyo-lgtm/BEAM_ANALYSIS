import streamlit as st
import pandas as pd
import numpy as np

# Import modules
from solver import BeamSolver
import design_view

# --- Page Config ---
st.set_page_config(page_title="Beam Analysis Pro", layout="wide", page_icon="🏗️")

# --- Session State Init ---
if 'spans' not in st.session_state:
    st.session_state['spans'] = [5.0, 5.0]
if 'supports' not in st.session_state:
    st.session_state['supports'] = [
        {'id': 0, 'type': 'Pin'},
        {'id': 1, 'type': 'Roller'},
        {'id': 2, 'type': 'Roller'}
    ]
if 'loads' not in st.session_state:
    st.session_state['loads'] = []

# --- Sidebar: Settings ---
st.sidebar.title("🏗️ Beam Settings")
st.sidebar.markdown("---")

# Reset Button
if st.sidebar.button("Reset Project", type="primary"):
    st.session_state['spans'] = [5.0]
    st.session_state['supports'] = [{'id': 0, 'type': 'Pin'}, {'id': 1, 'type': 'Roller'}]
    st.session_state['loads'] = []
    st.rerun()

st.sidebar.markdown("### 1. Section Properties")
E = st.sidebar.number_input("Elastic Modulus (E) [Pa]", value=2e11, format="%.2e", help="Concrete approx 2-3e10, Steel 2e11")

# --- NEW: Section Input Logic ---
input_method = st.sidebar.radio(
    "Input Method", 
    ["Rectangular Size (b x h)", "Custom Properties (I, A)"]
)

if input_method == "Rectangular Size (b x h)":
    col_dim1, col_dim2 = st.sidebar.columns(2)
    b = col_dim1.number_input("Width (b) [m]", value=0.30, min_value=0.01, step=0.05)
    h = col_dim2.number_input("Depth (h) [m]", value=0.50, min_value=0.01, step=0.05)
    
    # Calculate Properties
    I = (b * h**3) / 12
    A = b * h
    
    st.sidebar.info(f"Calculated:\nI = {I:.2e} m⁴\nA = {A:.4f} m²")

else:
    # Custom Input (Original)
    I = st.sidebar.number_input("Moment of Inertia (I) [m^4]", value=5e-5, format="%.2e")
    A = st.sidebar.number_input("Cross-sectional Area (A) [m^2]", value=0.01, format="%.4f")

# --- Timoshenko Inputs ---
use_timoshenko = st.sidebar.checkbox("Advanced: Timoshenko Beam", value=False, help="Enable for deep beams (accounts for shear deformation)")

if use_timoshenko:
    st.sidebar.caption("Shear Modulus is required:")
    # A is already determined from above
    G = st.sidebar.number_input("Shear Modulus (G) [Pa]", value=7.7e10, format="%.2e")
else:
    G = None # Let Solver estimate if needed (though mostly unused if not Timoshenko)

st.sidebar.markdown("---")
st.sidebar.markdown("### 2. Load Factors")
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
        
        current_spans = st.session_state['spans']
        if len(current_spans) < num_spans:
            current_spans.extend([5.0] * (num_spans - len(current_spans)))
        elif len(current_spans) > num_spans:
            st.session_state['spans'] = current_spans[:num_spans]
            
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
    
    sup_data = []
    existing_sups = {s['id']: s['type'] for s in st.session_state['supports']}
    
    for i in range(num_nodes):
        stype = existing_sups.get(i, "None")
        sup_data.append({"Node ID": i + 1, "Support Type": stype})
    
    df_sup = pd.DataFrame(sup_data)
    
    edited_df = st.data_editor(
        df_sup,
        column_config={
            "Node ID": st.column_config.NumberColumn(disabled=True, format="%d"),
            "Support Type": st.column_config.SelectboxColumn(
                "Type", options=["None", "Pin", "Roller", "Fixed"], required=True
            )
        },
        hide_index=True,
        use_container_width=True
    )
    
    new_sups = []
    for index, row in edited_df.iterrows():
        if row['Support Type'] != "None":
            internal_id = int(row['Node ID']) - 1
            new_sups.append({'id': internal_id, 'type': row['Support Type']})
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
        
        if "Uniform" in load_type:
            cols_pos = st.columns(2)
            with cols_pos[0]:
                x_start = st.number_input("Start Position (x1) [m]", 
                                          min_value=0.0, max_value=float(current_span_len), value=0.0)
            with cols_pos[1]:
                x_end = st.number_input("End Position (x2) [m]", 
                                        min_value=0.0, max_value=float(current_span_len), value=float(current_span_len))
            dist_val = x_end - x_start
            x_loc = x_start
            if x_end < x_start:
                st.warning("⚠️ End position must be greater than Start position.")
        else:
            x_loc = st.number_input("Position x (m) from left of span", 
                                    min_value=0.0, max_value=float(current_span_len), value=float(current_span_len)/2)
            dist_val = 0 
            
    if st.button("➕ Add Load", type="primary"):
        valid = True
        if "Uniform" in load_type and dist_val <= 0:
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
                'dist': dist_val,
                'case': load_case
            }
            st.session_state['loads'].append(new_load)
            st.success("Load added!")
            st.rerun()

    if st.session_state['loads']:
        st.markdown("##### Current Loads List")
        display_data = []
        for i, l in enumerate(st.session_state['loads']):
            s_num = l['span_idx'] + 1
            l_t = l['type']
            pos_desc = f"x={l['x']:.2f} m"
            if l_t == 'U':
                end_pos = l['x'] + l.get('dist', 0)
                pos_desc = f"x={l['x']:.2f} to {end_pos:.2f} m"
            display_data.append({
                "Index": i+1, "Span": s_num, "Type": l_t, 
                "Mag": l['mag'], "Case": l['case'], "Position": pos_desc
            })
        df_loads = pd.DataFrame(display_data)
        st.dataframe(df_loads, use_container_width=True, hide_index=True)
        
        col_del, _ = st.columns([1, 3])
        with col_del:
            idx_to_del = st.number_input("Remove Load #", min_value=1, max_value=max(1, len(st.session_state['loads'])), step=1)
            if st.button("🗑️ Remove Load"):
                internal_idx = idx_to_del - 1
                if 0 <= internal_idx < len(st.session_state['loads']):
                    st.session_state['loads'].pop(internal_idx)
                    st.rerun()

# --- CALCULATION & RESULTS ---
st.markdown("---")
if st.button("🚀 Run Analysis", type="primary", use_container_width=True):
    
    spans = st.session_state['spans']
    supports_df = pd.DataFrame(st.session_state['supports'])
    loads_df = pd.DataFrame(st.session_state['loads'])
    
    if len(supports_df) < 2:
        st.error("Structure unstable: Need at least 2 supports.")
    else:
        try:
            calc_loads = loads_df.copy()
            if not calc_loads.empty:
                def apply_factor(row):
                    f = dl_factor if row['case'] == 'DL' else ll_factor
                    return row['mag'] * f
                calc_loads['mag'] = calc_loads.apply(apply_factor, axis=1)
            
            solver = BeamSolver(spans, supports_df, calc_loads, E, I, A, G)
            df_res, reactions, summary = solver.solve()
            
            if df_res.empty:
                 st.error("Structure is Unstable or Error in calculation.")
            else:
                st.markdown("### 🎯 Analysis Results")

                # 1. Diagrams
                design_view.draw_interactive_diagrams(
                    df_res, reactions, spans, supports_df, 
                    st.session_state['loads'], 
                    dl_factor=dl_factor, ll_factor=ll_factor
                )
                
                # 2. Results Expander
                with st.expander("📊 View Critical Values & Reactions Details", expanded=False):
                    
                    if summary:
                        st.markdown("#### 1. Critical Design Values (Envelope)")
                        
                        col_c1, col_c2, col_c3, col_c4 = st.columns(4)
                        col_c1.metric("Max Shear", f"{summary['V_max']['value']:.2f}", f"@ {summary['V_max']['x']:.2f} m")
                        col_c2.metric("Max Moment (+)", f"{summary['M_pos']['value']:.2f}", f"@ {summary['M_pos']['x']:.2f} m")
                        col_c3.metric("Max Moment (-)", f"{summary['M_neg']['value']:.2f}", f"@ {summary['M_neg']['x']:.2f} m", delta_color="inverse")
                        
                        with col_c4:
                            limit_val = max(spans)/360 * 1000 
                            actual_val_mm = summary['D_max']['value'] * 1000
                            status_text = "✅ PASS" if abs(actual_val_mm) < limit_val else "⚠️ CHECK"
                            st.metric("Max Deflection", f"{actual_val_mm:.4f} mm", status_text)
                        
                        st.markdown("---")
                        st.markdown("#### 2. Support Reactions")
                        
                        support_ids = supports_df['id'].tolist()
                        r_cols = st.columns(len(support_ids))
                        
                        for idx, node_id in enumerate(support_ids):
                            fy = reactions[2*node_id]
                            mz = reactions[2*node_id+1]
                            sup_type = supports_df[supports_df['id'] == node_id]['type'].values[0]
                            
                            with r_cols[idx]:
                                st.markdown(f"**Node {node_id + 1} ({sup_type})**")
                                st.write(f"Fy: `{fy:.2f}` N")
                                if sup_type == 'Fixed' or abs(mz) > 0.01:
                                    st.write(f"Mz: `{mz:.2f}` Nm")
                        
                        st.markdown("---")
                    
                    st.markdown("#### 3. Detailed Data Table")
                    design_view.render_result_tables(df_res, reactions, spans)
            
        except Exception as e:
            st.error(f"Analysis Error: {str(e)}")
            st.code(e)
