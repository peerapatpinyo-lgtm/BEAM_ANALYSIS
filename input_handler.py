import streamlit as st
import pandas as pd

def render_sidebar():
    st.sidebar.header("⚙️ Settings")
    
    # 1. Units
    st.sidebar.subheader("1. Units")
    unit_sys = st.sidebar.radio("System", ["Metric (kg, m)", "SI (kN, m)"])
    if "kg" in unit_sys:
        u_force, u_len = "kg", "m"
    else:
        u_force, u_len = "kN", "m"
        
    # 2. Material
    st.sidebar.subheader("2. Material Properties")
    E = st.sidebar.number_input("Elastic Modulus (E)", value=2e6, step=1e5, format="%.2e", help="Young's Modulus")
    I = st.sidebar.number_input("Moment of Inertia (I) [cm^4]", value=10000.0, step=100.0)
    
    # Conversion Factors to consistent units (Force-meter)
    # Assumes Input E is in ksc or kPa/MPa? Let's generalize:
    # If Metric (kg, m): E usually ksc (kg/cm^2). Need to convert to kg/m^2 -> * 10^4
    # If SI (kN, m): E usually kPa/MPa (kN/m^2). No conversion if in kN/m^2. 
    # Let's simplify: User inputs E in consistent calculation units or we assume standard
    
    # Better: Assume user enters E in consistent Force/Area units relative to Force input
    # But usually E is huge. Let's keep the previous scaler for UX convenience
    I_m4 = I * 1e-8 
    E_calc = E * 10000 if "kg" in u_force else E * 1000 # Scaling ksc->kg/m2 or MPa->kPa
    
    return {'u_force': u_force, 'u_len': u_len, 'E': E_calc, 'I': I_m4}

def render_model_inputs(params):
    st.subheader("1. Structure Model")
    
    # Spans
    c1, c2 = st.columns([1, 2])
    with c1:
        n_spans = st.number_input("Number of Spans", min_value=1, max_value=10, value=2)
    
    st.write(f"**Span Lengths ({params['u_len']})**")
    spans = []
    cols = st.columns(min(n_spans, 5))
    for i in range(n_spans):
        with cols[i % 5]: 
            val = st.number_input(f"Span {i+1}", min_value=0.1, value=4.0, step=0.1, format="%.2f", key=f"len_{i}")
            spans.append(val)
            
    # Supports (Dropdown enforced via DataEditor)
    st.write("**Supports Conditions**")
    
    # Default Logic: Pin at Start, Roller at others
    default_types = ['Pin'] + ['Roller'] * (n_spans-1) + ['Roller']
    
    # Prepare DataFrame
    sup_data = []
    for i in range(n_spans + 1):
        sup_data.append({
            "Node ID": i+1,
            "Condition": default_types[i] if i < len(default_types) else 'Roller'
        })
    
    df_sup = pd.DataFrame(sup_data)
    
    edited_df = st.data_editor(
        df_sup,
        column_config={
            "Node ID": st.column_config.TextColumn("Node", disabled=True),
            "Condition": st.column_config.SelectboxColumn(
                "Support Type (Dropdown)", 
                options=['Pin', 'Roller', 'Fixed', 'None'], 
                required=True,
                width="medium"
            )
        },
        hide_index=True,
        use_container_width=True,
        key="support_editor"
    )
    
    # Convert back to list of dicts for solver
    sup_config = []
    for _, row in edited_df.iterrows():
        if row['Condition'] != 'None':
            sup_config.append({'id': row['Node ID']-1, 'type': row['Condition']})
            
    sup_df = pd.DataFrame(sup_config)
    
    # Stability Check (Basic)
    stable = len(sup_df) >= 2 or any(s['type'] == 'Fixed' for s in sup_config)
    
    return n_spans, spans, sup_df, stable

def render_loads(n_spans, spans, params):
    st.subheader("2. Applied Loads")
    
    if 'loads' not in st.session_state:
        st.session_state['loads'] = []

    # --- ADD NEW LOAD FORM ---
    with st.expander("➕ Add Load (Form)", expanded=True):
        c1, c2, c3, c4 = st.columns([1.5, 0.8, 1, 1])
        with c1:
            l_type = st.selectbox("Load Type", ["Point Load (P)", "Uniform Load (w)", "Moment Load (M)"])
        with c2:
            span_idx = st.selectbox("Span #", range(1, n_spans+1)) - 1
            max_len = spans[span_idx]
        with c3:
            # Dynamic Label
            if "Moment" in l_type:
                unit_label = f"{params['u_force']}-{params['u_len']}"
            elif "Uniform" in l_type:
                unit_label = f"{params['u_force']}/{params['u_len']}"
            else:
                unit_label = params['u_force']
                
            mag = st.number_input(f"Magnitude ({unit_label})", value=1000.0, step=100.0, key="input_mag")
            
        with c4:
            if "Uniform" not in l_type:
                loc = st.number_input(f"Distance @ x ({params['u_len']})", min_value=0.0, max_value=float(max_len), value=float(max_len)/2, step=0.1, key="input_loc")
            else:
                st.info("Full Span")
                loc = 0
                
        if st.button("Add to List", type="secondary"):
            # Map Type Code
            if "Point" in l_type: code = 'P'
            elif "Uniform" in l_type: code = 'U'
            else: code = 'M'
            
            new_load = {'type': code, 'span_idx': span_idx, 'mag': mag, 'x': loc}
            st.session_state['loads'].append(new_load)
            st.rerun()

    # --- INTERACTIVE TABLE (With @ Column) ---
    if st.session_state['loads']:
        st.write("**Current Loads List (Edit values directly below)**")
        
        # Prepare Display Data
        display_data = []
        for l in st.session_state['loads']:
            t_str = "Point (P)"
            if l['type'] == 'U': t_str = "Uniform (w)"
            elif l['type'] == 'M': t_str = "Moment (M)"
            
            display_data.append({
                "Type": t_str,
                "Span": l['span_idx'] + 1,
                "Mag": l['mag'],
                "Dist": l['x'] if l['type'] != 'U' else 0,
                "Delete": False
            })
        
        df_editor = pd.DataFrame(display_data)
        
        edited_df = st.data_editor(
            df_editor,
            column_config={
                "Type": st.column_config.TextColumn("Type", disabled=True),
                "Span": st.column_config.NumberColumn("Span", min_value=1, max_value=n_spans, disabled=True),
                "Mag": st.column_config.NumberColumn(f"Magnitude (+/-)", required=True),
                "Dist": st.column_config.NumberColumn(f"Location @ x ({params['u_len']})", min_value=0, required=True),
                "Delete": st.column_config.CheckboxColumn("Del", default=False)
            },
            hide_index=True,
            use_container_width=True,
            key="load_editor"
        )

        # Sync back to Session State
        new_loads_state = []
        for idx, row in edited_df.iterrows():
            if not row['Delete']:
                # Recover type code
                raw_type = st.session_state['loads'][idx]['type']
                
                # Validation: Location cannot exceed span length
                curr_span = row['Span'] - 1
                limit = spans[curr_span]
                safe_x = min(row['Dist'], limit)
                
                new_loads_state.append({
                    'type': raw_type,
                    'span_idx': curr_span,
                    'mag': row['Mag'],
                    'x': safe_x if raw_type != 'U' else 0
                })
        
        # Update State only if changed to prevent infinite loops
        if new_loads_state != st.session_state['loads']:
            st.session_state['loads'] = new_loads_state
            st.rerun()

    return st.session_state['loads']
