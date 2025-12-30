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
    E = st.sidebar.number_input("Elastic Modulus (E) [ksc/MPa]", value=2e6, step=1e5, format="%.2e")
    I = st.sidebar.number_input("Moment of Inertia (I) [cm^4]", value=10000.0, step=100.0)
    
    # Unit Conversion Factors (Simplified)
    I_m4 = I * 1e-8 
    E_calc = E * 10000 if "kg" in u_force else E * 1000 
    
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
            val = st.number_input(f"L{i+1}", min_value=0.1, value=4.0, step=0.1, format="%.2f", key=f"len_{i}")
            spans.append(val)
            
    # Supports
    st.write("**Supports**")
    default_types = ['Pin'] + ['Roller'] * (n_spans-1) + ['Roller']
    sup_data = [{"Node": i+1, "Type": default_types[i] if i < len(default_types) else 'Roller'} for i in range(n_spans + 1)]
    
    edited_df = st.data_editor(
        pd.DataFrame(sup_data),
        column_config={
            "Node": st.column_config.TextColumn("Node", disabled=True),
            "Type": st.column_config.SelectboxColumn("Support Type", options=['Pin', 'Roller', 'Fixed', 'None'], required=True)
        },
        hide_index=True,
        use_container_width=True
    )
    
    sup_config = [{'id': i, 'type': row['Type']} for i, row in edited_df.iterrows() if row['Type'] != 'None']
    sup_df = pd.DataFrame(sup_config)
    
    stable = len(sup_df) >= 2 or any(s['type'] == 'Fixed' for s in sup_config)
    return n_spans, spans, sup_df, stable

def render_loads(n_spans, spans, params):
    st.subheader("2. Applied Loads")
    
    if 'loads' not in st.session_state:
        st.session_state['loads'] = []

    # --- INPUT FORM (Add New) ---
    with st.expander("➕ Add New Load", expanded=True):
        c1, c2, c3, c4, c5 = st.columns([1.2, 1, 1, 1, 0.5])
        with c1:
            l_type = st.selectbox("Type", ["Point Load (P)", "Uniform Load (w)"])
        with c2:
            span_idx = st.selectbox("Span", range(1, n_spans+1)) - 1
            max_len = spans[span_idx]
        with c3:
            mag = st.number_input(f"Mag ({params['u_force']})", value=1000.0, step=100.0, key="new_mag")
        with c4:
            if "Point" in l_type:
                loc = st.number_input(f"Dist x ({params['u_len']})", min_value=0.0, max_value=float(max_len), value=float(max_len)/2, step=0.1, key="new_loc")
            else:
                st.info("Full Span")
                loc = 0
        with c5:
            st.write("") # Spacer
            st.write("") 
            if st.button("Add"):
                new_load = {'type': 'P' if "Point" in l_type else 'U', 'span_idx': span_idx, 'mag': mag, 'x': loc}
                st.session_state['loads'].append(new_load)
                st.rerun()

    # --- EDITABLE TABLE ---
    if st.session_state['loads']:
        st.write("**Current Loads (Edit or Delete)**")
        
        # Prepare data for editor
        display_data = []
        for l in st.session_state['loads']:
            display_data.append({
                "Type": "Point" if l['type'] == 'P' else "Uniform",
                "Span": l['span_idx'] + 1,
                "Magnitude": l['mag'],
                "Location": l['x'] if l['type'] == 'P' else 0, # 0 placeholder for UDL
                "Delete": False
            })
        
        df_editor = pd.DataFrame(display_data)
        
        edited_df = st.data_editor(
            df_editor,
            column_config={
                "Type": st.column_config.TextColumn("Type", disabled=True),
                "Span": st.column_config.NumberColumn("Span No.", min_value=1, max_value=n_spans, step=1, disabled=True),
                "Magnitude": st.column_config.NumberColumn(f"Magnitude ({params['u_force']})"),
                "Location": st.column_config.NumberColumn(f"Location x ({params['u_len']}) - (Point Load Only)", min_value=0),
                "Delete": st.column_config.CheckboxColumn("Delete?", default=False)
            },
            hide_index=True,
            use_container_width=True
        )

        # Update Session State based on Editor
        new_loads_state = []
        for idx, row in edited_df.iterrows():
            if not row['Delete']: # Keep if not checked
                original_type = st.session_state['loads'][idx]['type']
                # Validate Location for Point Load
                current_span = row['Span'] - 1
                max_l = spans[current_span]
                safe_x = min(row['Location'], max_l)
                
                new_loads_state.append({
                    'type': original_type,
                    'span_idx': current_span,
                    'mag': row['Magnitude'], # Updated Value
                    'x': safe_x if original_type == 'P' else 0 # Updated Location
                })
        
        # Sync if changed
        if new_loads_state != st.session_state['loads']:
            st.session_state['loads'] = new_loads_state
            st.rerun()

    return st.session_state['loads']
