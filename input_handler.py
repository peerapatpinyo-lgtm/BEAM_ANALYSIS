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
        
    # 2. Material (Needed for Deflection)
    st.sidebar.subheader("2. Material Properties")
    E = st.sidebar.number_input("Elastic Modulus (E) [ksc/MPa]", value=2e6, step=1e5, format="%.2e")
    I = st.sidebar.number_input("Moment of Inertia (I) [cm^4]", value=10000.0, step=100.0)
    
    # Convert I to m^4 for calculation (Assuming input is cm^4)
    I_m4 = I * 1e-8 
    # Convert E to force/m^2 (Assuming ksc to kg/m2 or MPa to kN/m2 roughly for generic solver)
    # Note: Simplified unit conversion for this demo context
    E_calc = E * 10000 if "kg" in u_force else E * 1000 # Just a scale factor for solver
    
    return {'u_force': u_force, 'u_len': u_len, 'E': E_calc, 'I': I_m4}

def render_model_inputs(params):
    st.subheader("1. Structure Model")
    
    # 1. Spans
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
            
    # 2. Supports
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
    
    # Stability Check
    stable = len(sup_df) >= 2 or any(s['type'] == 'Fixed' for s in sup_config)
        
    return n_spans, spans, sup_df, stable

def render_loads(n_spans, spans, params):
    st.subheader("2. Loads")
    
    if 'loads' not in st.session_state:
        st.session_state['loads'] = []

    with st.container():
        c1, c2, c3, c4 = st.columns([1.5, 1, 1, 1])
        with c1:
            l_type = st.radio("Load Type", ["Point Load", "Uniform Load"], horizontal=True)
        with c2:
            span_idx = st.selectbox("Span No.", range(1, n_spans+1)) - 1
            max_len = spans[span_idx]
        with c3:
            mag = st.number_input(f"Magnitude ({params['u_force']})", value=1000.0, step=100.0)
        
        with c4:
            if "Point" in l_type:
                loc = st.number_input(f"Distance x ({params['u_len']})", 
                                    min_value=0.0, max_value=float(max_len), 
                                    value=float(max_len)/2, step=0.1)
                if st.button("➕ Add"):
                    st.session_state['loads'].append({'type': 'P', 'span_idx': span_idx, 'P': mag, 'x': loc})
            else:
                st.info("Full Span")
                st.write("") 
                if st.button("➕ Add"):
                    st.session_state['loads'].append({'type': 'U', 'span_idx': span_idx, 'w': mag})

    if st.session_state['loads']:
        st.markdown("---")
        load_data = []
        for i, l in enumerate(st.session_state['loads']):
            desc = f"P = {l['P']} @ {l['x']}m" if l['type'] == 'P' else f"w = {l['w']} (Full Span)"
            load_data.append({"No.": i+1, "Span": l['span_idx']+1, "Description": desc})
            
        c_table, c_del = st.columns([8, 2])
        with c_table:
            st.dataframe(pd.DataFrame(load_data), hide_index=True, use_container_width=True)
        with c_del:
            if st.button("Clear All"):
                st.session_state['loads'] = []
                st.rerun()

    return st.session_state['loads']
