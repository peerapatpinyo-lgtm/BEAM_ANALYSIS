import streamlit as st
import pandas as pd

def render_sidebar():
    st.sidebar.header("⚙️ Settings")
    st.sidebar.subheader("1. Units")
    unit_sys = st.sidebar.radio("System", ["Metric (kg, m)", "SI (kN, m)"])
    if "kg" in unit_sys:
        u_force, u_len = "kg", "m"
    else:
        u_force, u_len = "kN", "m"
        
    st.sidebar.subheader("2. Material Properties")
    E = st.sidebar.number_input("Elastic Modulus (E)", value=2e6, format="%.2e")
    I = st.sidebar.number_input("Moment of Inertia (I) [cm^4]", value=10000.0)
    
    I_m4 = I * 1e-8 
    E_calc = E * 10000 if "kg" in u_force else E * 1000 
    
    return {'u_force': u_force, 'u_len': u_len, 'E': E_calc, 'I': I_m4}

def render_model_inputs(params):
    st.subheader("1. Structure Model")
    c1, c2 = st.columns([1, 2])
    with c1:
        n_spans = st.number_input("Number of Spans", min_value=1, max_value=10, value=2)
    
    spans = []
    cols = st.columns(min(n_spans, 5))
    for i in range(n_spans):
        with cols[i % 5]: 
            val = st.number_input(f"Span {i+1}", min_value=0.1, value=4.0, key=f"len_{i}")
            spans.append(val)
            
    st.write("**Supports Conditions**")
    default_types = ['Pin'] + ['Roller'] * (n_spans-1) + ['Roller']
    sup_data = [{"Node": i+1, "Type": default_types[i] if i < len(default_types) else 'Roller'} for i in range(n_spans + 1)]
    
    edited_df = st.data_editor(
        pd.DataFrame(sup_data),
        column_config={
            "Node": st.column_config.TextColumn("Node", disabled=True),
            "Type": st.column_config.SelectboxColumn("Type", options=['Pin', 'Roller', 'Fixed', 'None'], required=True)
        },
        hide_index=True, use_container_width=True
    )
    
    sup_config = [{'id': r['Node']-1, 'type': r['Type']} for _, r in edited_df.iterrows() if r['Type'] != 'None']
    sup_df = pd.DataFrame(sup_config)
    stable = len(sup_df) >= 2 or any(s['type'] == 'Fixed' for s in sup_config)
    
    return n_spans, spans, sup_df, stable

def render_loads(n_spans, spans, params, sup_df):
    st.subheader("2. Applied Loads")
    
    if 'loads' not in st.session_state: st.session_state['loads'] = []

    # --- ADD LOAD FORM ---
    with st.expander("➕ Add Load", expanded=True):
        c1, c2, c3, c4 = st.columns([1.5, 0.8, 1, 1])
        with c1: l_type = st.selectbox("Type", ["Point Load (P)", "Uniform Load (w)", "Moment Load (M)"])
        with c2: span_idx = st.selectbox("Span", range(1, n_spans+1)) - 1
        with c3: 
            u_label = f"{params['u_force']}-{params['u_len']}" if "Moment" in l_type else params['u_force']
            mag = st.number_input(f"Mag ({u_label})", value=1000.0, step=100.0)
        with c4:
            max_len = spans[span_idx]
            loc = st.number_input(f"Dist @ ({params['u_len']})", 0.0, float(max_len), float(max_len)/2) if "Uniform" not in l_type else 0

        if st.button("Add Load"):
            code = 'P' if "Point" in l_type else ('U' if "Uniform" in l_type else 'M')
            st.session_state['loads'].append({'type': code, 'span_idx': span_idx, 'mag': mag, 'x': loc})
            st.rerun()

    # --- LOAD TABLE ---
    if st.session_state['loads']:
        st.write("**Current Loads List**")
        df_disp = pd.DataFrame([{
            "Type": "Point" if l['type']=='P' else ("Uniform" if l['type']=='U' else "Moment"),
            "Span": l['span_idx']+1,
            "Mag": l['mag'],
            "Dist": l['x'],
            "Del": False
        } for l in st.session_state['loads']])
        
        edited = st.data_editor(df_disp, column_config={
            "Type": st.column_config.TextColumn(disabled=True),
            "Span": st.column_config.NumberColumn(disabled=True),
            "Mag": st.column_config.NumberColumn(f"Magnitude (+/-)"),
            "Dist": st.column_config.NumberColumn(f"@ Distance"),
            "Del": st.column_config.CheckboxColumn("Delete")
        }, hide_index=True, use_container_width=True)
        
        # Logic update
        new_loads = []
        for i, row in edited.iterrows():
            if not row['Del']:
                orig = st.session_state['loads'][i]
                new_loads.append({'type': orig['type'], 'span_idx': orig['span_idx'], 'mag': row['Mag'], 'x': row['Dist']})
        
        if new_loads != st.session_state['loads']:
            st.session_state['loads'] = new_loads
            st.rerun()

    # --- ENGINEERING WARNING CHECK ---
    # Check if Moment Load is applied at a Pin/Roller
    if not sup_df.empty:
        cum_spans = [0] + list(pd.Series(spans).cumsum())
        for l in st.session_state['loads']:
            if l['type'] == 'M':
                abs_x = cum_spans[l['span_idx']] + l['x']
                # Check against supports
                for _, s in sup_df.iterrows():
                    sup_x = cum_spans[int(s['id'])]
                    if abs(abs_x - sup_x) < 0.01 and s['type'] in ['Pin', 'Roller']:
                        st.warning(f"⚠️ **Engineering Check:** You applied a Moment Load at Node {int(s['id'])+1} ({s['type']}). "
                                   "Ensure this is intentional (e.g., motor torque). The diagram will show a non-zero moment at this support.")

    return st.session_state['loads']
