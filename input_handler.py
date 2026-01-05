import streamlit as st
import pandas as pd

def render_sidebar():
    st.sidebar.header("1. Global Parameters")
    params = {}
    params['E'] = st.sidebar.number_input("Elastic Modulus (E) [ksc]", value=2e6, step=1e5)
    params['I'] = st.sidebar.number_input("Moment of Inertia (I) [m4]", value=0.0001, format="%.6f")
    
    st.sidebar.header("2. Design Code (ACI/EIT)")
    params['gamma_dead'] = st.sidebar.number_input("Dead Load Factor", value=1.4)
    params['gamma_live'] = st.sidebar.number_input("Live Load Factor", value=1.7)
    params['fc'] = st.sidebar.number_input("f'c (ksc)", value=240)
    params['fy'] = st.sidebar.number_input("fy (ksc)", value=4000)
    params['b'] = st.sidebar.number_input("Beam Width (cm)", value=25)
    params['h'] = st.sidebar.number_input("Beam Depth (cm)", value=50)
    params['cover'] = st.sidebar.number_input("Cover (cm)", value=2.5)
    params['main_bar'] = st.sidebar.selectbox("Main Bar Size", ["DB12", "DB16", "DB20", "DB25"])
    
    params['u_force'] = "kg"
    params['u_len'] = "m"
    return params

def render_model_inputs(params):
    st.header("📍 Geometry & Supports")
    col1, col2 = st.columns([1, 2])
    with col1:
        n_spans = st.number_input("Number of Spans", min_value=1, max_value=10, value=2)
    
    # Spans
    spans = []
    cols = st.columns(n_spans)
    for i in range(n_spans):
        with cols[i]:
            s = st.number_input(f"Span {i+1} (m)", min_value=0.1, value=4.0, key=f"s_{i}")
            spans.append(s)
            
    # Supports
    st.subheader("Supports (Left to Right)")
    n_nodes = n_spans + 1
    sup_data = []
    cols_sup = st.columns(n_nodes)
    for i in range(n_nodes):
        with cols_sup[i]:
            st.write(f"Node {i}")
            sType = st.selectbox(f"Type {i}", ["Pin", "Roller", "Fixed", "None"], index=1 if i>0 else 0, key=f"sup_{i}")
            if sType != "None":
                sup_data.append({'id': i, 'type': sType})
                
    sup_df = pd.DataFrame(sup_data)
    
    # Check Stability (Basic)
    stable = True
    if len(sup_df) < 2 and not any(sup_df['type'] == 'Fixed'):
        stable = False
        st.warning("⚠️ Structure might be unstable!")
        
    return n_spans, spans, sup_df, stable

def render_loads(n_spans, spans, params, sup_df):
    st.header("⬇️ Applied Loads")
    
    if 'load_list' not in st.session_state:
        st.session_state.load_list = []
        
    with st.form("add_load_form"):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            l_type = st.selectbox("Type", ["Point (P)", "Uniform (w)"])
        with c2:
            span_idx = st.selectbox("Span Index", range(n_spans), format_func=lambda x: f"Span {x+1}")
        with c3:
            mag = st.number_input(f"Magnitude ({params['u_force']})", min_value=0.0, value=1000.0)
        with c4:
            x_loc = st.number_input("Location x (m)", min_value=0.0, max_value=spans[span_idx], value=spans[span_idx]/2)
            
        case_type = st.radio("Load Case", ["DL", "LL"], horizontal=True)
        submitted = st.form_submit_button("Add Load")
        
        if submitted:
            st.session_state.load_list.append({
                'type': 'P' if "Point" in l_type else 'U',
                'span_idx': span_idx,
                'mag': mag,
                'x': x_loc,
                'case': case_type
            })
            
    # Show List
    if st.session_state.load_list:
        df_loads = pd.DataFrame(st.session_state.load_list)
        st.dataframe(df_loads, use_container_width=True)
        if st.button("Clear All Loads"):
            st.session_state.load_list = []
            st.rerun()
        return df_loads
    return pd.DataFrame()
