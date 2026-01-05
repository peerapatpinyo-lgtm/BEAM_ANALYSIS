import streamlit as st
import pandas as pd

def render_sidebar():
    st.sidebar.header("⚙️ Design Parameters")
    params = {}
    with st.sidebar.expander("Material Properties", expanded=True):
        params['fc'] = st.number_input("Concrete f'c (ksc)", value=240)
        params['fy'] = st.number_input("Rebar fy (ksc)", value=4000)
        params['E'] = st.number_input("Elastic Modulus E (ksc)", value=2e6)
        
    with st.sidebar.expander("Section Geometry", expanded=True):
        params['b'] = st.number_input("Width b (cm)", value=25)
        params['h'] = st.number_input("Depth h (cm)", value=50)
        params['cover'] = st.number_input("Clear Cover (cm)", value=2.5)
        # Calculate I automatically for analysis
        params['I'] = (params['b']/100 * (params['h']/100)**3) / 12
        st.write(f"Inertia (I) = {params['I']:.6f} m4")
        
    with st.sidebar.expander("Load Factors (SDM)", expanded=False):
        params['gamma_dead'] = st.number_input("Dead Load Factor", value=1.4)
        params['gamma_live'] = st.number_input("Live Load Factor", value=1.7)
        
    with st.sidebar.expander("Rebar Settings", expanded=False):
        params['main_bar'] = st.selectbox("Main Rebar Size", ["DB12", "DB16", "DB20", "DB25", "DB28"], index=1)
        
    return params

def render_model_inputs(params):
    st.header("1. 🏗️ Model Setup")
    col1, col2 = st.columns([1, 2])
    n_spans = col1.number_input("Number of Spans", 1, 10, 2)
    
    spans = []
    cols = st.columns(n_spans)
    for i in range(n_spans):
        spans.append(cols[i].number_input(f"L{i+1} (m)", 1.0, 20.0, 5.0, key=f"span_{i}"))
        
    st.subheader("Supports")
    sup_data = []
    cols_sup = st.columns(n_spans + 1)
    for i in range(n_spans + 1):
        with cols_sup[i]:
            st.caption(f"Node {i}")
            stype = st.selectbox("", ["Pin", "Roller", "Fixed", "None"], index=0 if i==0 else (1 if i<=n_spans else 0), key=f"sup_{i}", label_visibility="collapsed")
            if stype != "None":
                sup_data.append({'id': i, 'type': stype})
    
    return n_spans, spans, pd.DataFrame(sup_data), True

def render_loads(n_spans, spans, params, sup_df):
    st.header("2. ⬇️ Loading")
    
    if 'loads' not in st.session_state:
        st.session_state.loads = []
        
    with st.form("load_form"):
        c1, c2, c3, c4 = st.columns(4)
        l_type = c1.selectbox("Type", ["Uniform Load (w)", "Point Load (P)"])
        span_idx = c2.selectbox("Span", range(n_spans), format_func=lambda x: f"Span {x+1}")
        mag = c3.number_input("Magnitude (kg or kg/m)", value=1000.0)
        
        x_loc = 0.0
        if "Point" in l_type:
            x_loc = c4.number_input("Location x (m from left)", 0.0, spans[span_idx], spans[span_idx]/2)
            
        case = st.radio("Load Case", ["Dead Load (DL)", "Live Load (LL)"], horizontal=True)
        
        if st.form_submit_button("Add Load"):
            st.session_state.loads.append({
                'type': 'P' if 'Point' in l_type else 'U',
                'span_idx': span_idx,
                'mag': mag,
                'x': x_loc,
                'case': 'DL' if 'Dead' in case else 'LL'
            })
            
    # Show Table
    if st.session_state.loads:
        st.table(pd.DataFrame(st.session_state.loads))
        if st.button("Clear All Loads"):
            st.session_state.loads = []
            st.rerun()
            
    return pd.DataFrame(st.session_state.loads)
