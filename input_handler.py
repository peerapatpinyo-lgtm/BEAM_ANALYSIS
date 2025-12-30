import streamlit as st
import pandas as pd
import numpy as np

def render_sidebar():
    """Renders the sidebar inputs and returns a dictionary of parameters."""
    
    # Material Properties
    st.markdown("### 🧱 Material Properties")
    fc = st.number_input("Concrete fc' (ksc)", value=240.0, step=10.0, help="Cylinder Strength")
    fy = st.number_input("Rebar fy (ksc)", value=4000.0, step=100.0, help="Yield Strength (SD40)")
    
    # Units (For display mainly, calculation uses SI/Metric consistency)
    u_len = "m"
    u_force = "kg"
    
    return {
        "fc": fc,
        "fy": fy,
        "u_len": u_len,
        "u_force": u_force
    }

def render_model_inputs(params):
    """Renders geometry inputs."""
    
    # Number of Spans
    n_spans = st.number_input("Number of Spans", min_value=1, max_value=10, value=2)
    
    # Spans Lengths
    st.markdown("**Span Lengths (m)**")
    cols = st.columns(n_spans)
    spans = []
    for i in range(n_spans):
        val = cols[i].number_input(f"L{i+1}", min_value=0.5, value=4.0, step=0.5, key=f"span_{i}")
        spans.append(val)
        
    # Supports
    st.markdown("**Support Conditions**")
    n_nodes = n_spans + 1
    sup_data = []
    
    # Default: Pin start, Roller others (Stable beam)
    sup_cols = st.columns(n_nodes)
    for i in range(n_nodes):
        default_idx = 0 if i == 0 else 1 # Pin first, then Rollers
        if i == 0:
             stype = sup_cols[i].selectbox(f"Sup {i+1}", ["Pin", "Fixed"], index=0, key=f"sup_{i}")
        else:
             stype = sup_cols[i].selectbox(f"Sup {i+1}", ["Roller", "Pin", "Fixed", "None"], index=1 if i < n_nodes-1 else 1, key=f"sup_{i}")
        
        if stype != "None":
            sup_data.append({"id": i, "type": stype})
            
    sup_df = pd.DataFrame(sup_data)
    
    # Basic Stability Check (Need at least 2 supports or 1 fixed)
    stable = len(sup_data) >= 2 or (len(sup_data)==1 and sup_data[0]['type']=='Fixed')
    
    return n_spans, spans, sup_df, stable

def render_loads(n_spans, spans, params):
    """Renders load inputs."""
    st.markdown("### ⬇️ Loads")
    
    if 'loads' not in st.session_state:
        st.session_state['loads'] = []
        
    # Add Load Form
    with st.expander("➕ Add Load", expanded=True):
        c1, c2, c3, c4 = st.columns([1,1,1,1])
        span_idx = c1.selectbox("Span #", range(1, n_spans+1)) - 1
        l_type = c2.selectbox("Type", ["Uniform (kg/m)", "Point (kg)"])
        mag = c3.number_input("Magnitude", value=1000.0, step=100.0)
        
        loc = 0.0
        if "Point" in l_type:
            loc = c4.number_input(f"Dist x (m) from left of Span {span_idx+1}", 
                                  min_value=0.0, max_value=spans[span_idx], value=spans[span_idx]/2)
            
        if st.button("Add Load"):
            st.session_state['loads'].append({
                "span_idx": span_idx,
                "type": "U" if "Uniform" in l_type else "P",
                "w": mag if "Uniform" in l_type else 0,
                "P": mag if "Point" in l_type else 0,
                "x": loc
            })
            
    # Display Current Loads
    if st.session_state['loads']:
        st.markdown("**Defined Loads:**")
        load_df = pd.DataFrame(st.session_state['loads'])
        
        # Display nicely
        display_data = []
        for i, row in load_df.iterrows():
            desc = f"Span {int(row['span_idx'])+1}: "
            if row['type'] == 'U':
                desc += f"Uniform Load {row['w']} kg/m"
            else:
                desc += f"Point Load {row['P']} kg @ {row['x']} m"
            display_data.append(desc)
            
        st.dataframe(pd.DataFrame(display_data, columns=["Load Description"]), use_container_width=True, hide_index=True)
        
        if st.button("Clear All Loads"):
            st.session_state['loads'] = []
            st.rerun()
            
    return st.session_state['loads']
