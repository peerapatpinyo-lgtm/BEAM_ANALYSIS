import streamlit as st
import pandas as pd
import numpy as np

def render_all_sidebar_inputs():
    """
    Renders the sidebar inputs with CORRECT UNITS.
    Fix: E modulus conversion (MPa -> Pa)
    """
    st.sidebar.markdown("### 1. Material & Section")
    
    # --- 1. Parameters ---
    col1, col2 = st.sidebar.columns(2)
    with col1:
        fc = st.number_input("f'c (MPa)", 15.0, 50.0, 24.0, step=1.0)
        b = st.number_input("Width b (m)", 0.1, 1.0, 0.20, step=0.05)
    with col2:
        fy = st.number_input("fy (MPa)", 240, 500, 400, step=10)
        h = st.number_input("Depth h (m)", 0.2, 2.0, 0.40, step=0.05)
        
    # [FIXED] Correct Unit Conversion for E
    # ACI Formula: E = 4700 * sqrt(fc)  (Result in MPa)
    # We need Pa (N/m^2) for the solver.
    # 1 MPa = 1,000,000 Pa
    E_c = 4700 * np.sqrt(fc) * 1e6  # คูณ 1,000,000 (ไม่ใช่ 1,000)
    
    I_g = (b * h**3) / 12
    params = {'fc': fc, 'fy': fy, 'b': b, 'h': h, 'E': E_c, 'I': I_g}

    # --- 2. Geometry ---
    st.sidebar.markdown("### 2. Geometry")
    n_spans = st.sidebar.number_input("Number of Spans", 1, 10, 1)
    spans = []
    
    st_cols = st.sidebar.columns(min(n_spans, 4))
    for i in range(n_spans):
        with st_cols[i % 4]:
            l = st.number_input(f"L{i+1}", 1.0, 20.0, 4.0, key=f"span_{i}")
            spans.append(l)

    # --- 3. Supports ---
    st.sidebar.markdown("### 3. Supports")
    node_coords = [0] + list(np.cumsum(spans))
    n_nodes = len(node_coords)
    
    # Default: Pin at start, Roller at end (Simple Beam default)
    default_sups = ["Pin"] + ["Roller"] * (n_nodes - 1)
        
    sup_data = []
    for i in range(n_nodes):
        stype = st.sidebar.selectbox(f"Node {i} (@{node_coords[i]:.2f}m)", 
                                     ["None", "Pin", "Roller", "Fixed"], 
                                     index=["None", "Pin", "Roller", "Fixed"].index(default_sups[i]),
                                     key=f"sup_{i}")
        if stype != "None":
            sup_data.append({"id": i, "x": node_coords[i], "type": stype})
            
    sup_df = pd.DataFrame(sup_data)

    # --- 4. Loads ---
    st.sidebar.markdown("### 4. Loads")
    
    if 'load_list' not in st.session_state:
        st.session_state.load_list = []

    with st.sidebar.expander("➕ Add Load", expanded=True):
        l_type = st.selectbox("Type", ["Point Load (P)", "Uniform Load (U)"])
        
        # Span Selection
        span_opts = [f"Span {i+1}" for i in range(n_spans)]
        l_span_str = st.selectbox("On Span", span_opts)
        l_span = span_opts.index(l_span_str)
        
        l_mag = st.number_input("Magnitude (kN or kN/m)", 0.0, 1000.0, 3.6) # Default 3.6 per your query
        
        l_dist = 0.0
        if l_type == "Point Load (P)":
            l_dist = st.slider(f"Dist. from left (m)", 0.0, spans[l_span], spans[l_span]/2)
        else:
            l_dist = spans[l_span] # Full span UDL
            
        if st.button("Add Load"):
            st.session_state.load_list.append({
                "id": len(st.session_state.load_list),
                "type": "P" if "Point" in l_type else "U",
                "span_index": l_span,
                "mag": l_mag * 1000, # Convert kN -> N
                "dist": l_dist,
                "case": "LL"
            })

    # Load Table & Clear
    loads_df = pd.DataFrame(st.session_state.load_list)
    if not loads_df.empty:
        # Show mini table
        display_df = loads_df.copy()
        display_df['mag'] = display_df['mag'] / 1000 # Show in kN
        st.sidebar.dataframe(display_df[['type', 'span_index', 'mag']])
        
        if st.sidebar.button("Clear All Loads"):
            st.session_state.load_list = []
            st.rerun()

    # --- Stability Check ---
    fixed_dof = 0
    for s in sup_data:
        if s['type'] == 'Pin': fixed_dof += 2
        elif s['type'] == 'Roller': fixed_dof += 1
        elif s['type'] == 'Fixed': fixed_dof += 3
        
    stable = True
    if fixed_dof < 3: stable = False
    
    return params, n_spans, spans, sup_df, loads_df, stable
