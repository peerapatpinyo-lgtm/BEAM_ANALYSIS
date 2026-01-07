import streamlit as st
import pandas as pd
import numpy as np

def render_all_sidebar_inputs():
    """
    Renders the sidebar for all inputs:
    1. Material & Section Properties
    2. Geometry (Spans)
    3. Supports
    4. Loads
    Returns: (params, n_spans, spans, sup_df, loads_df, stable)
    """
    st.sidebar.markdown("### 1. Material & Section")
    
    # --- 1. Parameters ---
    col1, col2 = st.sidebar.columns(2)
    with col1:
        fc = st.number_input("f'c (MPa)", 15.0, 50.0, 25.0, step=1.0)
        b = st.number_input("Width b (m)", 0.1, 1.0, 0.3, step=0.05)
    with col2:
        fy = st.number_input("fy (MPa)", 240, 500, 400, step=10)
        h = st.number_input("Depth h (m)", 0.2, 2.0, 0.5, step=0.05)
        
    E_c = 4700 * np.sqrt(fc) * 1000 # kPa -> kN/m2
    I_g = (b * h**3) / 12
    params = {'fc': fc, 'fy': fy, 'b': b, 'h': h, 'E': E_c, 'I': I_g}

    # --- 2. Geometry ---
    st.sidebar.markdown("### 2. Geometry")
    n_spans = st.sidebar.number_input("Number of Spans", 1, 10, 2)
    spans = []
    
    # Dynamic inputs for spans
    st_cols = st.sidebar.columns(min(n_spans, 4))
    for i in range(n_spans):
        with st_cols[i % 4]:
            l = st.number_input(f"L{i+1}", 1.0, 20.0, 5.0, key=f"span_{i}")
            spans.append(l)

    # --- 3. Supports ---
    st.sidebar.markdown("### 3. Supports")
    # Generate default nodes (0 to n_spans)
    node_coords = [0] + list(np.cumsum(spans))
    n_nodes = len(node_coords)
    
    # Default: Pin at start, Roller at others
    default_sups = []
    for i in range(n_nodes):
        if i == 0: default_sups.append("Pin")
        else: default_sups.append("Roller")
        
    sup_data = []
    for i in range(n_nodes):
        stype = st.sidebar.selectbox(f"Node {i} (@{node_coords[i]:.2f}m)", 
                                     ["None", "Pin", "Roller", "Fixed"], 
                                     index=["None", "Pin", "Roller", "Fixed"].index(default_sups[i]) if default_sups[i] in ["None", "Pin", "Roller", "Fixed"] else 0,
                                     key=f"sup_{i}")
        if stype != "None":
            sup_data.append({"id": i, "x": node_coords[i], "type": stype})
            
    sup_df = pd.DataFrame(sup_data)

    # --- 4. Loads ---
    st.sidebar.markdown("### 4. Loads")
    
    if 'load_list' not in st.session_state:
        st.session_state.load_list = []

    with st.sidebar.expander("➕ Add Load", expanded=False):
        l_type = st.selectbox("Type", ["Point Load (P)", "Uniform Load (U)"])
        l_span = st.selectbox("On Span", range(1, n_spans+1)) - 1 # 0-index
        l_mag = st.number_input("Magnitude (kN or kN/m)", 0.0, 1000.0, 10.0)
        
        l_dist = 0.0
        if l_type == "Point Load (P)":
            l_dist = st.slider(f"Distance from left of Span {l_span+1} (m)", 0.0, spans[l_span], spans[l_span]/2)
        else:
            l_dist = spans[l_span] # UDL covers full span in this simple version
            st.caption("Uniform load applied to full span length.")

        add_btn = st.button("Add Load")
        if add_btn:
            st.session_state.load_list.append({
                "id": len(st.session_state.load_list),
                "type": "P" if "Point" in l_type else "U",
                "span_index": l_span,
                "mag": l_mag * 1000, # Convert to N
                "dist": l_dist, # Location for P, Length for U
                "case": "LL" # Default to Live Load for added loads
            })

    # Show Loads
    loads_df = pd.DataFrame(st.session_state.load_list)
    if not loads_df.empty:
        st.sidebar.dataframe(loads_df[['type', 'span_index', 'mag', 'dist']])
        if st.sidebar.button("Clear All Loads"):
            st.session_state.load_list = []
            st.rerun()

    # --- 5. Stability Check (Basic) ---
    fixed_dof = 0
    for s in sup_data:
        if s['type'] == 'Pin': fixed_dof += 2
        elif s['type'] == 'Roller': fixed_dof += 1
        elif s['type'] == 'Fixed': fixed_dof += 3
        
    # Standard unstable check (very basic)
    stable = True
    if fixed_dof < 3: stable = False
    if len(sup_data) < 2 and fixed_dof < 3: stable = False # Cantilever needs Fixed

    return params, n_spans, spans, sup_df, loads_df, stable
