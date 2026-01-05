import streamlit as st
import pandas as pd

def render_sidebar():
    st.sidebar.header("⚙️ Global Settings")
    with st.sidebar.expander("1. Material & Section", expanded=True):
        E = st.number_input("Elastic Modulus (E) [ksc]", 2e6, step=1e5)
        fc = st.number_input("Concrete f'c [ksc]", 240)
        fy = st.number_input("Rebar fy [ksc]", 4000)
        b = st.number_input("Width b [cm]", 25)
        h = st.number_input("Depth h [cm]", 50)
        cover = st.number_input("Cover [cm]", 3.0)
        
        I = (b/100 * (h/100)**3) / 12
    
    with st.sidebar.expander("2. Load Factors", expanded=False):
        gamma_dead = st.number_input("Dead Load Factor", 1.4)
        gamma_live = st.number_input("Live Load Factor", 1.7)
        
    return {'E': E, 'I': I, 'fc': fc, 'fy': fy, 'b': b, 'h': h, 'cover': cover, 
            'gamma_dead': gamma_dead, 'gamma_live': gamma_live, 'main_bar': "DB16"}

def render_model_inputs(params):
    st.header("1. 🏗️ Model Geometry")
    col1, col2 = st.columns([1, 3])
    n_spans = col1.number_input("Number of Spans", 1, 10, 2)
    
    spans = []
    cols = st.columns(n_spans)
    for i in range(n_spans):
        spans.append(cols[i].number_input(f"Span {i+1} (m)", 1.0, 20.0, 5.0, key=f"s_{i}"))
        
    st.subheader("Supports Conditions")
    sup_data = []
    # Auto-generate standard setup (Pin - Roller - Roller...)
    cols_sup = st.columns(n_spans + 1)
    defaults = ["Pin"] + ["Roller"] * n_spans
    
    for i in range(n_spans + 1):
        with cols_sup[i]:
            sType = st.selectbox(f"Node {i}", ["Pin", "Roller", "Fixed", "None"], 
                               index=["Pin", "Roller", "Fixed", "None"].index(defaults[i]), key=f"sup_{i}")
            if sType != "None": sup_data.append({'id': i, 'type': sType})
            
    return n_spans, spans, pd.DataFrame(sup_data), True

def render_loads(n_spans, spans, params):
    st.header("2. ⬇️ Applied Loads")
    
    # แยก Tab เพื่อความชัดเจน
    tab1, tab2 = st.tabs(["📍 Point Loads", "🌊 Distributed Loads"])
    
    # --- TAB 1: POINT LOADS ---
    with tab1:
        st.caption("Point loads concentrated at a specific location.")
        if 'df_point' not in st.session_state:
            st.session_state.df_point = pd.DataFrame([
                {"Span": 2, "Position (m)": spans[1]/2 if len(spans)>1 else 0, "Magnitude (kg)": 2000.0, "Case": "LL"}
            ])
            
        edited_point = st.data_editor(
            st.session_state.df_point, num_rows="dynamic", use_container_width=True,
            column_config={
                "Span": st.column_config.NumberColumn("Span No.", min_value=1, max_value=n_spans, step=1, format="%d"),
                "Position (m)": st.column_config.NumberColumn("Dist from Left (m)", min_value=0.0, format="%.2f"),
                "Magnitude (kg)": st.column_config.NumberColumn("Load (kg)", min_value=0.0, format="%.2f"),
                "Case": st.column_config.SelectboxColumn("Case", options=["DL", "LL"], required=True)
            },
            key="editor_point"
        )

    # --- TAB 2: DISTRIBUTED LOADS ---
    with tab2:
        st.caption("Uniform loads distributed over a length (e.g., Wall load).")
        if 'df_dist' not in st.session_state:
            st.session_state.df_dist = pd.DataFrame([
                {"Span": 1, "Start (m)": 0.0, "End (m)": spans[0], "Magnitude (kg/m)": 1000.0, "Case": "DL"}
            ])
            
        edited_dist = st.data_editor(
            st.session_state.df_dist, num_rows="dynamic", use_container_width=True,
            column_config={
                "Span": st.column_config.NumberColumn("Span No.", min_value=1, max_value=n_spans, step=1, format="%d"),
                "Start (m)": st.column_config.NumberColumn("Start x (m)", min_value=0.0, format="%.2f"),
                "End (m)": st.column_config.NumberColumn("End x (m)", min_value=0.0, format="%.2f"),
                "Magnitude (kg/m)": st.column_config.NumberColumn("Load (kg/m)", min_value=0.0, format="%.2f"),
                "Case": st.column_config.SelectboxColumn("Case", options=["DL", "LL"], required=True)
            },
            key="editor_dist"
        )
    
    # Merge Data for Solver
    final_loads = []
    
    # Process Point Loads
    for _, row in edited_point.iterrows():
        s_idx = int(row["Span"]) - 1
        if 0 <= s_idx < len(spans):
            final_loads.append({
                'type': 'P', 'span_idx': s_idx, 
                'mag': row["Magnitude (kg)"], 
                'x': row["Position (m)"], 
                'case': row["Case"]
            })
            
    # Process Distributed Loads
    for _, row in edited_dist.iterrows():
        s_idx = int(row["Span"]) - 1
        if 0 <= s_idx < len(spans):
            # Auto-fix if End < Start
            start, end = sorted([row["Start (m)"], row["End (m)"]])
            # If end is 0 or unassigned, assume full span? No, let user define.
            if end == 0: end = spans[s_idx] 
            
            final_loads.append({
                'type': 'U', 'span_idx': s_idx, 
                'mag': row["Magnitude (kg/m)"], 
                'x': start, # For solver start
                'end': end, # For viz and calc
                'case': row["Case"]
            })
            
    return pd.DataFrame(final_loads)
