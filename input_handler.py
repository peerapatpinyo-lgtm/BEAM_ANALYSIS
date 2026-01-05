import streamlit as st
import pandas as pd

def render_sidebar():
    st.sidebar.header("⚙️ Global Settings")
    
    # Material
    with st.sidebar.expander("1. Material Properties", expanded=True):
        E = st.number_input("Elastic Modulus (E) [ksc]", value=2e6, step=1e5)
        fc = st.number_input("Concrete f'c [ksc]", value=240)
        fy = st.number_input("Rebar fy [ksc]", value=4000)
    
    # Section
    with st.sidebar.expander("2. Section Geometry", expanded=True):
        b = st.number_input("Width b [cm]", value=25)
        h = st.number_input("Depth h [cm]", value=50)
        cover = st.number_input("Clear Cover [cm]", value=3.0)
        
        # Calculate I automatically (convert to m4)
        I = (b/100 * (h/100)**3) / 12
        st.write(f"Inertia (I) = {I:.2e} m⁴")

    # Factors
    with st.sidebar.expander("3. Load Factors (SDM)", expanded=False):
        gamma_dead = st.number_input("Dead Load Factor", value=1.4)
        gamma_live = st.number_input("Live Load Factor", value=1.7)
        
    # Rebar
    with st.sidebar.expander("4. Rebar Settings", expanded=False):
        main_bar = st.selectbox("Main Bar", ["DB12", "DB16", "DB20", "DB25", "DB28"], index=1)
    
    return {
        'E': E, 'I': I, 'fc': fc, 'fy': fy, 'b': b, 'h': h, 'cover': cover,
        'gamma_dead': gamma_dead, 'gamma_live': gamma_live, 'main_bar': main_bar
    }

def render_model_inputs(params):
    st.header("1. 🏗️ Model Geometry")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        n_spans = st.number_input("Number of Spans", min_value=1, max_value=10, value=2)
    
    # Dynamic Span Input
    spans = []
    cols = st.columns(n_spans)
    for i in range(n_spans):
        spans.append(cols[i].number_input(f"Span {i+1} (m)", min_value=1.0, value=5.0, key=f"span_{i}"))
        
    # Support Types Table
    st.subheader("Supports Conditions")
    sup_data = []
    
    # Default supports: Pin at start, Roller at others
    default_sups = ["Pin"] + ["Roller"] * n_spans
    
    cols_sup = st.columns(n_spans + 1)
    for i in range(n_spans + 1):
        with cols_sup[i]:
            sType = st.selectbox(
                f"Node {i}", 
                ["Pin", "Roller", "Fixed", "None"], 
                index=["Pin", "Roller", "Fixed", "None"].index(default_sups[i]) if i <= n_spans else 3,
                key=f"sup_{i}"
            )
            if sType != "None":
                sup_data.append({'id': i, 'type': sType})
    
    return n_spans, spans, pd.DataFrame(sup_data), True

def render_loads(n_spans, spans, params, sup_df):
    st.header("2. ⬇️ Applied Loads")
    st.info("💡 Tip: Edit the table below to add loads. 'Start' is distance from left of the chosen span.")

    # Prepare default data structure for Data Editor
    if 'load_df' not in st.session_state:
        st.session_state.load_df = pd.DataFrame([
            {"Type": "Uniform", "Span": 1, "Magnitude": 1000.0, "Start (m)": 0.0, "Case": "DL"},
            {"Type": "Point", "Span": 2, "Magnitude": 2000.0, "Start (m)": spans[1]/2 if len(spans)>1 else 0, "Case": "LL"},
        ])

    # Config for Data Editor
    edited_df = st.data_editor(
        st.session_state.load_df,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "Type": st.column_config.SelectboxColumn(
                "Load Type",
                help="Uniform (kg/m) or Point (kg)",
                width="medium",
                options=["Uniform", "Point"],
                required=True,
            ),
            "Span": st.column_config.NumberColumn(
                "Span Index",
                help="Which span? (1, 2, 3...)",
                min_value=1,
                max_value=n_spans,
                step=1,
                format="%d"
            ),
            "Magnitude": st.column_config.NumberColumn(
                "Magnitude",
                help="kg or kg/m",
                min_value=0.0,
                format="%.2f"
            ),
            "Start (m)": st.column_config.NumberColumn(
                "Position x",
                help="Distance from left support of the span",
                min_value=0.0,
                format="%.2f"
            ),
            "Case": st.column_config.SelectboxColumn(
                "Load Case",
                options=["DL", "LL"],
                required=True
            )
        }
    )
    
    # Process Data for Solver
    clean_loads = []
    if not edited_df.empty:
        for _, row in edited_df.iterrows():
            # Validation
            span_idx = int(row["Span"]) - 1
            if span_idx < len(spans):
                clean_loads.append({
                    'type': 'U' if row["Type"] == "Uniform" else 'P',
                    'span_idx': span_idx,
                    'mag': row["Magnitude"],
                    'x': row["Start (m)"],
                    'case': row["Case"]
                })
                
    return pd.DataFrame(clean_loads)
