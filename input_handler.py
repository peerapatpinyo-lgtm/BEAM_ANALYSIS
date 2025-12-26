import streamlit as st

def render_sidebar():
    st.sidebar.header("⚙️ Design Code & Safety")
    
    design_code = st.sidebar.selectbox("Design Code", ["EIT Standard (WSD)", "ACI 318 (SDM)"])
    method = "SDM" if "ACI" in design_code else "WSD"
    
    st.sidebar.markdown("### 🛡️ Load Factors")
    default_dl = 1.4 if method == "SDM" else 1.0
    default_ll = 1.7 if method == "SDM" else 1.0
    
    fact_dl = st.sidebar.number_input("Dead Load Factor (DL)", value=default_dl, step=0.1)
    fact_ll = st.sidebar.number_input("Live Load Factor (LL)", value=default_ll, step=0.1)

    if method == "WSD":
        st.sidebar.caption("Note: Standard WSD uses factor 1.0")

    unit_sys = st.sidebar.selectbox("Unit System", ["Metric (kg, m)", "SI (kN, m)"])
    return design_code, method, fact_dl, fact_ll, unit_sys

def render_geometry_input():
    st.markdown("### 📐 Beam Geometry")
    n_span = st.number_input("Number of Spans", min_value=1, max_value=5, value=2)
    
    c1, c2 = st.columns([1.5, 1])
    spans = []
    supports = []
    
    with c1:
        st.write(" **Span Lengths (m)**")
        for i in range(n_span):
            l = st.number_input(f"Span {i+1} Length (m)", min_value=1.0, value=4.0, key=f"L{i}")
            spans.append(l)
            
    with c2:
        st.write("**Support Types** (Left -> Right)")
        for i in range(n_span + 1):
            def_idx = 0 if i == 0 else 1 
            s = st.selectbox(f"Support {i+1}", ["Pin", "Roller", "Fixed", "None"], index=def_idx, key=f"sup{i}")
            supports.append(s)
    
    return n_span, spans, supports

def render_loads_input(n_span, spans, f_dl, f_ll, unit_sys):
    st.markdown("### 🧱 Loads Input")
    u_load = "kN/m" if "kN" in unit_sys else "kg/m"
    
    loads = []
    for i in range(n_span):
        with st.expander(f"Loads on Span {i+1}", expanded=True):
            col_dl, col_ll = st.columns(2)
            wd = col_dl.number_input(f"DL ({u_load}) - Span {i+1}", value=1000.0)
            wl = col_ll.number_input(f"LL ({u_load}) - Span {i+1}", value=500.0)
            
            w_total = (wd * f_dl) + (wl * f_ll)
            loads.append({'type': 'uniform', 'span_idx': i, 'w': w_total})
            
    return loads

def render_design_input(unit_sys):
    st.markdown("### 🏗️ Design Parameters")
    
    c1, c2 = st.columns(2)
    u_str = "MPa" if "kN" in unit_sys else "ksc"
    fc = c1.number_input(f"Concrete f'c ({u_str})", value=240)
    fy = c2.number_input(f"Steel fy ({u_str})", value=4000)
    
    st.markdown("---")
    c3, c4, c5 = st.columns(3)
    b_mm = c3.number_input("Width b (mm)", value=250, step=50)
    h_mm = c4.number_input("Depth h (mm)", value=500, step=50)
    cov_mm = c5.number_input("Covering (mm)", value=25, step=5)
    
    st.markdown("---")
    c6, c7, c8 = st.columns(3)
    main_bar = c6.selectbox("Main Bar Size", ["DB12", "DB16", "DB20", "DB25", "DB28"], index=1)
    stir_bar = c7.selectbox("Stirrup Size", ["RB6", "RB9", "DB10", "DB12"], index=0)
    
    # Input mm -> Convert to cm for calculation
    manual_s_mm = c8.number_input("Manual Stirrup Spacing (mm) [0=Auto]", value=0, help="ใส่ 0 เพื่อให้คำนวณอัตโนมัติ")
    manual_s_cm = manual_s_mm / 10.0

    b_cm = b_mm / 10.0
    h_cm = h_mm / 10.0
    cov_cm = cov_mm / 10.0
    
    return fc, fy, b_cm, h_cm, cov_cm, main_bar, stir_bar, manual_s_cm
