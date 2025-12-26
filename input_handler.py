import streamlit as st

def render_sidebar():
    st.sidebar.header("⚙️ Design Code & Safety")
    
    # 1. Code Selection
    design_code = st.sidebar.selectbox("Design Code", ["EIT Standard (WSD)", "ACI 318 (SDM)"])
    method = "SDM" if "ACI" in design_code else "WSD"
    
    # 2. Safety Factors
    st.sidebar.markdown("### 🛡️ Load Factors")
    if method == "SDM":
        fact_dl = st.sidebar.number_input("Dead Load Factor (DL)", value=1.4, step=0.1)
        fact_ll = st.sidebar.number_input("Live Load Factor (LL)", value=1.7, step=0.1)
    else:
        fact_dl = 1.0
        fact_ll = 1.0
        st.sidebar.info("WSD uses Service Load (Factor = 1.0)")

    # 3. Unit System (Display Only, logic handled internally)
    unit_sys = st.sidebar.selectbox("Unit System", ["Metric (kg, m)", "SI (kN, m)"])
    
    return design_code, method, fact_dl, fact_ll, unit_sys

def render_geometry_input():
    st.markdown("### 📐 Beam Geometry")
    n_span = st.number_input("Number of Spans", min_value=1, max_value=5, value=2)
    
    c1, c2 = st.columns(2)
    spans = []
    supports = ["Pin"] # Start with Pin
    
    with c1:
        st.write(" **Span Lengths (m)**")
        for i in range(n_span):
            l = st.number_input(f"Span {i+1} Length (m)", min_value=1.0, value=4.0, key=f"L{i}")
            spans.append(l)
            supports.append("Roller") # Default intermediate
            
    with c2:
        st.write("**Support Types**")
        # Simple Logic: Left is Pin, others Roller/Fixed (Can be expanded)
        st.caption("Default: Pin-Roller system. (Advanced support customization coming soon)")
    
    return n_span, spans, supports

def render_loads_input(n_span, spans, f_dl, f_ll, unit_sys):
    st.markdown("### 🧱 Loads Input")
    u_load = "kN/m" if "kN" in unit_sys else "kg/m"
    
    loads = []
    # Simplified: Uniform Load per span
    for i in range(n_span):
        with st.expander(f"Loads on Span {i+1}", expanded=True):
            col_dl, col_ll = st.columns(2)
            wd = col_dl.number_input(f"DL ({u_load}) - Span {i+1}", value=1000.0)
            wl = col_ll.number_input(f"LL ({u_load}) - Span {i+1}", value=500.0)
            
            # Combine
            w_total = (wd * f_dl) + (wl * f_ll)
            loads.append({'type': 'uniform', 'span_idx': i, 'w': w_total})
            
    return loads

def render_design_input(unit_sys):
    st.markdown("### 🏗️ Design Parameters")
    
    # Material
    c1, c2 = st.columns(2)
    u_str = "MPa" if "kN" in unit_sys else "ksc"
    fc = c1.number_input(f"Concrete f'c ({u_str})", value=240)
    fy = c2.number_input(f"Steel fy ({u_str})", value=4000) # SD40
    
    st.markdown("---")
    # Section Properties (UPDATED TO mm)
    c3, c4, c5 = st.columns(3)
    b_mm = c3.number_input("Width b (mm)", value=250, step=50)
    h_mm = c4.number_input("Depth h (mm)", value=500, step=50)
    cov_mm = c5.number_input("Covering (mm)", value=25, step=5)
    
    # Rebar Selection
    st.markdown("---")
    c6, c7, c8 = st.columns(3)
    main_bar = c6.selectbox("Main Bar Size", ["DB12", "DB16", "DB20", "DB25", "DB28"], index=1)
    stir_bar = c7.selectbox("Stirrup Size", ["RB6", "RB9", "DB10", "DB12"], index=0)
    manual_s = c8.number_input("Manual Stirrup Spacing (cm) [0=Auto]", value=0, help="ใส่ 0 เพื่อให้โปรแกรมคำนวณอัตโนมัติ")

    # Convert mm inputs to cm for calculation logic
    b_cm = b_mm / 10.0
    h_cm = h_mm / 10.0
    cov_cm = cov_mm / 10.0
    
    return fc, fy, b_cm, h_cm, cov_cm, main_bar, stir_bar, manual_s
