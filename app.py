import streamlit as st
import utils
import solver
import input_handler 

# --- IMPORT MODULES ใหม่ ---
import tab_analysis
import tab_design
import tab_report

st.set_page_config(page_title="RC Beam Pro", layout="wide")

# --- SIDEBAR & INPUTS ---
with st.sidebar:
    st.header("Project Info")
    project_name = st.text_input("Project Name", "Project A")
    engineer_name = st.text_input("Engineer", "Eng. Somchai")
    
    # เรียก input handler เดิม
    params, n_spans, spans, sup_df, loads_df, stable = input_handler.render_all_sidebar_inputs()

    # Load Factors
    mode_select = st.radio("Mode", ["Check Deflection (Service)", "Design (Ultimate)"], index=1)
    if "Service" in mode_select:
        f_dl, f_ll = 1.0, 1.0
        is_service = True
        tag = "Service"
    else:
        f_dl, f_ll = 1.4, 1.7
        is_service = False
        tag = "Ultimate"

if stable:
    # --- SOLVER PROCESS ---
    with st.spinner('Calculating...'):
        # Run Ultimate
        loads_ult = utils.prepare_load_dataframe(loads_df, n_spans, spans, params, f_dl, f_ll)
        x_ult, M_ult, V_ult, D_ult, R_ult = solver.solve_beam(spans, sup_df, loads_ult, params)
        
        # Run Service
        loads_svc = utils.prepare_load_dataframe(loads_df, n_spans, spans, params, 1.0, 1.0)
        x_svc, M_svc, V_svc, D_svc, R_svc = solver.solve_beam(spans, sup_df, loads_svc, params)

    # เลือกชุดข้อมูลที่จะนำไปพล็อต Analysis
    if is_service:
        x_plot, M_plot, V_plot, D_plot, R_plot = x_svc, M_svc, V_svc, D_svc, R_svc
        display_loads = loads_svc
    else:
        x_plot, M_plot, V_plot, D_plot, R_plot = x_ult, M_ult, V_ult, D_ult, R_ult
        display_loads = loads_ult

    # --- TABS MANAGEMENT ---
    t1, t2, t3 = st.tabs(["1. Analysis", "2. Concrete Design", "3. Report"])

    with t1:
        tab_analysis.render(x_plot, M_plot, V_plot, D_plot, R_plot, spans, sup_df, display_loads, tag)

    with t2:
        # ส่งค่า sup_df เพิ่มเข้าไปตามที่คุณต้องการ
        design_results = tab_design.render(
            n_spans, spans, params, 
            x_ult, M_ult, V_ult, 
            x_svc, M_svc, D_svc, 
            is_service,
            sup_df  # <--- ส่งตัวนี้เพิ่มเข้าไป
        )
       
    with t3:
        tab_report.render(design_results, project_name, engineer_name)

else:
    st.error("Structure Unstable")
