import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- 1. IMPORT CUSTOM MODULES ---
import input_handler
import solver
import rc_design
import design_view
import section_plotter

# --- 2. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Beam Analysis & Design Pro", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Style เพื่อความสวยงามและอ่านง่าย
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 20px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); border: 1px solid #eee; }
    div.stTabs [data-baseweb="tab-list"] { gap: 24px; }
    div.stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: #f0f2f6; border-radius: 4px 4px 0 0; gap: 1px; padding-top: 10px; }
    div.stTabs [aria-selected="true"] { background-color: #ffffff; border-bottom: 2px solid #ff4b4b; }
    </style>
    """, unsafe_allow_html=True)

st.title("🏗️ RC Beam Analysis & Design Pro")
st.caption("Professional Structural Engineering Tool for Continuous Beams (ACI 318-19 / EIT 2024)")

# --- 3. SIDEBAR INPUTS & GLOBAL SETTINGS ---
# ดึงค่าพารามิเตอร์เบื้องต้น
params, n_spans, spans, sup_df, loads_df, stable = input_handler.render_all_sidebar_inputs()

# --- 4. ANALYSIS SETTINGS (FIXED VARIABLE SCOPE) ---
st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Analysis Control")

# ประกาศตัวแปรเป็น Global/Outside เพื่อป้องกัน NameError
is_service = False 
tag = "Ultimate"
f_dl, f_ll = 1.4, 1.7 # Default values

mode_select = st.sidebar.radio(
    "Select Analysis Mode:",
    ["Service Load (1.0DL + 1.0LL)", "Ultimate Load (Factored)"],
    index=1
)

if mode_select.startswith("Service"):
    f_dl, f_ll = 1.0, 1.0
    is_service = True
    tag = "Service"
else:
    col_f1, col_f2 = st.sidebar.columns(2)
    f_dl = col_f1.number_input("DL Factor", value=1.4, step=0.1, key="dl_f_global")
    f_ll = col_f2.number_input("LL Factor", value=1.7, step=0.1, key="ll_f_global")
    is_service = False
    tag = "Ultimate"

# --- 5. MAIN PROCESSING BLOCK ---
if not stable:
    st.error("🚨 **Structure Unstable:** Please check supports. A stable beam requires at least 3 reaction components.")
else:
    try:
        # 5.1 Self-Weight Logic (Fix 0.00 Issue)
        # คำนวณเป็น kN/m เสมอ (b*h*24)
        b_m = params.get('b', 0.20)
        h_m = params.get('h', 0.40)
        w_sw_base = b_m * h_m * 24.0  
        w_sw_factored = w_sw_base * f_dl
        
        # Lists สำหรับแยกข้อมูลคำนวณ (N) และข้อมูลแสดงผล (kN)
        solver_payload = [] # สำหรับ solver.solve_beam
        display_payload = [] # สำหรับ design_view.plot_analysis_results
        
        # 5.2 Add Self-Weight to Lists
        for i in range(n_spans):
            # ไปยัง Solver (หน่วย N/m)
            solver_payload.append({
                'span_index': i, 'type': 'U', 
                'mag': w_sw_factored * 1000.0, 
                'dist': spans[i]
            })
            # ไปยัง Display (หน่วย kN/m)
            display_payload.append({
                'span_index': i, 'type': 'U', 
                'mag': w_sw_factored, 
                'dist': spans[i], 'desc': 'Self-Weight'
            })
            
        # 5.3 Process User Loads (Fix 3600 kN Issue)
        if not loads_df.empty:
            for _, row in loads_df.iterrows():
                try:
                    s_idx = int(row['span_index'])
                    if s_idx >= n_spans: continue
                    
                    l_type = row['type']
                    val_kN = float(row['mag']) # เช่น 3.6 kN
                    factored_val_kN = val_kN * f_ll
                    dist_m = float(row['dist'])
                    
                    # ไปยัง Solver (หน่วย Newton)
                    solver_payload.append({
                        'span_index': s_idx, 'type': l_type, 
                        'mag': factored_val_kN * 1000.0, 
                        'dist': dist_m
                    })
                    # ไปยัง Display (หน่วย kN)
                    display_payload.append({
                        'span_index': s_idx, 'type': l_type, 
                        'mag': factored_val_kN, 
                        'dist': dist_m, 'desc': 'User Load'
                    })
                except Exception: continue

        calc_loads_df = pd.DataFrame(solver_payload)
        plot_loads_df = pd.DataFrame(display_payload)

        # --- 6. CORE SOLVER EXECUTION ---
        # Solver ทำงานในหน่วย N, m
        x_raw, M_raw, V_raw, D_raw, R_raw = solver.solve_beam(spans, sup_df, calc_loads_df, params)
        
        # แปลงผลลัพธ์จาก SI (N) เป็น Engineer Units (kN) ทันทีหลังจบ Solver
        res_df = pd.DataFrame({
            'x': x_raw,
            'moment': M_raw / 1000.0,    # kNm
            'shear': V_raw / 1000.0,     # kN
            'deflection': D_raw * 1000.0 # mm
        })
        
        # แปลง Reactions เป็น kN
        R_kN = {k: v / 1000.0 for k, v in R_raw.items()}

        # --- 7. PRESENTATION LAYER (TABS) ---
        tab_analysis, tab_design, tab_report = st.tabs([
            "📊 Analysis Diagrams", 
            "📝 RC Design & Detailing", 
            "📋 Load & Equilibrium Report"
        ])

        # ================= TAB 1: ANALYSIS =================
        with tab_analysis:
            st.subheader(f"Internal Forces & Displacement ({tag})")
            # ใช้ plot_loads_df ที่เป็นหน่วย kN เพื่อความถูกต้องในกราฟ
            st.plotly_chart(design_view.plot_analysis_results(res_df, spans, sup_df, plot_loads_df, R_kN), use_container_width=True)
            
            # Metrics Row
            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            v_max = res_df['shear'].abs().max()
            m_pos_max = res_df['moment'].max()
            m_neg_max = res_df['moment'].min()
            def_max = res_df['deflection'].abs().max()
            
            m_col1.metric("Max Shear (Vu)", f"{v_max:.2f} kN")
            m_col2.metric("Max Moment (+)", f"{m_pos_max:.2f} kNm")
            m_col3.metric("Max Moment (-)", f"{abs(m_neg_max):.2f} kNm")
            m_col4.metric("Max Deflection", f"{def_max:.2f} mm")

            st.markdown("### 📍 Support Reactions")
            reac_data = [{"Node": int(str(k).replace('R','')), "Reaction (kN)": f"{v:.2f}"} for k, v in R_kN.items()]
            st.table(pd.DataFrame(reac_data).sort_values("Node"))

        # ================= TAB 2: RC DESIGN =================
        with tab_design:
            st.header("Reinforced Concrete Design Detail")
            if is_service:
                st.warning("⚠️ **Note:** Standard RC design requires factored (Ultimate) loads. Current factors are 1.0.")

            span_designs = []
            x_offset = 0.0
            
            for i, L in enumerate(spans):
                # กรองข้อมูลช่วง Span
                mask = (res_df['x'] >= x_offset - 1e-6) & (res_df['x'] <= x_offset + L + 1e-6)
                data = res_df[mask]
                
                if not data.empty:
                    mu_p = data['moment'].max()
                    mu_n = abs(data['moment'].min())
                    vu = data['shear'].abs().max()
                    d_eff = params['h'] - 0.05 # 5cm cover
                    
                    # เรียก Module ออกแบบ
                    As_p, _, _, stp_p = rc_design.design_beam_flexure(mu_p, params['b'], d_eff, params['fc'], params['fy'])
                    As_n, _, _, stp_n = rc_design.design_beam_flexure(mu_n, params['b'], d_eff, params['fc'], params['fy'])
                    s_v, _, stp_v = rc_design.check_shear(vu, params['b'], d_eff, params['fc'], params['fy'])
                    
                    # คำนวณจำนวนเหล็ก DB16 (Area = 201 mm2)
                    n_pos = max(2, int(np.ceil((As_p * 1e6) / 201.0)))
                    n_neg = max(2, int(np.ceil((As_n * 1e6) / 201.0)))
                    
                    span_designs.append({
                        'span': i+1, 'pos': {'n': n_pos}, 'neg': {'n': n_neg}, 'shear': {'s': s_v}
                    })
                    
                    with st.expander(f"📖 Span {i+1} Calculation Steps"):
                        c_p, c_n = st.columns(2)
                        with c_p:
                            st.write("**Bottom Steel**")
                            for s in stp_p: st.latex(s)
                        with c_n:
                            st.write("**Top Steel**")
                            for s in stp_n: st.latex(s)
                        st.write("**Stirrup Spacing**")
                        for s in stp_v: st.latex(s)
                
                x_offset += L

            st.markdown("---")
            st.subheader("🛠️ Detailing Preview")
            if span_designs:
                dcol1, dcol2 = st.columns([1, 2])
                with dcol1:
                    # Section View
                    st.pyplot(section_plotter.plot_section(
                        params['b'], params['h'], 40, 16, 
                        span_designs[0]['neg']['n'], span_designs[0]['pos']['n'], 
                        f"RB6@{span_designs[0]['shear']['s']*1000:.0f} mm",
                        params['fc'], params['fy']
                    ))
                with dcol2:
                    # Longitudinal View
                    st.pyplot(section_plotter.plot_longitudinal_section_detailed(spans, sup_df, span_designs, params['h'], 40))

        # ================= TAB 3: REPORT =================
        with tab_report:
            st.header("🧮 Detailed Load & Equilibrium Report")
            
            # 1. Self-Weight Check
            st.subheader("1. Self-Weight Calculation")
            sw_rep = {
                "Property": ["Width (b)", "Height (h)", "Concrete Density", "Load Factor", "Factored Load"],
                "Value": [f"{b_m} m", f"{h_m} m", "24 kN/m³", f"{f_dl}", f"{w_sw_factored:.2f} kN/m"]
            }
            st.table(pd.DataFrame(sw_rep))

            # 2. Live Loads Check
            st.subheader("2. Applied User Loads")
            if not loads_df.empty:
                st.dataframe(loads_df.assign(Factored_kN=lambda x: x['mag']*f_ll), use_container_width=True)
            else:
                st.write("No additional loads applied.")

            # 3. Static Equilibrium Verification
            st.subheader("3. Verification of Equilibrium (Sum Fy = 0)")
            sum_reac = sum(R_kN.values())
            sum_load = w_sw_factored * sum(spans)
            if not loads_df.empty:
                for _, r in loads_df.iterrows():
                    sum_load += (r['mag'] * f_ll) * (1.0 if r['type'] == 'P' else r['dist'])
            
            eq_col1, eq_col2 = st.columns(2)
            eq_col1.metric("Total Reactions", f"{sum_reac:.2f} kN")
            eq_col2.metric("Total Applied Loads", f"{sum_load:.2f} kN")
            
            if abs(sum_reac - sum_load) < 0.1:
                st.success("✅ Equilibrium Verified: The structure is in static balance.")
            else:
                st.error(f"❌ Equilibrium Error: Difference of {abs(sum_reac - sum_load):.4f} kN detected.")

    except Exception as e:
        st.error(f"❌ **Application Error:** {str(e)}")
        st.exception(e)

# Footer
st.markdown("---")
st.caption("Developed by Gemini Structural Engine | Professional RC Design Suite v2026")
