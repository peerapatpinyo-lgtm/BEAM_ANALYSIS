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

# Custom Style เพื่อให้ UI ดูเป็นโปรแกรมวิศวกรรมมืออาชีพ
st.markdown("""
    <style>
    .reportview-container { background: #f0f2f6; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

st.title("🏗️ RC Beam Analysis & Design Pro")
st.caption("Continuous Beam Analysis (Timoshenko) & Reinforced Concrete Detailing")

# --- 3. SIDEBAR INPUTS ---
params, n_spans, spans, sup_df, loads_df, stable = input_handler.render_all_sidebar_inputs()

if not stable:
    st.error("🚨 ระบบโครงสร้างไม่เสถียร (Unstable Structure): โปรดตรวจสอบจุดรองรับให้มีแรงปฏิกิริยาเพียงพอ")
else:
    # --- 4. ANALYSIS SETTINGS & LOAD FACTORS ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Analysis Settings")
    mode_select = st.sidebar.radio(
        "Select Analysis Mode:",
        ["Service Load (1.0DL + 1.0LL)", "Ultimate Load (Factored)"],
        index=1
    )
    
    # กำหนดค่า Factor ให้ชัดเจนเพื่อใช้ใน Report
    if mode_select.startswith("Service"):
        f_dl, f_ll = 1.0, 1.0
        tag = "Service"
    else:
        col_f1, col_f2 = st.sidebar.columns(2)
        f_dl = col_f1.number_input("DL Factor", value=1.4, step=0.1, key="dl_f")
        f_ll = col_f2.number_input("LL Factor", value=1.7, step=0.1, key="ll_f")
        tag = "Ultimate"

    # --- 5. UNIT-CONSISTENT LOAD CALCULATION ---
    try:
        # 5.1 Self-Weight (kN/m)
        # สูตร: b(m) * h(m) * 24 kN/m3
        b_m, h_m = params['b'], params['h']
        w_sw_base = b_m * h_m * 24.0  
        w_sw_factored = w_sw_base * f_dl
        
        # ป้องกันปัญหา Self-weight เป็น 0.00
        if w_sw_base <= 0:
            st.warning("⚠️ หน้าตัดคานมีขนาดเป็น 0 หรือค่าผิดปกติ โปรดตรวจสอบหน้าตัด")

        # แยกชุดข้อมูล: Solver (Newton) vs Display (kN)
        solver_input = [] 
        display_input = [] 
        
        # 5.2 ลงข้อมูล Self-Weight ทุก Span
        for i in range(n_spans):
            # เข้า Solver (N/m)
            solver_input.append({
                'span_index': i, 'type': 'U', 
                'mag': w_sw_factored * 1000.0, 
                'dist': spans[i]
            })
            # เข้า Plot/Table (kN/m) - แก้ปัญหาโชว์ 0.00
            display_input.append({
                'span_index': i, 'type': 'U', 
                'mag': w_sw_factored, 
                'dist': spans[i], 'desc': 'Self-Weight'
            })
            
        # 5.3 ลงข้อมูล User Loads (เช่น 3.6 kN)
        if not loads_df.empty:
            for _, row in loads_df.iterrows():
                try:
                    s_idx = int(row['span_index'])
                    if s_idx >= n_spans: continue
                    
                    l_type = row['type']
                    mag_raw = float(row['mag']) # ค่าดิบ 3.6
                    factored_kN = mag_raw * f_ll
                    dist = float(row['dist'])
                    
                    # เข้า Solver (Newton)
                    solver_input.append({
                        'span_index': s_idx, 'type': l_type, 
                        'mag': factored_kN * 1000.0, 
                        'dist': dist
                    })
                    # เข้า Plot (kN) - แก้ปัญหาโชว์ 3600
                    display_input.append({
                        'span_index': s_idx, 'type': l_type, 
                        'mag': factored_kN, 
                        'dist': dist, 'desc': 'User Load'
                    })
                except: continue

        # สร้าง DataFrame สำหรับส่งต่อ
        calc_loads_df = pd.DataFrame(solver_input)
        plot_loads_df = pd.DataFrame(display_input)

        # --- 6. CORE SOLVER ---
        # แก้ไข: รับค่า N, m ออกมาเป็น SI Unit
        x_eval, M_n, V_n, D_m, R_n = solver.solve_beam(spans, sup_df, calc_loads_df, params)
        
        # แปลงผลลัพธ์จาก Newton เป็น kN ทันทีเพื่อความปลอดภัย
        res_df = pd.DataFrame({
            'x': x_eval,
            'moment': M_n / 1000.0,    # kNm
            'shear': V_n / 1000.0,     # kN
            'deflection': D_m * 1000.0 # mm
        })
        
        # Reactions (kN)
        R_kN = {k: v / 1000.0 for k, v in R_n.items()}

        # --- 7. PRESENTATION (TABS) ---
        tab1, tab2, tab3 = st.tabs(["📊 Analysis Results", "📝 RC Design Detail", "📋 Loads & Equilibrium"])

        with tab1:
            st.subheader(f"Internal Force Diagrams ({tag})")
            # ใช้ plot_loads_df ที่เป็นหน่วย kN เท่านั้น
            st.plotly_chart(design_view.plot_analysis_results(res_df, spans, sup_df, plot_loads_df, R_kN), use_container_width=True)
            
            # Metrics Summary
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Max Shear (V_u)", f"{res_df['shear'].abs().max():.2f} kN")
            m2.metric("Max Moment (+)", f"{res_df['moment'].max():.2f} kNm")
            m3.metric("Max Moment (-)", f"{abs(res_df['moment'].min()):.2f} kNm")
            m4.metric("Max Deflection", f"{res_df['deflection'].abs().max():.2f} mm")

            st.markdown("### 📍 Support Reactions (kN)")
            reac_df = pd.DataFrame([{"Node": int(str(k).replace('R','')), "Reaction (kN)": f"{v:.2f}"} for k, v in R_kN.items()])
            st.table(reac_df.sort_values("Node"))

        with tab2:
            st.header("Reinforced Concrete Design")
            if is_service:
                st.warning("⚠️ การออกแบบเหล็กเสริมควรใช้โหมด Ultimate Load")

            design_results = []
            cur_x = 0.0
            for i, L in enumerate(spans):
                # กรองข้อมูลใน Span
                span_mask = (res_df['x'] >= cur_x - 1e-5) & (res_df['x'] <= cur_x + L + 1e-5)
                span_data = res_df[span_mask]
                
                if not span_data.empty:
                    m_pos = span_data['moment'].max()
                    m_neg = abs(span_data['moment'].min())
                    v_max = span_data['shear'].abs().max()
                    d_eff = params['h'] - 0.05
                    
                    # เรียก Module ออกแบบ (หน่วย kN, kNm)
                    as_p, _, _, stp_p = rc_design.design_beam_flexure(m_pos, params['b'], d_eff, params['fc'], params['fy'])
                    as_n, _, _, stp_n = rc_design.design_beam_flexure(m_neg, params['b'], d_eff, params['fc'], params['fy'])
                    s_v, _, stp_v = rc_design.check_shear(v_max, params['b'], d_eff, params['fc'], params['fy'])
                    
                    # คำนวณจำนวนเส้นเหล็ก DB16
                    n_p = max(2, int(np.ceil((as_p * 1e6) / (np.pi * 8**2))))
                    n_n = max(2, int(np.ceil((as_n * 1e6) / (np.pi * 8**2))))
                    
                    design_results.append({'span': i+1, 'pos': {'n': n_p}, 'neg': {'n': n_n}, 'shear': {'s': s_v}})
                    
                    with st.expander(f"📑 Detailed Calculation: Span {i+1}"):
                        c_left, c_right = st.columns(2)
                        with c_left:
                            st.write("**Flexure Design (Bottom)**")
                            for s in stp_p: st.latex(s)
                        with c_right:
                            st.write("**Flexure Design (Top)**")
                            for s in stp_n: st.latex(s)
                        st.write("**Shear Design (Stirrups)**")
                        for s in stp_v: st.latex(s)
                cur_x += L

            st.markdown("---")
            st.subheader("🛠️ Detailing Preview")
            if design_results:
                dcol1, dcol2 = st.columns([1, 2])
                with dcol1:
                    st.pyplot(section_plotter.plot_section(params['b'], params['h'], 40, 16, design_results[0]['neg']['n'], design_results[0]['pos']['n'], f"RB6@{design_results[0]['shear']['s']*1000:.0f}", params['fc'], params['fy']))
                with dcol2:
                    st.pyplot(section_plotter.plot_longitudinal_section_detailed(spans, sup_df, design_results, params['h'], 40))

        with tab3:
            st.header("🧮 Load Combination & Equilibrium")
            
            st.subheader("1. Self-Weight (Dead Load)")
            st.info(f"Section {b_m}x{h_m}m | ρ=24kN/m³ | Base = {w_sw_base:.2f} kN/m | Factored = {w_sw_factored:.2f} kN/m")
            
            st.subheader("2. User Loads (Live Load)")
            if not loads_df.empty:
                st.dataframe(loads_df.assign(Factored_kN=lambda x: x['mag']*f_ll), use_container_width=True)
            
            st.subheader("3. Global Equilibrium Check")
            total_reac = sum(R_kN.values())
            total_load = w_sw_factored * sum(spans)
            if not loads_df.empty:
                for _, r in loads_df.iterrows():
                    total_load += (r['mag'] * f_ll) * (1.0 if r['type'] == 'P' else r['dist'])
            
            e_col1, e_col2 = st.columns(2)
            e_col1.write(f"Total Reaction: **{total_reac:.2f} kN**")
            e_col2.write(f"Total Applied: **{total_load:.2f} kN**")
            
            if abs(total_reac - total_load) < 0.1:
                st.success("✅ Equilibrium Verified")
            else:
                st.error(f"❌ Equilibrium Error: {abs(total_reac - total_load):.4f} kN")

    except Exception as e:
        st.error(f"❌ Application Error: {e}")
        st.exception(e)

st.markdown("---")
st.caption("Engineered by Gemini v3.0 | 2026 Structural Suite")
