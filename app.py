import streamlit as st
import pandas as pd
import numpy as np

# --- 1. Import Custom Modules ---
import input_handler
import solver
import rc_design
import design_view
import section_plotter

# --- 2. Page Configuration ---
st.set_page_config(
    page_title="RC Beam Analysis & Design Pro", 
    layout="wide", 
    page_icon="🏗️"
)

# Custom Style เพื่อความสวยงาม
st.markdown("""
    <style>
    .stMetric { background-color: #f8f9fa; padding: 10px; border-radius: 5px; border: 1px solid #dee2e6; }
    </style>
    """, unsafe_allow_html=True)

st.title("🏗️ RC Beam Analysis & Design Pro (Timoshenko)")

# --- 3. Sidebar Inputs ---
# รับค่าจากโมดูล input_handler
try:
    params, n_spans, spans, sup_df, loads_df, stable = input_handler.render_all_sidebar_inputs()
except Exception as e:
    st.error(f"❌ Error ในส่วน Input Sidebar: {e}")
    st.stop()

if not stable:
    st.error("🚨 **Structure is unstable!** โปรดตรวจสอบจุดรองรับ (ต้องมีจุดยึดรั้งอย่างน้อย 3 จุด เพื่อความมั่นคงทางโครงสร้าง)")
else:
    # --- 4. Analysis Settings & Load Factors ---
    st.markdown("### ⚙️ Analysis Settings & Load Factors")
    
    mode_select = st.radio(
        "เลือกโหมดการวิเคราะห์ (Analysis Mode):",
        ["Service Load (1.0DL + 1.0LL)", "Ultimate / Custom Factors"],
        horizontal=True
    )
    
    col_fac1, col_fac2, col_fac3 = st.columns([1, 1, 2])
    
    is_service = False
    if mode_select.startswith("Service"):
        f_dl, f_ll = 1.0, 1.0
        with col_fac1:
            st.number_input("Dead Load Factor (DL)", value=1.0, disabled=True, key="fdl_serv")
        with col_fac2:
            st.number_input("Live Load Factor (LL)", value=1.0, disabled=True, key="fll_serv")
        tag = "Service"
        st.info("ℹ️ ใช้ **Service Load** สำหรับการเช็คระยะตกท้องช้าง (Deflection Check)")
        is_service = True
    else:
        with col_fac1:
            f_dl = st.number_input("Dead Load Factor (DL)", value=1.4, step=0.1, format="%.2f", key="fdl_ult")
        with col_fac2:
            f_ll = st.number_input("Live Load Factor (LL)", value=1.7, step=0.1, format="%.2f", key="fll_ult")
        tag = "Ultimate"
        st.warning(f"⚡ ใช้ **Factored Load**: {f_dl}DL + {f_ll}LL สำหรับการออกแบบกำลัง")

    # --- 5. Load Combination Logic ---
    try:
        # 5.1 คำนวณน้ำหนักตัวเอง (Self-Weight)
        w_sw_base_kN = params['b'] * params['h'] * 24.0  # kN/m
        w_sw_factored_kN = w_sw_base_kN * f_dl
        w_sw_factored_N_m = w_sw_factored_kN * 1000.0
        
        # 5.2 เตรียม List สำหรับ Solver
        span_total_udl = {i: w_sw_factored_N_m for i in range(n_spans)}
        combined_loads_list = []
        
        # 5.3 ประมวลผลน้ำหนักบรรทุกจากผู้ใช้
        if not loads_df.empty:
            for _, row in loads_df.iterrows():
                s_idx = int(row['span_index'])
                if s_idx >= n_spans: continue
                
                l_type = row['type']
                mag_factored = row['mag'] * f_ll
                dist = row['dist']
                
                # หากเป็น UDL เต็มช่วง ให้รวมเข้ากับ Self-weight เพื่อความเร็วในการคำนวณ
                if l_type == 'U' and dist >= (spans[s_idx] - 0.01):
                    span_total_udl[s_idx] += mag_factored
                else:
                    combined_loads_list.append({
                        'span_index': s_idx,
                        'type': l_type,
                        'mag': mag_factored,
                        'dist': dist,
                        'desc': 'User Load'
                    })
        
        # รวม UDL ที่ประมวลผลแล้วเข้าสู่ list หลัก
        for i in range(n_spans):
            if span_total_udl[i] > 0:
                combined_loads_list.append({
                    'span_index': i,
                    'type': 'U',
                    'mag': span_total_udl[i],
                    'dist': spans[i],
                    'desc': 'Combined UDL'
                })
        
        calc_loads_df = pd.DataFrame(combined_loads_list)

        # --- 6. Beam Solver Execution ---
        x_eval, M, V, D, R = solver.solve_beam(spans, sup_df, calc_loads_df, params)
        
        res_df = pd.DataFrame({
            'x': x_eval,
            'moment': M,
            'shear': V,
            'deflection': D * 1000  # แปลงเป็น mm
        })

        # --- 7. Display Results & Tabs ---
        tab1, tab2 = st.tabs(["📊 1. Analysis Results & Checks", "📝 2. RC Design & Report"])

        # ================= TAB 1: Analysis =================
        with tab1:
            # กราฟ Diagrams
            st.plotly_chart(design_view.plot_analysis_results(res_df, spans, sup_df, calc_loads_df, R), use_container_width=True)
            
            # สรุปค่าสูงสุด
            st.subheader("📌 Analysis Summary")
            v_max = res_df['shear'].abs().max() / 1000
            m_max_pos = res_df['moment'].max() / 1000
            m_max_neg = res_df['moment'].min() / 1000
            d_abs_max = res_df['deflection'].abs().max()
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric(f"Max Shear ({tag})", f"{v_max:.2f} kN")
            c2.metric("Max Moment (+)", f"{m_max_pos:.2f} kNm")
            c3.metric("Max Moment (-)", f"{abs(m_max_neg):.2f} kNm")
            c4.metric("Max Deflection", f"{d_abs_max:.2f} mm")

            # ตารางแรงปฏิกิริยา
            st.markdown("### 📍 Support Reactions")
            reaction_list = [{"Node": k.replace('R',''), "Reaction (kN)": v/1000} for k, v in R.items()]
            df_reac = pd.DataFrame(reaction_list).sort_values(by="Node")
            st.dataframe(df_reac.style.format({"Reaction (kN)": "{:.2f}"}), use_container_width=True, hide_index=True)

            # ตรวจสอบความสมดุล (Equilibrium Check)
            with st.expander("⚖️ Engineering Checks (Equilibrium & Deflection)", expanded=True):
                ec1, ec2 = st.columns(2)
                with ec1:
                    st.markdown("**Static Equilibrium ($\Sigma F_y = 0$)**")
                    sum_R = sum(R.values()) / 1000
                    # คำนวณโหลดรวมเพื่อเปรียบเทียบ
                    total_load_kN = (calc_loads_df[calc_loads_df['type']=='U']['mag']/1000 * calc_loads_df[calc_loads_df['type']=='U']['dist']).sum() + \
                                     (calc_loads_df[calc_loads_df['type']=='P']['mag']/1000).sum()
                    st.write(f"Total Reactions: {sum_R:.2f} kN")
                    st.write(f"Total Loads: {total_load_kN:.2f} kN")
                    if abs(sum_R - total_load_kN) < 0.1:
                        st.success("✅ Equilibrium Pass")
                    else:
                        st.warning("⚠️ Difference detected (Check load inputs)")
                
                with ec2:
                    st.markdown("**Deflection Control**")
                    L_max = max(spans) * 1000
                    limit = L_max / 240
                    st.write(f"Actual: {d_abs_max:.2f} mm")
                    st.write(f"Limit (L/240): {limit:.2f} mm")
                    if d_abs_max <= limit: st.success("✅ Pass")
                    else: st.error("❌ Exceeds Limit")

        # ================= TAB 2: Design =================
        with tab2:
            if is_service:
                st.warning("⚠️ คุณกำลังอยู่ในโหมด Service Load กรุณาเปลี่ยนเป็น Ultimate Load เพื่อคำนวณเหล็กเสริมเสริม")
            
            st.header(f"Reinforced Concrete Design ({tag})")
            
            design_res = []
            span_start = 0
            for i, span_len in enumerate(spans):
                span_end = span_start + span_len
                span_data = res_df[(res_df['x'] >= span_start) & (res_df['x'] <= span_end)]
                
                if not span_data.empty:
                    mu_pos = span_data['moment'].max() / 1000
                    mu_neg = abs(span_data['moment'].min()) / 1000
                    vu_max = span_data['shear'].abs().max() / 1000
                    d = params['h'] - 0.05 # d โดยประมาณ
                    
                    # เรียกใช้โมดูลออกแบบ
                    as_pos, _, _, steps_pos = rc_design.design_beam_flexure(mu_pos, params['b'], d, params['fc'], params['fy'])
                    as_neg, _, _, steps_neg = rc_design.design_beam_flexure(mu_neg, params['b'], d, params['fc'], params['fy'])
                    s_req, _, steps_shear = rc_design.check_shear(vu_max, params['b'], d, params['fc'], params['fy'])
                    
                    def get_n_bars(As): return max(2, int(np.ceil(As / (np.pi * (0.008)**2))))
                    
                    design_res.append({
                        'span': i+1,
                        'pos': {'As': as_pos, 'n': get_n_bars(as_pos)},
                        'neg': {'As': as_neg, 'n': get_n_bars(as_neg)},
                        'shear': {'s': s_req}
                    })
                    
                    with st.expander(f"📘 Detail Span {i+1} (Mu+={mu_pos:.1f}, Mu-={mu_neg:.1f})", expanded=False):
                        c_flex1, c_flex2 = st.columns(2)
                        with c_flex1:
                            st.write("**เหล็กล่าง (Bottom)**")
                            for step in steps_pos: st.latex(step)
                        with c_flex2:
                            st.write("**เหล็กบน (Top)**")
                            for step in steps_neg: st.latex(step)
                        st.write("**การออกแบบแรงเฉือน (Shear)**")
                        for step in steps_shear: st.latex(step)
                
                span_start += span_len

            # การแสดงผลแบบรูปวาด
            st.subheader("3. Detailing Preview")
            if design_res:
                col_p1, col_p2 = st.columns([1, 2])
                with col_p1:
                    st.write("**หน้าตัดคาน (Cross Section)**")
                    fig_sec = section_plotter.plot_section(params['b'], params['h'], 40, 16, design_res[0]['neg']['n'], design_res[0]['pos']['n'], "RB6@200", params['fc'], params['fy'])
                    st.pyplot(fig_sec)
                with col_p2:
                    st.write("**รูปด้านตามยาว (Longitudinal Section)**")
                    fig_long = section_plotter.plot_longitudinal_section_detailed(spans, sup_df, design_res, params['h'], 40)
                    st.pyplot(fig_long)

    except Exception as e:
        st.error(f"❌ เกิดข้อผิดพลาดในการคำนวณ: {e}")
        st.write("คำแนะนำ: ตรวจสอบการป้อนค่า Load หรือความยาว Span ให้ถูกต้อง (ต้องไม่เป็น 0)")
