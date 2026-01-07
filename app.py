import streamlit as st
import pandas as pd
import numpy as np

# Import custom modules
import input_handler
import solver
import rc_design
import design_view
import section_plotter

# --- PAGE CONFIG ---
st.set_page_config(page_title="RC Beam Pro", layout="wide", page_icon="🏗️")

# Custom CSS for better look
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; border: 1px solid #e0e0e0; }
    </style>
    """, unsafe_allow_html=True)

st.title("🏗️ RC Beam Analysis & Design Pro (Timoshenko)")
st.caption("Advanced Structural Analysis and Reinforced Concrete Design Suite")

# --- Sidebar Inputs ---
# รับค่า Parameters, จำนวน Span, ความยาว, ข้อมูล Support และ Load จากโมดูล input_handler
params, n_spans, spans, sup_df, loads_df, stable = input_handler.render_all_sidebar_inputs()

if not stable:
    st.error("🚨 **Structure is unstable!** Please check supports. A stable beam must have at least 3 reaction components (e.g., one pin and multiple rollers, or fixed support).")
else:
    # --- 0. Analysis Settings (Load Factors) ---
    st.markdown("### ⚙️ Analysis Settings & Load Factors")
    
    # เลือกระหว่าง Service Load (ใช้เช็ค Deflection) หรือ Ultimate Load (ใช้สำหรับออกแบบเหล็กเสริม)
    mode_select = st.radio(
        "Select Analysis Mode:",
        ["Service Load (1.0DL + 1.0LL)", "Ultimate / Custom Factors"],
        horizontal=True,
        help="Service load is typically used for deflection checks, while Ultimate load is used for strength design (As)."
    )
    
    col_fac1, col_fac2, col_fac3 = st.columns([1, 1, 2])
    
    # Logic การกำหนด Load Factor
    is_service = False
    if mode_select.startswith("Service"):
        f_dl, f_ll = 1.0, 1.0
        with col_fac1:
            st.number_input("Dead Load Factor (DL)", value=1.0, disabled=True, key="fdl_serv")
        with col_fac2:
            st.number_input("Live Load Factor (LL)", value=1.0, disabled=True, key="fll_serv")
        tag = "Service"
        st.info("ℹ️ **Service Mode:** Factors are set to 1.0. Ideal for Deflection and Crack control.")
        is_service = True
    else:
        with col_fac1:
            f_dl = st.number_input("Dead Load Factor (DL)", value=1.4, step=0.1, format="%.2f", key="fdl_ult")
        with col_fac2:
            f_ll = st.number_input("Live Load Factor (LL)", value=1.7, step=0.1, format="%.2f", key="fll_ult")
        tag = "Ultimate"
        st.warning(f"⚡ **Factored Mode:** Using {f_dl}DL + {f_ll}LL for Strength Design.")

    # --- 1. Prepare & Combine Loads ---
    try:
        # 1.1 คำนวณน้ำหนักบรรทุกคงที่จากน้ำหนักตัวเอง (Self-Weight)
        # ความหนาแน่นคอนกรีต = 24 kN/m^3
        w_sw_base_kN = params['b'] * params['h'] * 24.0  
        w_sw_factored_kN = w_sw_base_kN * f_dl
        w_sw_factored_N_m = w_sw_factored_kN * 1000.0 # แปลงเป็น N/m สำหรับ Solver
        
        # 1.2 เตรียม Bucket สำหรับเก็บ Load รวมแต่ละ Span
        span_total_udl = {i: w_sw_factored_N_m for i in range(n_spans)}
        combined_loads_list = []
        
        # 1.3 จัดการน้ำหนักบรรทุกจากผู้ใช้ (User Loads)
        if not loads_df.empty:
            for _, row in loads_df.iterrows():
                try:
                    s_idx = int(row['span_index'])
                    if s_idx >= n_spans: continue 
                    
                    l_type = row['type']
                    mag_base = row['mag']   # ค่าดิบที่กรอกมา
                    mag_factored = mag_base * f_ll 
                    
                    dist = row['dist'] 
                    current_span_len = spans[s_idx]
                    
                    # ตรวจสอบ: ถ้าเป็น UDL เต็มช่วง ให้รวมเข้ากับ Self-Weight เพื่อลดความซับซ้อนใน Solver
                    if l_type == 'U' and dist >= (current_span_len - 0.01):
                        span_total_udl[s_idx] += mag_factored
                    else:
                        # กรณี Point Load หรือ UDL บางส่วน
                        combined_loads_list.append({
                            'span_index': s_idx,
                            'type': l_type,
                            'mag': mag_factored,
                            'dist': dist,
                            'desc': 'User Load'
                        })
                except Exception:
                    continue
        
        # 1.4 นำ UDL ที่รวมแล้วใส่กลับเข้าไปใน List
        for i in range(n_spans):
            total_mag = span_total_udl[i]
            if total_mag > 0:
                combined_loads_list.append({
                    'span_index': i,
                    'type': 'U',
                    'mag': total_mag,
                    'dist': spans[i],
                    'desc': 'Total Combined UDL'
                })
                
        calc_loads_df = pd.DataFrame(combined_loads_list)

        # --- 2. Solve Beam ---
        # เรียกใช้ Solver หลักเพื่อคำนวณ Moment, Shear, Deflection และ Reactions
        x_eval, M, V, D, R = solver.solve_beam(spans, sup_df, calc_loads_df, params)
        
        res_df = pd.DataFrame({
            'x': x_eval,
            'moment': M,
            'shear': V,
            'deflection': D * 1000 # แปลงเป็น mm
        })
        
        # --- 3. DISPLAY RESULTS ---
        tab1, tab2 = st.tabs(["📊 1. Analysis Results & Checks", "📝 2. RC Design & Report"])
        
        # ================= TAB 1: DIAGRAMS & CHECKS =================
        with tab1:
            # PART 1: กราฟ SFD, BMD, Deflection
            # design_view จะถูกปรับแต่งให้ Plot จุดซ้ำที่ตำแหน่ง Point Load เพื่อให้ SFD เป็นเส้นดิ่ง
            st.plotly_chart(design_view.plot_analysis_results(res_df, spans, sup_df, calc_loads_df, R), use_container_width=True)
            
            st.info("💡 **SFD Logic:** Vertical jumps at point loads and supports are calculated based on concentrated forces for theoretical precision.")

            st.markdown("---")

            # PART 2: บทสรุปผลการคำนวณ (Analysis Summary)
            st.subheader("📌 Analysis Summary")
            
            v_max = res_df['shear'].abs().max()/1000
            m_max_pos = res_df['moment'].max()/1000
            m_max_neg = res_df['moment'].min()/1000
            d_abs_max = res_df['deflection'].abs().max()
            
            col_res1, col_res2, col_res3, col_res4 = st.columns(4)
            col_res1.metric(f"Max Shear ({tag})", f"{v_max:.2f} kN")
            col_res2.metric(f"Max Moment (+)", f"{m_max_pos:.2f} kNm")
            col_res3.metric(f"Max Moment (-)", f"{abs(m_max_neg):.2f} kNm")
            col_res4.metric("Max Deflection", f"{d_abs_max:.2f} mm")
            
            # PART 3: ตารางแรงปฏิกิริยา (Support Reactions)
            st.markdown("### 📍 Support Reactions")
            if R:
                try:
                    reaction_data = []
                    for key, val in R.items():
                        node_idx = int(str(key).replace('R', ''))
                        val_kN = val / 1000.0
                        reaction_data.append({
                            "Node": node_idx,
                            "Reaction (kN)": val_kN,
                            "Direction": "Upward" if val_kN > 0 else "Downward"
                        })
                    
                    df_reac = pd.DataFrame(reaction_data).sort_values(by="Node")
                    st.dataframe(df_reac.style.format({"Reaction (kN)": "{:.2f}"}), use_container_width=True, hide_index=True)
                except Exception as e:
                    st.warning(f"⚠️ Reaction Details: {R}")

            st.markdown("---")
            
            # PART 4: การตรวจสอบทางวิศวกรรม (Engineering Checks)
            # ส่วนนี้สำคัญมากสำหรับการส่งงาน เพื่อยืนยันว่าโปรแกรมคำนวณถูกต้อง (Sum Forces = 0)
            with st.expander("✅ Engineering Integrity Checks", expanded=True):
                ec1, ec2 = st.columns(2)
                
                with ec1:
                    st.markdown("### ⚖️ Static Equilibrium ($\Sigma F_y = 0$)")
                    try:
                        sum_R_kN = sum(R.values()) / 1000.0
                        # คำนวณโหลดทั้งหมดที่กดลง
                        sum_Load_kN = 0.0
                        # 1. SW
                        sum_Load_kN += (w_sw_factored_kN * sum(spans))
                        # 2. User Loads
                        if not loads_df.empty:
                            for _, row in loads_df.iterrows():
                                mag_k = row['mag'] / 1000.0 * f_ll
                                if row['type'] == 'P': sum_Load_kN += mag_k
                                elif row['type'] == 'U': sum_Load_kN += (mag_k * row['dist'])
                        
                        diff = abs(sum_R_kN - sum_Load_kN)
                        st.write(rf"Total Reactions ($\uparrow$): **{sum_R_kN:.2f} kN**")
                        st.write(rf"Total Applied Loads ($\downarrow$): **{sum_Load_kN:.2f} kN**")
                        
                        if diff < 0.1: # Tolerance 0.1 kN
                            st.success(f"✅ Equilibrium Satisfied (Diff: {diff:.4f} kN)")
                        else:
                            st.warning(f"⚠️ Check Equilibrium: Diff = {diff:.2f} kN")
                    except Exception:
                        st.error("Verification unavailable")

                with ec2:
                    st.markdown("### 📉 Serviceability (Deflection)")
                    max_span_L = max(spans) * 1000 
                    allowable_def = max_span_L / 240.0 # มาตรฐานทั่วไป L/240
                    
                    st.write(f"Max Span Deflection: **{d_abs_max:.2f} mm**")
                    st.write(f"Allowable Limit ($L/240$): **{allowable_def:.2f} mm**")
                    
                    if d_abs_max <= allowable_def:
                        st.success(f"✅ Deflection within limits")
                    else:
                        st.error(f"❌ Deflection Exceeds Limit")

            # PART 5: ตารางแจกแจงน้ำหนักบรรทุก (Load Breakdown)
            with st.expander("🧮 Detailed Load Breakdown Table", expanded=False):
                st.table(pd.DataFrame(detailed_load_rows) if 'detailed_load_rows' in locals() else pd.DataFrame(combined_loads_list))

        # ================= TAB 2: RC DESIGN =================
        with tab2:
            try:
                if is_service:
                    st.warning("⚠️ **Reminder:** You are viewing design based on 'Service Loads'. Strength design usually requires 'Ultimate Loads' (e.g., 1.4DL + 1.7LL).")
                
                st.header(f"Reinforced Concrete Design ({tag})")
                
                design_res = []
                span_start = 0
                for i, span_len in enumerate(spans):
                    span_end = span_start + span_len
                    mask = (res_df['x'] >= span_start) & (res_df['x'] <= span_end)
                    span_data = res_df[mask]
                    
                    if not span_data.empty:
                        mu_pos = span_data['moment'].max() / 1000 
                        mu_neg = abs(span_data['moment'].min()) / 1000
                        vu_max = span_data['shear'].abs().max() / 1000
                        d = params['h'] - (params.get('cover', 40)/1000) - 0.01 # Effective depth
                        
                        # --- Flexure & Shear Design Logic ---
                        As_pos, rho_pos, _, steps_pos = rc_design.design_beam_flexure(mu_pos, params['b'], d, params['fc'], params['fy'])
                        As_neg, rho_neg, _, steps_neg = rc_design.design_beam_flexure(mu_neg, params['b'], d, params['fc'], params['fy'])
                        s_req, _, steps_shear = rc_design.check_shear(vu_max, params['b'], d, params['fc'], params['fy'])
                        
                        # คำนวณจำนวนเหล็กเส้นเบื้องต้น (สมมติ DB16)
                        def get_bars(As): 
                            return max(2, int(np.ceil(As / (np.pi * (0.008)**2)))) 
                        
                        design_res.append({
                            'span': i+1,
                            'pos': {'As': As_pos, 'n': get_bars(As_pos)},
                            'neg': {'As': As_neg, 'n': get_bars(As_neg)},
                            'shear': {'s': s_req}
                        })
                        
                        with st.expander(f"📘 Span {i+1} Calculation Steps", expanded=False):
                            c1, c2 = st.columns(2)
                            with c1:
                                st.markdown("**Bottom Reinforcement (Positive Moment)**")
                                for s in steps_pos: st.latex(s)
                            with c2:
                                st.markdown("**Top Reinforcement (Negative Moment)**")
                                for s in steps_neg: st.latex(s)
                            st.markdown("---")
                            st.markdown("**Shear Reinforcement Design**")
                            for s in steps_shear: st.latex(s)

                    span_start += span_len

                st.markdown("---")
                st.subheader("🛠️ Detailing Preview")
                c_det1, c_det2 = st.columns([1, 2])
                with c_det1:
                    st.write("**Cross Section**")
                    if design_res:
                        # แสดงผลภาพหน้าตัด
                        fig_sec = section_plotter.plot_section(
                            params['b'], params['h'], 40, 16, 
                            design_res[0]['neg']['n'], design_res[0]['pos']['n'], 
                            f"RB6@{design_res[0]['shear']['s']*1000:.0f}mm", params['fc'], params['fy']
                        )
                        st.pyplot(fig_sec)
                with c_det2:
                    st.write("**Longitudinal Section**")
                    if design_res:
                        # แสดงผลภาพรูปด้านตามยาว
                        fig_long = section_plotter.plot_longitudinal_section_detailed(spans, sup_df, design_res, params['h'], 40)
                        st.pyplot(fig_long)
            except Exception as e:
                st.error(f"Design Module Error: {e}")

    except Exception as e:
        st.error(f"❌ **System Error:** {e}")
        st.info("Advice: Check if span lengths are > 0 and supports are within the beam range.")
