import streamlit as st
import pandas as pd
import numpy as np

# --- 1. IMPORT CUSTOM MODULES ---
import input_handler
import solver
import rc_design
import design_view
import section_plotter

# --- 2. PAGE CONFIGURATION ---
st.set_page_config(page_title="Beam Analysis & Design Pro", layout="wide")
st.title("🏗️ RC Beam Analysis & Design Pro (Timoshenko)")

# --- 3. SIDEBAR INPUTS ---
params, n_spans, spans, sup_df, loads_df, stable = input_handler.render_all_sidebar_inputs()

if not stable:
    st.error("🚨 Structure is unstable! Please check supports (Must have at least 3 reaction components).")
else:
    # --- 4. ANALYSIS SETTINGS & LOAD FACTORS ---
    st.markdown("### ⚙️ Analysis Settings & Load Factors")
    
    mode_select = st.radio(
        "Select Analysis Mode:",
        ["Service Load (1.0DL + 1.0LL)", "Ultimate / Custom Factors"],
        horizontal=True
    )
    
    col_fac1, col_fac2, _ = st.columns([1, 1, 2])
    
    is_service = False
    if mode_select.startswith("Service"):
        f_dl, f_ll = 1.0, 1.0
        with col_fac1:
            st.number_input("Dead Load Factor (DL)", value=1.0, disabled=True, key="fdl_serv")
        with col_fac2:
            st.number_input("Live Load Factor (LL)", value=1.0, disabled=True, key="fll_serv")
        tag = "Service"
        st.info("ℹ️ Using **Service Load** (Factors = 1.0) for Deflection & Serviceability checks.")
        is_service = True
    else:
        with col_fac1:
            f_dl = st.number_input("Dead Load Factor (DL)", value=1.4, step=0.1, format="%.2f", key="fdl_ult")
        with col_fac2:
            f_ll = st.number_input("Live Load Factor (LL)", value=1.7, step=0.1, format="%.2f", key="fll_ult")
        tag = "Ultimate"
        st.warning(f"⚡ Using **Factored Load**: {f_dl} DL + {f_ll} LL for Strength Design.")

    # --- 5. LOAD CALCULATIONS & COMBINATIONS ---
    try:
        # 5.1 Self-Weight Calculation (Unit Weight = 24 kN/m³)
        w_sw_base_kN = params['b'] * params['h'] * 24.0   
        w_sw_factored_kN = w_sw_base_kN * f_dl
        
        # 5.2 Initialize Total UDL per span (Newton for Solver)
        span_total_udl_N = {i: w_sw_factored_kN * 1000.0 for i in range(n_spans)} 
        combined_loads_solver = [] # สำหรับส่งเข้า solver (หน่วย N)
        combined_loads_plot = []   # สำหรับส่งเข้า design_view (หน่วย kN)
        
        # 5.3 Process User-Defined Loads
        if not loads_df.empty:
            for _, row in loads_df.iterrows():
                try:
                    s_idx = int(row['span_index'])
                    if s_idx >= n_spans: continue 
                    
                    l_type = row['type']
                    mag_base_kN = float(row['mag'])
                    mag_factored_kN = mag_base_kN * f_ll
                    dist = float(row['dist']) 
                    
                    if l_type == 'U' and dist >= (spans[s_idx] - 0.01):
                        span_total_udl_N[s_idx] += (mag_factored_kN * 1000.0)
                    else:
                        # ข้อมูลสำหรับ Solver (N)
                        combined_loads_solver.append({
                            'span_index': s_idx, 'type': l_type,
                            'mag': mag_factored_kN * 1000.0, 'dist': dist
                        })
                        # ข้อมูลสำหรับ Plot (kN) เพื่อให้รูปโชว์ 6.6 ไม่ใช่ 6600
                        combined_loads_plot.append({
                            'span_index': s_idx, 'type': l_type,
                            'mag': mag_factored_kN, 'dist': dist, 'desc': 'User'
                        })
                except Exception: continue
        
        # 5.4 Merge combined UDLs
        for i in range(n_spans):
            if span_total_udl_N[i] > 0:
                mag_N = span_total_udl_N[i]
                combined_loads_solver.append({
                    'span_index': i, 'type': 'U', 'mag': mag_N, 'dist': spans[i]
                })
                combined_loads_plot.append({
                    'span_index': i, 'type': 'U', 'mag': mag_N / 1000.0, 'dist': spans[i], 'desc': 'Total UDL'
                })
        
        calc_loads_df = pd.DataFrame(combined_loads_solver)
        plot_loads_df = pd.DataFrame(combined_loads_plot)

        # --- 6. BEAM SOLVER ---
        x_eval, M, V, D, R = solver.solve_beam(spans, sup_df, calc_loads_df, params)
        
        # ปรับหน่วยผลลัพธ์จาก N เป็น kN สำหรับการแสดงผล
        res_df_kN = pd.DataFrame({
            'x': x_eval,
            'moment': M / 1000.0,
            'shear': V / 1000.0,
            'deflection': D * 1000 # m to mm
        })
        R_kN = {k: v / 1000.0 for k, v in R.items()}
        
        # --- 7. DISPLAY RESULTS ---
        tab1, tab2 = st.tabs(["📊 1. Analysis Results & Checks", "📝 2. RC Design & Report"])
        
        with tab1:
            # ใช้ plot_loads_df ที่เป็น kN และ res_df_kN เพื่อให้กราฟแสดงตัวเลขถูกต้อง
            st.plotly_chart(design_view.plot_analysis_results(res_df_kN, spans, sup_df, plot_loads_df, R_kN), use_container_width=True)
            
            st.subheader("📌 Analysis Summary")
            v_max = res_df_kN['shear'].abs().max()
            m_max_pos = res_df_kN['moment'].max()
            m_max_neg = res_df_kN['moment'].min()
            d_abs_max = res_df_kN['deflection'].abs().max()
            
            c_res1, c_res2, c_res3, c_res4 = st.columns(4)
            c_res1.metric(f"Max Shear ({tag})", f"{v_max:.2f} kN")
            c_res2.metric("Max Moment (+)", f"{m_max_pos:.2f} kNm")
            c_res3.metric("Max Moment (-)", f"{abs(m_max_neg):.2f} kNm")
            c_res4.metric("Max Deflection", f"{d_abs_max:.2f} mm")

            st.markdown("### 📍 Support Reactions")
            if R_kN:
                reaction_data = [{"Node": int(str(k).replace('R', '')), "Reaction (kN)": v} for k, v in R_kN.items()]
                df_reac = pd.DataFrame(reaction_data).sort_values(by="Node")
                st.dataframe(df_reac.style.format({"Reaction (kN)": "{:.2f}"}), use_container_width=True, hide_index=True)

            st.markdown("---")
            with st.expander("🧮 Detailed Load Calculation Report", expanded=True):
                st.markdown("#### A. Self-Weight Calculation (Dead Load)")
                sw_rep = [{"Span": i+1, "Result": f"{w_sw_factored_kN:.2f} kN/m"} for i in range(n_spans)]
                st.table(pd.DataFrame(sw_rep))

                st.markdown("#### B. Load Combination Breakdown")
                combo_report = []
                for i in range(n_spans):
                    combo_report.append({"Span": i+1, "Type": "SW (DL)", "Unfactored": f"{w_sw_base_kN:.2f}", "Factor": f"x{f_dl}", "Factored": f"{w_sw_factored_kN:.2f} kN/m"})
                if not loads_df.empty:
                    for _, row in loads_df.iterrows():
                        val = float(row['mag'])
                        combo_report.append({"Span": int(row['span_index'])+1, "Type": row['type'], "Unfactored": f"{val:.2f}", "Factor": f"x{f_ll}", "Factored": f"{(val * f_ll):.2f} kN"})
                st.table(pd.DataFrame(combo_report))

            with st.expander("✅ Equilibrium & Deflection Checks", expanded=True):
                ec1, ec2 = st.columns(2)
                with ec1:
                    sum_R = sum(R_kN.values())
                    total_sw = w_sw_factored_kN * sum(spans)
                    total_user = 0
                    if not loads_df.empty:
                        for _, r in loads_df.iterrows():
                            total_user += (float(r['mag']) * f_ll) * (1.0 if r['type'] == 'P' else float(r['dist']))
                    total_applied = total_sw + total_user
                    st.write(f"Total Reactions: **{sum_R:.2f} kN**")
                    st.write(f"Total Applied Loads: **{total_applied:.2f} kN**")
                    if abs(sum_R - total_applied) < 0.1: st.success("Balance Check: PASS")

                with ec2:
                    limit = (max(spans) * 1000) / 240.0
                    st.write(f"Max Deflection: {d_abs_max:.2f} mm")
                    st.write(f"Limit (L/240): {limit:.2f} mm")
                    if d_abs_max <= limit: st.success("Deflection: PASS")

        # ================= TAB 2: RC DESIGN =================
        with tab2:
            if is_service: st.warning("⚠️ Warning: Strength Design requires Ultimate factors.")
            st.header(f"Reinforced Concrete Design ({tag})")
            design_res = []
            span_start = 0
            for i, span_len in enumerate(spans):
                span_end = span_start + span_len
                span_data = res_df_kN[(res_df_kN['x'] >= span_start - 1e-6) & (res_df_kN['x'] <= span_end + 1e-6)]
                if not span_data.empty:
                    mu_pos, mu_neg = span_data['moment'].max(), abs(span_data['moment'].min())
                    vu_max, d_eff = span_data['shear'].abs().max(), params['h'] - 0.05
                    As_pos, _, _, steps_pos = rc_design.design_beam_flexure(mu_pos, params['b'], d_eff, params['fc'], params['fy'])
                    As_neg, _, _, steps_neg = rc_design.design_beam_flexure(mu_neg, params['b'], d_eff, params['fc'], params['fy'])
                    s_req, _, steps_shear = rc_design.check_shear(vu_max, params['b'], d_eff, params['fc'], params['fy'])
                    def n_bars(As): return max(2, int(np.ceil(As / (np.pi * 0.008**2)))) 
                    design_res.append({'span': i+1, 'pos': {'n': n_bars(As_pos)}, 'neg': {'n': n_bars(As_neg)}, 'shear': {'s': s_req}})
                    with st.expander(f"📘 Detailed Design: Span {i+1}"):
                        st.write(f"Mu+: {mu_pos:.2f} kNm, Mu-: {mu_neg:.2f} kNm")
                        for s in steps_pos: st.latex(s)
                span_start += span_len
            st.markdown("---")
            if design_res:
                col_det1, col_det2 = st.columns([1, 2])
                with col_det1: st.pyplot(section_plotter.plot_section(params['b'], params['h'], 40, 16, design_res[0]['neg']['n'], design_res[0]['pos']['n'], "RB6@200", params['fc'], params['fy']))
                with col_det2: st.pyplot(section_plotter.plot_longitudinal_section_detailed(spans, sup_df, design_res, params['h'], 40))
    except Exception as e:
        st.error(f"❌ Error: {e}")
        st.exception(e)
