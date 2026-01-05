# app.py (Modified Version)
import streamlit as st
import pandas as pd
import numpy as np

# Import custom modules
import input_handler
import beam_analysis  # ใช้ไฟล์ใหม่แทน solver.py
import rc_design      # เพิ่มส่วนออกแบบ
import design_view

# Page Config
st.set_page_config(page_title="Beam Analysis Pro", layout="wide", page_icon="🏗️")

def main():
    st.title("🏗️ Structural Beam Analysis & Design")
    st.markdown("---")

    # --- 1. Sidebar Settings ---
    params = input_handler.render_sidebar()

    # --- 2. Model Inputs ---
    n_spans, spans, sup_df, stable = input_handler.render_model_inputs(params)
    
    st.markdown("---")

    # --- 3. Loads Input (Service Loads) ---
    raw_loads = input_handler.render_loads(n_spans, spans, params, sup_df)

    st.markdown("---")

    # --- 4. Calculation Loop ---
    if st.button("🚀 Run Analysis & Design", type="primary", use_container_width=True):
        if not stable:
            st.error("❌ Structure is Unstable! Please check supports.")
            return
            
        # A. LOAD FACTORING (Service -> Ultimate)
        factored_loads_list = []
        if raw_loads is not None and not raw_loads.empty:
            raw_dict = raw_loads.to_dict('records')
            for l in raw_dict:
                factor = 1.0
                if l['case'] == 'DL': factor = params['gamma_dead']
                elif l['case'] == 'LL': factor = params['gamma_live']
                
                new_load = l.copy()
                new_load['mag'] = l['mag'] * factor
                factored_loads_list.append(new_load)
        
        factored_loads_df = pd.DataFrame(factored_loads_list) if factored_loads_list else None

        # B. ANALYSIS ENGINE
        # เรียกใช้ Class จาก beam_analysis.py
        engine = beam_analysis.BeamAnalysisEngine(spans, sup_df, factored_loads_list) 
        
        try:
            # Run Solver
            df_results, reactions = engine.solve()
            
            if df_results is None:
                st.error("Analysis Error: Could not solve the structure.")
                return

            # C. VISUALIZATION (SFD, BMD, Deflection)
            design_view.draw_interactive_diagrams(
                df_results, 
                reactions, 
                spans, 
                sup_df, 
                raw_loads.to_dict('records') if raw_loads is not None else [], 
                unit_force=params['u_force'], 
                unit_len=params['u_len'],
                dl_factor=params['gamma_dead'],
                ll_factor=params['gamma_live']
            )
            
            design_view.render_result_tables(df_results, reactions, spans, params['u_force'], params['u_len'])

            # D. RC DESIGN (ส่วนที่เพิ่มเข้ามา)
            st.markdown("---")
            st.header("🧱 Reinforced Concrete Design")
            
            # หาค่า Max Moment (+/-) และ Max Shear ในแต่ละช่วงคาน
            design_results = []
            cum_dist = [0] + list(np.cumsum(spans))
            
            for i in range(len(spans)):
                start_x = cum_dist[i]
                end_x = cum_dist[i+1]
                
                # Filter results for this span
                span_res = df_results[(df_results['x'] >= start_x) & (df_results['x'] <= end_x)]
                
                # 1. Get Critical Values
                mu_pos = span_res['moment'].max()
                mu_neg = span_res['moment'].min()
                vu_max = span_res['shear'].abs().max()
                
                # 2. Design Section (Positive Moment - Midspan)
                if mu_pos > 0:
                    res_pos = rc_design.calculate_flexure_sdm(mu_pos, f"Span {i+1} (+M)", params['b'], params['h'], params['cover'], params)
                    design_results.append(res_pos)
                
                # 3. Design Section (Negative Moment - Support)
                if abs(mu_neg) > 0:
                    res_neg = rc_design.calculate_flexure_sdm(mu_neg, f"Span {i+1} (-M)", params['b'], params['h'], params['cover'], params)
                    design_results.append(res_neg)
                    
                # 4. Design Shear (Stirrups)
                _, _, stir_txt, shear_log = rc_design.calculate_shear_capacity(vu_max, params['b'], params['h'], params['cover'], params)
                
                # Append Shear info to the last design entry or create new if needed
                # (For simplicity, we show Shear as a separate note or column in a real table)
            
            # แสดงผลการออกแบบแบบการ์ด
            if design_results:
                cols = st.columns(len(design_results))
                for idx, res in enumerate(design_results):
                    # Wrap cards if too many
                    with cols[idx % len(cols)]: 
                        status_color = "green" if "OK" in res['Status'] else "red"
                        st.markdown(f"""
                        <div style="padding:10px; border:1px solid #ddd; border-radius:5px; margin-bottom:10px;">
                            <h4>{res['Type']}</h4>
                            <p><b>Mu:</b> {res['Mu']:.2f}</p>
                            <p><b>Rebar:</b> {res['Bars']}</p>
                            <p style="color:{status_color};"><b>{res['Status']}</b></p>
                            <details><summary>Calc Log</summary>
                            <small>{'<br>'.join(res['Log'])}</small>
                            </details>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.info("Moment is zero, no reinforcement calculation needed.")

        except Exception as e:
            st.error(f"System Error: {str(e)}")
            st.exception(e)

if __name__ == "__main__":
    main()
