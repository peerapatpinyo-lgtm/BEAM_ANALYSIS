import streamlit as st
import pandas as pd
import input_handler
import solver # Calls the NEW solver.py
import design_view
import rc_design # Import Module RC

# ... (Page Config & Header code remains same) ...

def main():
    # ... (Sidebar & Input code remains same) ...
    # 1. Sidebar Settings
    params = input_handler.render_sidebar()
    # เพิ่ม Parameters สำหรับ RC Design เข้าไปใน params (ถ้า input_handler ยังไม่ส่งมา ต้องเพิ่ม default หรือรับค่า)
    # สมมติ input_handler ใน sidebar มีรับค่า fc, fy แล้ว หรือเรา Hardcode test ไปก่อน
    if 'fc' not in params: params.update({'fc': 240, 'fy': 4000, 'fys': 2400, 'db_main': 12, 'db_stirrup': 6, 'unit': 'Metric'})

    # 2. Model Inputs
    n_spans, spans, sup_df, stable = input_handler.render_model_inputs(params)
    
    st.markdown("---")
    
    # 3. Loads
    raw_loads = input_handler.render_loads(n_spans, spans, params, sup_df)
    
    st.markdown("---")

    # 4. Calculation
    if st.button("🚀 Run Analysis & Design", type="primary", use_container_width=True):
        if not stable:
            st.error("❌ Structure is Unstable!")
            return
            
        # --- Prepare Loads (Factor) ---
        factored_loads_list = []
        if raw_loads is not None and not raw_loads.empty:
            raw_dict = raw_loads.to_dict('records')
            for l in raw_dict:
                factor = params['gamma_dead'] if l['case'] == 'DL' else params['gamma_live']
                new_load = l.copy()
                new_load['mag'] = l['mag'] * factor
                # Handle UDL end/dist for new solver
                if l['type'] == 'U':
                     # If code uses spans list, find length
                     span_len = spans[int(l['span_idx'])]
                     # Assuming full span if dist not specified
                     new_load['dist'] = l.get('dist', span_len - l['x'])
                factored_loads_list.append(new_load)
        
        factored_loads_df = pd.DataFrame(factored_loads_list) if factored_loads_list else pd.DataFrame()

        # Initialize NEW Solver
        beam_solver = solver.BeamSolver(spans, sup_df, factored_loads_df, E=params['E'], I=params['I'])
        
        try:
            # Solve
            df_results, reactions = beam_solver.solve()
            
            # --- 5. Visualization ---
            design_view.draw_interactive_diagrams(
                df_results, reactions, spans, sup_df, raw_loads, 
                unit_force=params['u_force'], unit_len=params['u_len'],
                dl_factor=params['gamma_dead'], ll_factor=params['gamma_live']
            )
            
            # --- 6. Results & RC Design ---
            c1, c2 = st.columns([1, 2])
            with c1:
                design_view.render_result_tables(df_results, reactions, spans, params['u_force'], params['u_len'])
            
            with c2:
                st.markdown("### 🏗️ RC Design Results")
                # Find Max Positive and Negative Moments
                max_pos_M = df_results['moment'].max()
                max_neg_M = df_results['moment'].min()
                max_V = df_results['shear'].abs().max()
                
                # Design Tabs
                tab1, tab2 = st.tabs(["Top/Bottom Rebar", "Shear Links"])
                
                with tab1:
                    # Positive Moment Design (Bottom Steel)
                    if max_pos_M > 0:
                        res_pos = rc_design.calculate_flexure_sdm(max_pos_M, "Max Positive (Bottom)", 30, 60, 4, params)
                        st.success(f"**{res_pos['Type']}**")
                        st.write(f"Use: **{res_pos['Bars']}**")
                        with st.expander("Calculation Details"):
                            for line in res_pos['Log']: st.write(line)
                    
                    st.divider()
                    
                    # Negative Moment Design (Top Steel)
                    if max_neg_M < 0:
                        res_neg = rc_design.calculate_flexure_sdm(abs(max_neg_M), "Max Negative (Top)", 30, 60, 4, params)
                        st.error(f"**{res_neg['Type']}**")
                        st.write(f"Use: **{res_neg['Bars']}**")
                        with st.expander("Calculation Details"):
                            for line in res_neg['Log']: st.write(line)

                with tab2:
                     res_shear = rc_design.calculate_shear_capacity(max_V, 30, 60, 4, params)
                     st.info(f"**Max Shear: {max_V:.2f}**")
                     st.write(f"Stirrups: **{res_shear['Stirrups']}**")
                     with st.expander("Shear Check Log"):
                            for line in res_shear['Log']: st.write(line)

        except Exception as e:
            st.error(f"Analysis Failed: {str(e)}")
            st.exception(e) # Show stack trace for debugging

if __name__ == "__main__":
    main()
