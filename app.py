import streamlit as st

import pandas as pd

import input_handler

import solver

import design_view



# Page Config

st.set_page_config(page_title="Beam Analysis Pro", layout="wide", page_icon="🏗️")



def main():

    st.title("🏗️ Structural Beam Analysis Professional")

    st.markdown("---")



    # 1. Sidebar Settings

    params = input_handler.render_sidebar()



    # 2. Model Inputs

    n_spans, spans, sup_df, stable = input_handler.render_model_inputs(params)

    

    st.markdown("---")



    # 3. Loads Input

    # loads ที่รับมาคือ Unfactored Load (Raw Data)

    raw_loads = input_handler.render_loads(n_spans, spans, params, sup_df)



    st.markdown("---")



    # 4. Calculation & Solver

    if st.button("🚀 Run Analysis (Factored)", type="primary", use_container_width=True):

        if not stable:

            st.error("❌ Structure is Unstable!")

            return

            

        # --- PRE-PROCESS: Apply Load Factors ---

        # สร้างรายการ Load ใหม่ที่คูณ Factor แล้วเพื่อส่งให้ Solver

        factored_loads_list = []

        

        if raw_loads is not None and not raw_loads.empty:

            raw_dict = raw_loads.to_dict('records')

            for l in raw_dict:

                factor = 1.0

                if l['case'] == 'DL':

                    factor = params['gamma_dead']

                elif l['case'] == 'LL':

                    factor = params['gamma_live']

                

                # Clone and Scale Magnitude

                new_load = l.copy()

                new_load['mag'] = l['mag'] * factor

                factored_loads_list.append(new_load)

        

        factored_loads_df = pd.DataFrame(factored_loads_list) if factored_loads_list else None



        # Initialize Solver with FACTORED loads

        beam_solver = solver.BeamSolver(spans, sup_df, factored_loads_df, E=params['E'], I=params['I'])

        

        # Solve

        try:

            df_results, reactions = beam_solver.solve()

            

            # --- 5. Visualization ---

            # ส่ง raw_loads ไปวาดรูป (เพื่อให้เห็นค่าจริงที่ใส่)

            # แต่กราฟผลลัพธ์ (V, M, D) จะมาจาก factored_loads

            design_view.draw_interactive_diagrams(

                df_results, 

                reactions, 

                spans, 

                sup_df, 

                raw_loads,  # <--- ส่ง Raw Load ไปแสดงผล เพื่อให้รู้ว่า input คืออะไร

                unit_force=params['u_force'], 

                unit_len=params['u_len'],

                dl_factor=params['gamma_dead'],

                ll_factor=params['gamma_live']

            )

            

            # --- 6. Result Tables ---

            design_view.render_result_tables(

                df_results, 

                reactions, 

                spans, 

                params['u_force'], 

                params['u_len']

            )

            

        except Exception as e:

            st.error(f"Analysis Failed: {str(e)}")



if __name__ == "__main__":

    main()
