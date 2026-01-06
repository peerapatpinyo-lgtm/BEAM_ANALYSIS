import streamlit as st
import pandas as pd
import numpy as np

# Import custom modules
import input_handler as ih
import solver as slv
import rc_design as rc
import design_view as view
import section_plotter as plotter

st.set_page_config(page_title="Pro-Beam RC", layout="wide")

def main():
    st.title("🏗️ Professional RC Beam Analysis & Design")
    
    # 1. Sidebar Inputs
    params = ih.render_sidebar()
    
    # 2. Geometry & Supports
    n_spans, spans, sup_df, stable = ih.render_model_inputs(params)
    
    # 3. Loads
    load_df = ih.render_loads(n_spans, spans, params, sup_df)
    
    if not stable:
        st.error("⚠️ โครงสร้างไม่เสถียร (Unstable) กรุณาตรวจสอบ Support")
        return

    if load_df is not None and st.button("🚀 คำนวณและออกแบบ", use_container_width=True):
        # --- ANALYSIS PHASE ---
        with st.spinner("กำลังวิเคราะห์ด้วย Timoshenko Beam Theory..."):
            factored_loads = load_df.copy()
            factored_loads.loc[factored_loads['case'] == 'DL', 'mag'] *= params['gamma_dead']
            factored_loads.loc[factored_loads['case'] == 'LL', 'mag'] *= params['gamma_live']
            
            beam_solver = slv.BeamSolver(
                spans=spans, supports_input=sup_df, loads_input=factored_loads,
                E=params['E'], b=params['b'], h=params['h'], I_custom=params['I']
            )
            res_df, reac_res, status = beam_solver.solve()
            
        if "error" in status:
            st.error(status["error"])
            return

        # --- DISPLAY ANALYSIS ---
        st.header("📊 ผลการวิเคราะห์แรงภายใน (Factored)")
        fig_analysis = view.plot_analysis_results(res_df, spans)
        st.plotly_chart(fig_analysis, use_container_width=True)

        # --- DESIGN PHASE ---
        st.header("🎨 การออกแบบเหล็กเสริม (ACI 318)")
        design_results = []
        cum_spans = [0] + list(np.cumsum(spans))
        
        for i in range(n_spans):
            span_res = res_df[(res_df['x'] >= cum_spans[i]) & (res_df['x'] <= cum_spans[i+1])]
            m_pos = span_res['moment'].max() / 1000
            m_neg = span_res['moment'].min() / 1000
            v_max = span_res['shear'].abs().max() / 1000
            
            design = rc.design_span_expert(
                m_pos, m_neg, v_max, params['b'], params['h'], 
                params['fc'], params['fy'], params['cover'], params['db_main']
            )
            design['db'] = params['db_main']
            design_results.append(design)

        # Summary Table
        summary = [{"Span": i+1, "Top": f"{d['neg']['n']}-DB{d['db']}", 
                    "Bottom": f"{d['pos']['n']}-DB{d['db']}", "Stirrups": d['shear_stirrups']} 
                   for i, d in enumerate(design_results)]
        st.table(pd.DataFrame(summary))

        # --- DRAWING PHASE ---
        st.header("📐 แบบรายละเอียด (Detailing)")
        col1, col2 = st.columns([2, 1])
        with col1:
            st.pyplot(plotter.plot_longitudinal_section(spans, sup_df, design_results, params['h'], params['cover']))
        with col2:
            sel_s = st.selectbox("เลือกช่วงคาน", range(1, n_spans+1))
            d_sel = design_results[sel_s-1]
            st.pyplot(plotter.plot_section(params['b'], params['h'], params['cover'], params['db_main'], 
                                         d_sel['neg']['n'], d_sel['pos']['n'], d_sel['shear_stirrups'], 
                                         params['fc'], params['fy']))

if __name__ == "__main__":
    main()
