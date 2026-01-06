import streamlit as st
import pandas as pd
import numpy as np

# Import custom modules
import input_handler as ih
import solver as slv
import rc_design as rc
import design_view as view
import section_plotter as plotter

st.set_page_config(page_title="Pro-Beam RC Designer", layout="wide")

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

    if load_df is not None and st.button("🚀 Run Analysis & Design", use_container_width=True):
        
        # --- PHASE 1: ANALYSIS ---
        with st.spinner("Analyzing structure using Timoshenko Beam Theory..."):
            factored_loads = load_df.copy()
            # Apply Load Factors
            factored_loads.loc[factored_loads['case'] == 'DL', 'mag'] *= params['gamma_dead']
            factored_loads.loc[factored_loads['case'] == 'LL', 'mag'] *= params['gamma_live']
            
            beam_solver = slv.BeamSolver(
                spans=spans,
                supports_input=sup_df,
                loads_input=factored_loads,
                E=params['E'],
                b=params['b'],
                h=params['h'],
                I_custom=params['I']
            )
            
            res_df, reac_res, status = beam_solver.solve()
            
        if "error" in status:
            st.error(f"Solver Error: {status['error']}")
            return

        # --- PHASE 2: DISPLAY RESULTS ---
        st.header("3. Analysis Results (Factored)")
        tab1, tab2 = st.tabs(["📈 Diagrams", "⚓ Reactions"])
        
        with tab1:
            fig_analysis = view.plot_analysis_results(res_df, spans)
            st.plotly_chart(fig_analysis, use_container_width=True)
            
        with tab2:
            reac_disp = [{"Node": k, "Reaction (kN)": v/1000} for k, v in reac_res.items()]
            st.table(pd.DataFrame(reac_disp))

        # --- PHASE 3: DESIGN ---
        st.header("4. Reinforcement Design (ACI 318)")
        design_results = []
        cum_spans = [0] + list(np.cumsum(spans))
        
        for i in range(n_spans):
            # Filter results for this span
            span_mask = (res_df['x'] >= cum_spans[i]) & (res_df['x'] <= cum_spans[i+1])
            span_res = res_df[span_mask]
            
            if span_res.empty: continue
            
            m_pos = span_res['moment'].max() / 1000 # kNm
            m_neg = span_res['moment'].min() / 1000 # kNm
            v_max = span_res['shear'].abs().max() / 1000 # kN
            
            design = rc.design_span_expert(
                m_pos=m_pos, m_neg=m_neg, v_u=v_max,
                b=params['b'], h=params['h'],
                fc=params['fc'], fy=params['fy'],
                cover=params['cover'], db=params['db_main']
            )
            design['db'] = params['db_main']
            design_results.append(design)

        # Summary Table
        summary_data = []
        for i, d in enumerate(design_results):
            summary_data.append({
                "Span": i+1,
                "Top Bars": f"{d['neg']['n']}-DB{d['db']}",
                "Bot Bars": f"{d['pos']['n']}-DB{d['db']}",
                "Stirrups": d['shear_stirrups'],
                "Capacity +": f"{d['pos']['capacity']:.1f} kNm",
                "Status": "✅ Pass" if d['shear_status'] != "Fail" else "❌ Fail"
            })
        st.dataframe(pd.DataFrame(summary_data), use_container_width=True)

        # --- PHASE 4: DRAWINGS ---
        st.header("5. Detailing Drawings")
        
        col_long, col_sec = st.columns([2, 1])
        with col_long:
            st.subheader("Longitudinal Section")
            fig_long = plotter.plot_longitudinal_section(spans, sup_df, design_results, params['h'], params['cover'])
            st.pyplot(fig_long)
            
        with col_sec:
            st.subheader("Cross Section")
            sel_span = st.selectbox("Select Span", range(1, n_spans+1))
            idx = sel_span - 1
            d = design_results[idx]
            
            fig_sec = plotter.plot_section(
                b=params['b'], h=params['h'], cover_mm=params['cover'], db_mm=d['db'],
                n_top=d['neg']['n'], n_bot=d['pos']['n'],
                stirrup_info=d['shear_stirrups'],
                fc=params['fc'], fy=params['fy']
            )
            st.pyplot(fig_sec)

if __name__ == "__main__":
    main()
