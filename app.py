import streamlit as st
import pandas as pd
from input_handler import render_all_sidebar_inputs
from solver import BeamSolver
from design_view import plot_analysis_results
from rc_design import design_span_expert, generate_bbs, get_boq
from section_plotter import plot_section, plot_longitudinal_section_detailed
from file_manager import export_data, load_data

st.set_page_config(page_title="RC Beam Expert Pro", layout="wide")

# Sidebar
with st.sidebar:
    st.title("🏗️ RC Beam Expert")
    params, n_spans, spans, sup_df, load_df, stable = render_all_sidebar_inputs()

# Main
if not stable:
    st.error("🚨 Structure Unstable! Add supports.")
else:
    # 1. Solve
    solver = BeamSolver(spans, sup_df.to_dict('records'), load_df.to_dict('records'), 
                        params['E'], params['b'], params['h'], params['I'])
    res_df, reactions, status = solver.solve()
    
    if status.get("error"):
        st.error(f"Error: {status['error']}")
    else:
        eq_check = solver.check_equilibrium(reactions)
        
        # Tabs
        tab1, tab2, tab3 = st.tabs(["📊 Analysis", "🧱 Design", "📋 BOQ"])
        
        with tab1:
            # Equilibrium Check
            st.subheader("1. System Equilibrium (Inc. Self-weight)")
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Load (Down)", f"{eq_check['load_down']/1000:.2f} kN")
            c2.metric("Total Reaction (Up)", f"{eq_check['react_up']/1000:.2f} kN")
            is_ok = abs(eq_check['diff_fy']) < 1.0
            c3.markdown(f"Status: **{'✅ Balanced' if is_ok else '❌ Error'}**")
            
            # Diagrams
            st.divider()
            fig = plot_analysis_results(res_df, spans, sup_df, load_df.to_dict('records'), reactions)
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.subheader("Reinforced Concrete Design")
            # Design Logic Loop ... (เหมือนเดิม)
            design_res = []
            cum_dist = [0] + list(pd.Series(spans).cumsum())
            
            cols = st.columns(len(spans))
            for i, span_len in enumerate(spans):
                x_start, x_end = cum_dist[i], cum_dist[i+1]
                mask = (res_df['x'] >= x_start) & (res_df['x'] <= x_end)
                span_data = res_df[mask]
                
                # Get Abs Max Moment for design
                m_max_pos = span_data['moment'].max() / 1000.0
                m_max_neg = span_data['moment'].min() / 1000.0
                v_max = span_data['shear'].abs().max() / 1000.0
                
                res = design_span_expert(m_max_pos, m_max_neg, v_max, 
                                       params['b'], params['h'], params['fc'], params['fy'], 
                                       40, 16)
                design_res.append(res)
                
                with cols[i]:
                    st.info(f"Span {i+1}")
                    st.write(f"M+ {m_max_pos:.1f} → {res['pos']['n']} DB16")
                    st.write(f"M- {m_max_neg:.1f} → {res['neg']['n']} DB16")
                    st.pyplot(plot_section(params['b'], params['h'], 40, 16, res['neg']['n'], res['pos']['n'], res['shear_stirrups'], params['fc'], params['fy']))

            st.pyplot(plot_longitudinal_section_detailed(spans, sup_df, design_res, params['h']))

        with tab3:
            bbs = generate_bbs(design_res, spans, params['b'], params['h'], 40)
            st.dataframe(pd.DataFrame(bbs))
