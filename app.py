import streamlit as st
import pandas as pd
from input_handler import render_all_sidebar_inputs
from solver import BeamSolver
from design_view import plot_analysis_results
from rc_design import design_span_expert, generate_bbs, get_boq
from section_plotter import plot_section, plot_longitudinal_section_detailed
from file_manager import export_data, load_data

st.set_page_config(page_title="RC Beam Expert Pro", layout="wide")

# --- Sidebar Inputs ---
with st.sidebar:
    st.title("🏗️ RC Beam Expert")
    
    # Load/Save Project
    with st.expander("📂 Project File"):
        uploaded = st.file_uploader("Load JSON", type=["json"])
        if uploaded:
            data = load_data(uploaded)
            if data:
                # In real app, you would load these into session_state
                st.success("Loaded! (Simulated)")
    
    params, n_spans, spans, sup_df, load_df, stable = render_all_sidebar_inputs()
    
    if st.button("💾 Save Project"):
        json_str = export_data(params, spans, sup_df, load_df.to_dict('records') if not load_df.empty else [])
        st.download_button("Download .json", json_str, file_name="beam_project.json")

# --- Main App ---
if not stable:
    st.error("🚨 Structure is Unstable! Please check supports (Must have at least 1 Pin/Fixed or 2 Rollers).")
else:
    # 1. Solve
    solver = BeamSolver(spans, sup_df.to_dict('records'), load_df.to_dict('records'), 
                        params['E'], params['b'], params['h'], params['I'])
    res_df, reactions, status = solver.solve()
    
    if status.get("error"):
        st.error(f"Analysis Failed: {status['error']}")
    else:
        # Perform Equilibrium Check
        eq_check = solver.check_equilibrium(reactions)
        
        # 2. Tabs
        tab1, tab2, tab3 = st.tabs(["📊 Analysis & Checks", "🧱 RC Design & Detailing", "📋 BBS & BOQ"])
        
        # --- TAB 1: ANALYSIS ---
        with tab1:
            st.subheader("1. Static Equilibrium Check")
            
            # Display Check Metrics
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Total Load (Down)", f"{eq_check['load_down']/1000:.2f} kN")
            with c2:
                delta = abs(eq_check['diff_fy'])/1000
                color = "normal" if delta < 0.01 else "inverse"
                st.metric("Total Reaction (Up)", f"{eq_check['react_up']/1000:.2f} kN", delta_color=color)
            with c3:
                st.metric("Error (ΣFy)", f"{eq_check['diff_fy']:.4f} N")
            with c4:
                check_res = "✅ OK" if abs(eq_check['diff_fy']) < 1.0 else "❌ Warning"
                st.write(f"## {check_res}")
            
            st.divider()
            
            # Plot Diagrams
            st.subheader("2. Diagrams (FBD, SFD, BMD, Deflection)")
            fig = plot_analysis_results(res_df, spans, sup_df, load_df.to_dict('records'), reactions)
            st.plotly_chart(fig, use_container_width=True)
            
            # Show Reaction Table
            st.caption("Reaction Forces at Nodes:")
            reac_disp = {k: f"{v/1000:.2f} kN" for k,v in reactions.items()}
            st.json(reac_disp)

        # --- TAB 2: DESIGN ---
        with tab2:
            st.subheader("Reinforced Concrete Design (ACI/EIT)")
            
            design_res = []
            
            # Loop each span to design
            cum_dist = [0] + list(pd.Series(spans).cumsum())
            
            cols = st.columns(len(spans))
            for i, span_len in enumerate(spans):
                # Extract max forces for this span
                # Logic: Get simple max/min in span range
                x_start, x_end = cum_dist[i], cum_dist[i+1]
                mask = (res_df['x'] >= x_start) & (res_df['x'] <= x_end)
                span_data = res_df[mask]
                
                m_max = span_data['moment'].max() / 1000.0
                m_min = span_data['moment'].min() / 1000.0
                v_max = span_data['shear'].abs().max() / 1000.0
                
                # Design
                res = design_span_expert(m_max, m_min, v_max, 
                                       params['b'], params['h'], params['fc'], params['fy'], 
                                       40, 16) # Cover 40mm, DB16 Main
                design_res.append(res)
                
                # Display Card
                with cols[i]:
                    st.info(f"**Span {i+1}** (L={span_len}m)")
                    st.write(f"**M+**: {m_max:.1f} kNm → {res['pos']['n']}-DB16")
                    st.write(f"**M-**: {m_min:.1f} kNm → {res['neg']['n']}-DB16")
                    st.write(f"**V**: {v_max:.1f} kN → {res['shear_stirrups']}")
                    
                    # Section Plot
                    fig_sec = plot_section(params['b'], params['h'], 40, 16, 
                                           res['neg']['n'], res['pos']['n'], res['shear_stirrups'],
                                           params['fc'], params['fy'])
                    st.pyplot(fig_sec)

            st.divider()
            st.subheader("Longitudinal Detailing")
            fig_long = plot_longitudinal_section_detailed(spans, sup_df, design_res, params['h'])
            st.pyplot(fig_long)

        # --- TAB 3: BOQ ---
        with tab3:
            st.subheader("Bill of Quantities & Bar Bending")
            bbs = generate_bbs(design_res, spans, params['b'], params['h'], 40)
            df_bbs = pd.DataFrame(bbs)
            st.dataframe(df_bbs, use_container_width=True)
            
            vol, w_steel = get_boq(spans, params['b'], params['h'], bbs)
            c1, c2 = st.columns(2)
            c1.metric("Concrete Volume", f"{vol:.2f} m³")
            c2.metric("Total Steel Weight", f"{w_steel:.2f} kg")
