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
    
    with st.expander("📂 Project File"):
        uploaded = st.file_uploader("Load JSON", type=["json"])
        if uploaded:
            data = load_data(uploaded)
            if data: st.success("Loaded!")
    
    params, n_spans, spans, sup_df, load_df, stable = render_all_sidebar_inputs()
    
    if st.button("💾 Save Project"):
        json_str = export_data(params, spans, sup_df, load_df.to_dict('records') if not load_df.empty else [])
        st.download_button("Download .json", json_str, file_name="beam_project.json")

# --- Main App ---
if not stable:
    st.error("🚨 Structure is Unstable! Please add supports (Need Pin/Fixed or 2+ Rollers).")
else:
    # 1. Solve
    solver = BeamSolver(spans, sup_df.to_dict('records'), load_df.to_dict('records'), 
                        params['E'], params['b'], params['h'], params['I'])
    res_df, reactions, status = solver.solve()
    
    if status.get("error"):
        st.error(f"Analysis Failed: {status['error']}")
    else:
        # Check Equilibrium
        eq_check = solver.check_equilibrium(reactions)
        w_sw = params['b'] * params['h'] * 24.0 # For display only
        
        # 2. Tabs
        tab1, tab2, tab3 = st.tabs(["📊 Analysis & Checks", "🧱 RC Design & Detailing", "📋 BBS & BOQ"])
        
        # --- TAB 1: ANALYSIS ---
        with tab1:
            st.info(f"ℹ️ **Note:** Self-weight ({w_sw:.2f} kN/m) is included automatically.")
            
            # 1. Equilibrium Check
            st.subheader("1. Static Equilibrium Check")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Total Load (Down)", f"{eq_check['load_down']/1000:.2f} kN")
            with c2:
                st.metric("Total Reaction (Up)", f"{eq_check['react_up']/1000:.2f} kN")
            with c3:
                st.metric("Diff", f"{eq_check['diff_fy']:.4f} N")
            with c4:
                status_text = "✅ OK" if abs(eq_check['diff_fy']) < 1.0 else "❌ Unbalanced"
                st.write(f"## {status_text}")
            
            st.divider()
            
            # 2. Diagrams
            st.subheader("2. Structural Diagrams")
            fig = plot_analysis_results(res_df, spans, sup_df, load_df.to_dict('records'), reactions)
            st.plotly_chart(fig, use_container_width=True)

        # --- TAB 2: RC DESIGN (Full Code Restored) ---
        with tab2:
            st.subheader("Reinforced Concrete Design (ACI 318 / EIT)")
            
            design_res = []
            cum_dist = [0] + list(pd.Series(spans).cumsum())
            
            # Design Loop for Each Span
            cols = st.columns(len(spans))
            for i, span_len in enumerate(spans):
                # Filter results for this span
                x_start, x_end = cum_dist[i], cum_dist[i+1]
                mask = (res_df['x'] >= x_start) & (res_df['x'] <= x_end)
                span_data = res_df[mask]
                
                # Get Critical Forces (Convert to kN, kNm)
                # Max Positive Moment (Sagging)
                m_max_pos = span_data['moment'].max() / 1000.0
                # Max Negative Moment (Hogging - usually at supports, but we check span range)
                m_max_neg = span_data['moment'].min() / 1000.0
                # Max Shear (Abs)
                v_max = span_data['shear'].abs().max() / 1000.0
                
                # Default inputs for design
                cover = 40 # mm
                main_bar_dia = 16 # mm
                
                # Call Expert Design Function
                res = design_span_expert(m_max_pos, m_max_neg, v_max, 
                                       params['b'], params['h'], params['fc'], params['fy'], 
                                       cover, main_bar_dia)
                design_res.append(res)
                
                # Display Results in Column
                with cols[i]:
                    st.success(f"**Span {i+1}** (L={span_len}m)")
                    
                    st.write("--- Forces ---")
                    st.write(f"M+ : {m_max_pos:.2f} kNm")
                    st.write(f"M- : {m_max_neg:.2f} kNm")
                    st.write(f"Vmax : {v_max:.2f} kN")
                    
                    st.write("--- Rebar ---")
                    st.write(f"**Bot**: {res['pos']['n']} - DB{main_bar_dia}")
                    st.write(f"**Top**: {res['neg']['n']} - DB{main_bar_dia}")
                    st.write(f"**Stirrup**: {res['shear_stirrups']}")
                    
                    # Plot Cross Section
                    fig_sec = plot_section(params['b'], params['h'], cover, main_bar_dia, 
                                           res['neg']['n'], res['pos']['n'], res['shear_stirrups'],
                                           params['fc'], params['fy'])
                    st.pyplot(fig_sec)

            st.divider()
            
            # Longitudinal Profile Plot
            st.subheader("Longitudinal Reinforcement Detail")
            fig_long = plot_longitudinal_section_detailed(spans, sup_df, design_res, params['h'])
            st.pyplot(fig_long)

        # --- TAB 3: BOQ ---
        with tab3:
            st.subheader("Bill of Quantities & Bar Bending Schedule")
            
            # Generate BBS
            bbs = generate_bbs(design_res, spans, params['b'], params['h'], 40)
            df_bbs = pd.DataFrame(bbs)
            st.dataframe(df_bbs, use_container_width=True)
            
            # Calculate Totals
            vol, w_steel = get_boq(spans, params['b'], params['h'], bbs)
            
            st.write("### Summary")
            c1, c2 = st.columns(2)
            c1.metric("Concrete Volume (m³)", f"{vol:.2f}")
            c2.metric("Total Steel Weight (kg)", f"{w_steel:.2f}")
