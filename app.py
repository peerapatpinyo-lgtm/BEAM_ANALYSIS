# --- RUN ---
st.markdown("---")
if st.button("🚀 Run Analysis", type="primary", use_container_width=True):
    if len(st.session_state['supports']) < 2:
        st.error("Error: Unstable Structure. Please assign at least 2 supports.")
    else:
        g = 9.81
        calc_loads = []
        
        # --- 1. GEOMETRY SETUP ---
        spans = st.session_state['spans']
        cum_spans = [0.0] + list(np.cumsum(spans))
        total_beam_length = cum_spans[-1]
        
        # Variables for Checking
        check_applied_force_y_clipped = 0.0  # แรงที่ลงบนคานจริง (Physics Correct)
        phantom_force_y = 0.0                # แรงที่ยื่นเกินคาน (The Overhang)
        
        overhang_warnings = [] # เก็บข้อความแจ้งเตือน

        # --- 2. PROCESS LOADS & DETECT OVERHANG ---
        for i, l in enumerate(st.session_state['loads']):
            fac = dl_factor if l['case']=='DL' else ll_factor
            factored_mag = l['mag'] * fac * g
            
            # Send raw load to solver
            new_l = l.copy()
            new_l['mag'] = factored_mag
            calc_loads.append(new_l)
            
            # --- INTELLIGENT CHECK LOGIC ---
            span_idx = int(l.get('span_index', 0))
            if span_idx < len(spans):
                start_node_x = cum_spans[span_idx]
                local_x = float(l['x'])
                global_start = start_node_x + local_x
                
                # Point Load Logic
                if l['type'] == 'P':
                    if 0 <= global_start <= total_beam_length + 1e-4:
                        check_applied_force_y_clipped += factored_mag
                    elif global_start > total_beam_length:
                         phantom_force_y += factored_mag
                         overhang_warnings.append(f"• Point Load at x={global_start:.2f}m is outside beam (L={total_beam_length:.2f}m)")

                # Uniform Load Logic
                elif l['type'] == 'U':
                    dist = float(l['dist'])
                    global_end = global_start + dist
                    
                    # Calculate Intersection with Beam
                    overlap_start = max(0.0, global_start)
                    overlap_end = min(total_beam_length, global_end)
                    
                    # 1. Valid Force (On Beam)
                    if overlap_end > overlap_start:
                        effective_len = overlap_end - overlap_start
                        check_applied_force_y_clipped += factored_mag * effective_len
                    
                    # 2. Phantom Force (Off Beam) - The culprit of your error!
                    # Case A: Overhang at the end
                    if global_end > total_beam_length + 1e-4:
                        overhang_len = global_end - total_beam_length
                        force_excess = factored_mag * overhang_len
                        phantom_force_y += force_excess
                        overhang_warnings.append(f"• UDL on Span {span_idx+1} extends **{overhang_len:.4f}m** beyond beam end.")
                    
                    # Case B: Starts before beam (rare)
                    if global_start < -1e-4:
                        overhang_len = abs(global_start)
                        phantom_force_y += factored_mag * overhang_len
                        overhang_warnings.append(f"• UDL on Span {span_idx+1} starts **{overhang_len:.4f}m** before beam start.")

        # --- 3. SOLVE ---
        solver = BeamSolver(st.session_state['spans'], st.session_state['supports'], calc_loads, E, I, A, G)
        df, r, summ = solver.solve()
        
        if not df.empty:
            design_view.draw_interactive_diagrams(df, r, st.session_state['spans'], st.session_state['supports'], st.session_state['loads'], dl_factor, ll_factor)
            
            with st.expander("📊 View Critical Values, Reactions & Equilibrium Check", expanded=True):
                # ... (Keep Critical Values section same as before) ...
                st.markdown("#### 1. Critical Design Forces (Factored)")
                c1, c2, c3, c4 = st.columns(4)
                v_max_kn = summ['V_max']['value'] / 1000
                m_pos_knm = summ['M_pos']['value'] / 1000
                m_neg_knm = summ['M_neg']['value'] / 1000
                d_mm = summ['D_max']['value'] * 1000
                c1.metric("Max Shear (V_u)", f"{v_max_kn:.2f} kN", f"@ {summ['V_max']['x']:.2f} m")
                c2.metric("Max Moment (+M_u)", f"{m_pos_knm:.2f} kNm", f"@ {summ['M_pos']['x']:.2f} m")
                c3.metric("Max Moment (-M_u)", f"{m_neg_knm:.2f} kNm", f"@ {summ['M_neg']['x']:.2f} m", delta_color="inverse")
                
                limit_val = (total_beam_length / 360) * 1000
                c4.metric("Max Deflection", f"{d_mm:.4f} mm", f"{'✅' if abs(d_mm)<limit_val else '⚠️'} < L/360")

                st.markdown("---")
                
                # --- REACTIONS ---
                n_nodes = len(st.session_state['spans']) + 1
                cols = st.columns(n_nodes)
                sup_map = {int(s['id']): s['type'] for s in st.session_state['supports'] if 'id' in s}
                total_react_y = 0.0
                
                for i in range(n_nodes):
                    with cols[i]:
                        Ry = r[2*i]
                        total_react_y += Ry
                        if i in sup_map or abs(Ry) > 1.0:
                            st.markdown(f"**Node {i+1}**")
                            st.write(f"Fy: {Ry/1000:.2f} kN")

                # --- ADVANCED EQUILIBRIUM DIAGNOSTICS ---
                st.markdown("---")
                st.markdown("#### 3. Equilibrium Diagnostics")
                
                # Calculate Error
                # Error = Reaction - (Force on Beam)
                # If Error == Phantom Force, then the Solver is calculating the Phantom Force
                raw_diff = total_react_y - check_applied_force_y_clipped
                residual = raw_diff - phantom_force_y # Should be ~0 if Phantom Force explains the error
                
                col_e1, col_e2 = st.columns(2)
                col_e1.write(f"Total Applied Load (On Beam): **{check_applied_force_y_clipped/1000:.2f} kN**")
                col_e2.write(f"Total Reaction Sum: **{total_react_y/1000:.2f} kN**")
                
                if abs(raw_diff) < 10.0:
                    st.success(f"✅ Equilibrium Perfect! (Diff: {raw_diff:.2f} N)")
                elif abs(residual) < 10.0:
                    # Case: Error is exactly explained by the Overhang
                    st.warning(f"⚠️ **Warning:** Equilibrium matches if we include **{phantom_force_y:.2f} N** of load that is hanging off the beam!")
                    st.error(f"❌ **Action Required:** You have load(s) extending beyond the beam length:")
                    for msg in overhang_warnings:
                        st.markdown(msg)
                else:
                    # Case: Real math error
                    st.error(f"❌ Unknown Equilibrium Error: {raw_diff:.2f} N. (Overhang explains {phantom_force_y:.2f} N)")

                st.markdown("---")
                design_view.render_result_tables(df, r, st.session_state['spans'])
