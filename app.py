# ------------------------------------------
# PART 2: ANALYSIS
# ------------------------------------------
st.markdown('<div class="section-header">2️⃣ Structural Analysis (วิเคราะห์โครงสร้าง)</div>', unsafe_allow_html=True)

if st.button("🚀 Calculate Analysis", type="primary", use_container_width=True):
    # 1. Draw Beam Diagram
    fig_beam = draw_beam_diagram(spans, supports, loads_input, U_L, U_F)
    # 2. Solve
    solver = SimpleBeamSolver(spans, supports, loads_input)
    u, err = solver.solve() 
    
    if err:
        st.error(err)
    else:
        # 3. Get Results (Raw Data)
        df = solver.get_internal_forces(100)
        # เก็บแค่ Raw Data ลง Session State (ยังไม่แปลงหน่วย)
        st.session_state['res_df'] = df
        st.session_state['analyzed'] = True
        st.rerun()

# Show Graphs if analyzed
if st.session_state.get('analyzed', False) and st.session_state.get('res_df') is not None:
    # ดึงค่าจาก State มาคำนวณหน่วยปัจจุบัน (แก้บั๊ก KeyError และทำให้เปลี่ยนหน่วยได้ Realtime)
    df = st.session_state['res_df'].copy()
    
    # สร้างคอลัมน์สำหรับแสดงผลใหม่ทุกครั้ง
    df['V_show'] = df['shear'] * FROM_N
    df['M_show'] = df['moment'] * FROM_N
    
    col_res1, col_res2 = st.columns(2)
    max_M = df['M_show'].abs().max()
    max_V = df['V_show'].abs().max()
    
    col_res1.metric(f"Max Moment (Mu)", f"{max_M:.2f} {U_M}")
    col_res2.metric(f"Max Shear (Vu)", f"{max_V:.2f} {U_F}")

    # Plot SFD & BMD
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, 
                        subplot_titles=(f"Shear Diagram ({U_F})", f"Moment Diagram ({U_M})"))
    
    fig.add_trace(go.Scatter(x=df['x'], y=df['V_show'], fill='tozeroy', line=dict(color='#E53935'), name="V"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['x'], y=df['M_show'], fill='tozeroy', line=dict(color='#1E88E5'), name="M"), row=2, col=1)
    
    fig.update_layout(height=500, showlegend=False, hovermode="x unified")
    fig.update_yaxes(autorange="reversed", row=2, col=1) # Flip Moment
    st.plotly_chart(fig, use_container_width=True)

    # ------------------------------------------
    # PART 3: DESIGN (แยกส่วนชัดเจน)
    # ------------------------------------------
    st.markdown("---")
    st.markdown('<div class="section-header">3️⃣ Part Design (ออกแบบหน้าตัดคอนกรีตเสริมเหล็ก)</div>', unsafe_allow_html=True)
    
    # Container for Design
    with st.container():
        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        
        # Design Inputs
        c_d1, c_d2, c_d3 = st.columns([1, 1, 1.5])
        
        with c_d1:
            st.markdown("##### 🧱 Material")
            fc = st.number_input("f'c (ksc)", value=240.0)
            fy = st.number_input("fy (ksc)", value=4000.0)
            
        with c_d2:
            st.markdown("##### 📏 Section Size")
            b = st.number_input("Width b (cm)", value=25.0)
            h = st.number_input("Depth h (cm)", value=50.0)
            cov = st.number_input("Cover (cm)", value=3.0)
            
        with c_d3:
            st.markdown("##### ⛓️ Rebar Spec")
            main_bar = st.selectbox("Main Bar", ["DB12", "DB16", "DB20", "DB25"], index=2)
            stirrup = st.selectbox("Stirrup", ["RB6", "RB9", "DB10"], index=1)
            
            bar_areas = {'DB12':1.13, 'DB16':2.01, 'DB20':3.14, 'DB25':4.91}
            bar_dias = {'DB12':12, 'DB16':16, 'DB20':20, 'DB25':25}
            stir_areas = {'RB6':0.28, 'RB9':0.64, 'DB10':0.78}
            stir_dias = {'RB6':6, 'RB9':9, 'DB10':10}

        st.markdown('</div>', unsafe_allow_html=True)

        # --- CALCULATION ---
        # ใช้ค่าจาก df ที่แปลงหน่วยแล้ว
        Mu_val = df['M_show'].abs().max()
        Vu_val = df['V_show'].abs().max()
        
        # Convert all to kg, cm for calc internally
        if "kN" in unit_sys:
            Mu_calc = Mu_val * 101.97 * 100 # kN-m -> kg-cm
            Vu_calc = Vu_val * 101.97       # kN -> kg
        else:
            Mu_calc = Mu_val * 100          # kg-m -> kg-cm
            Vu_calc = Vu_val
        
        d = h - cov - 0.9 # est d
        
        # Flexure
        if method == "SDM":
            phi = 0.9
            Rn = Mu_calc / (phi * b * d**2)
            # Check complex number domain error
            check_val = 1 - (2*Rn)/(0.85*fc)
            if check_val < 0:
                rho = 0.02 # Fallback/Fail
                st.error("❌ Section too small for Moment!")
            else:
                rho = (0.85*fc/fy) * (1 - np.sqrt(check_val))
            
            As_req = max(rho, 14/fy) * b * d
        else:
            # WSD simplified
            As_req = Mu_calc / (0.875 * 0.5 * fy * d)
            
        nb = int(np.ceil(As_req / bar_areas[main_bar]))
        nb = max(2, nb)
        
        # Shear
        vc = 0.53 * np.sqrt(fc) * b * d
        if method == "SDM": vc = 0.85 * vc
            
        if Vu_calc > vc:
            vs_req = Vu_calc - vc
            av = 2 * stir_areas[stirrup]
            # Avoid divide by zero
            if vs_req <= 0: s_req = 30
            else: s_req = (av * fy * d) / vs_req
            
            s = int(5 * round(min(s_req, d/2, 30)/5))
            if s == 0: s = 5
            shear_res = f"@{s} cm"
            shear_status = "⚠️ Shear Reinf. Req"
        else:
            s = int(d/2)
            shear_res = f"@{s} cm (Min)"
            shear_status = "✅ Conc. OK"

        # --- OUTPUT DISPLAY ---
        st.markdown('<div class="design-box">', unsafe_allow_html=True)
        col_out1, col_out2 = st.columns([1, 1])
        
        with col_out1:
            st.markdown("#### 🎯 Design Result")
            st.write(f"**Design Moment (Mu):** {Mu_val:.2f} {U_M}")
            st.write(f"**Main Steel:** {nb}-{main_bar} (As = {nb*bar_areas[main_bar]:.2f} cm²)")
            st.write(f"**Stirrups:** {stirrup} {shear_res}")
            st.caption(f"Shear Status: {shear_status}")
            
        with col_out2:
            st.markdown("#### 📐 Section View")
            fig_sec = draw_section_view(b, h, cov, nb, bar_dias[main_bar], stir_dias[stirrup])
            st.plotly_chart(fig_sec, use_container_width=True, config={'displayModeBar': False})
            
        st.markdown('</div>', unsafe_allow_html=True)

else:
    st.info("👈 กรุณากดปุ่ม 'Calculate Analysis' ด้านบน เพื่อเริ่มคำนวณ")
