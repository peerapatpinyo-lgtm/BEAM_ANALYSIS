# design_view.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from rc_design_engine import get_phi_Mn_details, check_shear_details, calculate_layer_properties
from section_plotter import plot_longitudinal_section_detailed, plot_cross_section

def plot_analysis_results(res_df, spans, supports, loads, reactions):
    """
    สร้างกราฟวิเคราะห์โครงสร้าง (Textbook-style)
    หน่วยแสดงผล: Force (kN), Moment (kN-m), Deflection (mm)
    """
    
    # --- Create Subplots ---
    fig = make_subplots(
        rows=4, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.08,
        subplot_titles=(
            "<b>1. Free Body Diagram (FBD) - [Units: kN, kN/m]</b>", 
            "<b>2. Shear Force Diagram (SFD) - [Unit: kN]</b>", 
            "<b>3. Bending Moment Diagram (BMD) - [Unit: kN-m]</b>",
            "<b>4. Elastic Curve (Deflection) - [Unit: mm]</b>"
        ),
        row_heights=[0.20, 0.25, 0.25, 0.30]
    )

    # ==========================================
    # ROW 1: LOAD MODEL (FBD)
    # ==========================================
    total_L = sum(spans)
    cum_dist = [0] + list(np.cumsum(spans))
    
    # Beam Line
    fig.add_trace(go.Scatter(
        x=[0, total_L], y=[0, 0], 
        mode='lines', line=dict(color='black', width=4), hoverinfo='skip'
    ), row=1, col=1)
    
    # Supports
    for idx, row in supports.iterrows():
        sym = "triangle-up"
        if row['type'] == 'Fixed': sym = "square"
        elif row['type'] == 'Roller': sym = "circle"
        
        fig.add_trace(go.Scatter(
            x=[row['x']], y=[-0.08], 
            mode='markers+text',
            marker=dict(symbol=sym, size=14, color='white', line=dict(width=2, color='black')),
            text=[row['type'][0]], textposition="bottom center",
            hoverinfo='name', name=f"Support"
        ), row=1, col=1)

    # Loads
    if isinstance(loads, pd.DataFrame):
        load_iter = loads.to_dict('records')
    else:
        load_iter = loads

    for l in load_iter:
        span_idx = int(l['span_index'])
        start_x = cum_dist[span_idx]
        mag_kN = l['mag'] / 1000.0 # แปลงหน่วย N เป็น kN
        
        if l['type'] == 'P':
            x_loc = start_x + float(l['d_start']) 
            fig.add_annotation(
                x=x_loc, y=0, ax=0, ay=-50,
                xref="x1", yref="y1",
                showarrow=True, arrowhead=2, arrowsize=1.2, arrowwidth=2, arrowcolor="#c0392b",
                text=f"<b>P={mag_kN:.2f} kN</b>", yshift=5, row=1, col=1
            )
        elif l['type'] == 'U':
            x_s = start_x + float(l.get('d_start', 0))
            x_e = x_s + float(l['dist'])
            h_vis = 0.25
            fig.add_trace(go.Scatter(
                x=[x_s, x_e], y=[h_vis, h_vis],
                mode='lines', line=dict(color='#2980b9', width=2), hoverinfo='skip'
            ), row=1, col=1)
            
            n_arrows = max(3, int(float(l['dist']) * 3)) 
            arrow_x = np.linspace(x_s, x_e, n_arrows)
            for ax_x in arrow_x:
                fig.add_annotation(
                    x=ax_x, y=0, ax=0, ay=-30,
                    xref="x1", yref="y1",
                    showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1, arrowcolor="#2980b9",
                    row=1, col=1
                )
            fig.add_annotation(
                x=(x_s+x_e)/2, y=h_vis,
                text=f"<b>w={mag_kN:.2f} kN/m</b>",
                showarrow=False, yshift=10, font=dict(color="#2980b9"), row=1, col=1
            )

    # ==========================================
    # ROW 2: SHEAR FORCE (SFD)
    # ==========================================
    fig.add_hline(y=0, line_color="black", line_width=1, row=2, col=1)
    fig.add_trace(go.Scatter(
        x=res_df['x'], y=res_df['shear']/1000, 
        mode='lines', name='Shear (kN)', line=dict(color='#e74c3c', width=2),
        fill='tozeroy', fillcolor='rgba(231, 76, 60, 0.1)'
    ), row=2, col=1)
    
    v_max = res_df['shear'].max() / 1000
    v_min = res_df['shear'].min() / 1000
    for val in [v_max, v_min]:
        if abs(val) > 0.01:
            idx = (res_df['shear']/1000 - val).abs().idxmin()
            fig.add_annotation(
                x=res_df['x'].iloc[idx], y=val,
                text=f"<b>{val:.2f} kN</b>", showarrow=False, yshift=15 if val>0 else -15,
                font=dict(color='#e74c3c', size=11), row=2, col=1
            )

    # ==========================================
    # ROW 3: BENDING MOMENT (BMD)
    # ==========================================
    fig.add_hline(y=0, line_color="black", line_width=1, row=3, col=1)
    fig.add_trace(go.Scatter(
        x=res_df['x'], y=res_df['moment']/1000, 
        mode='lines', name='Moment (kN-m)', line=dict(color='#27ae60', width=2),
        fill='tozeroy', fillcolor='rgba(39, 174, 96, 0.1)'
    ), row=3, col=1)

    m_max = res_df['moment'].max() / 1000
    m_min = res_df['moment'].min() / 1000
    for val in [m_max, m_min]:
        if abs(val) > 0.01:
            idx = (res_df['moment']/1000 - val).abs().idxmin()
            fig.add_annotation(
                x=res_df['x'].iloc[idx], y=val,
                text=f"<b>{val:.2f} kN-m</b>", 
                showarrow=True, arrowhead=1, ay=30 if val>0 else -30,
                font=dict(color='#27ae60', size=11), row=3, col=1
            )

    # ==========================================
    # ROW 4: DEFLECTION (Elastic Curve)
    # ==========================================
    fig.add_hline(y=0, line_color="black", line_width=1, row=4, col=1)
    fig.add_trace(go.Scatter(
        x=res_df['x'], y=res_df['deflection'], 
        mode='lines', name='Deflection (mm)', line=dict(color='#8e44ad', width=2)
    ), row=4, col=1)
    
    idx_max_def = res_df['deflection'].abs().idxmax()
    max_def_val = res_df['deflection'].iloc[idx_max_def]
    
    fig.add_annotation(
        x=res_df['x'].iloc[idx_max_def], y=max_def_val,
        text=f"<b>Max δ: {max_def_val:.3f} mm</b>",
        showarrow=True, arrowhead=1, 
        ay=40 if max_def_val < 0 else -40,
        font=dict(color='#8e44ad', size=11), row=4, col=1
    )

    # ==========================================
    # LAYOUT & STYLING
    # ==========================================
    for x_pos in cum_dist:
        fig.add_vline(x=x_pos, line_width=1, line_dash="dash", line_color="gray", opacity=0.3)

    fig.update_layout(
        title="<b>Structural Analysis Results (Design Forces)</b>",
        height=950, showlegend=False, template="plotly_white", hovermode="x unified",
        margin=dict(t=80, b=60, l=60, r=20)
    )
    
    fig.update_yaxes(visible=False, range=[-0.5, 0.8], row=1, col=1)
    fig.update_yaxes(title_text="Shear, V (kN)", showgrid=True, row=2, col=1)
    fig.update_yaxes(title_text="Moment, M (kN-m)", autorange="reversed", showgrid=True, row=3, col=1)
    fig.update_yaxes(title_text="Deflection, δ (mm)", showgrid=True, zeroline=True, row=4, col=1)
    fig.update_xaxes(title_text="Beam Length, x (m)", row=4, col=1)

    return fig

def display_design_comparison(mu_pos, mu_neg, vu, design_res):
    st.markdown("---")
    st.subheader("🛠 RC Design Verification")
    
    # ดึงค่าพารามิเตอร์พื้นฐาน (ถ้าไม่มีให้ default ไว้ก่อนเพื่อป้องกัน error)
    fc = design_res.get('fc', 24) # MPa
    fy = design_res.get('fy', 400) # MPa
    b = design_res.get('b', 200)   # mm
    h = design_res.get('h', 400)   # mm
    
    # Use d from bottom steel for checking min reinforcement roughly
    # Or just use h-50 as standard approx if d not explicitly calculated yet
    # But design_res usually has d or we approximate it.
    d = h - 50 
    
    # คำนวณ As_min ตามมาตรฐาน
    as_min = max((0.25 * np.sqrt(fc) / fy) * b * d, (1.4 / fy) * b * d)

    st.markdown("#### 📏 Reinforcement Area Check ($A_s$)")
    as_col1, as_col2 = st.columns(2)
    
    with as_col1:
        # ใช้ค่าที่มากระหว่าง As_req (จากแรง) กับ As_min (ตามมาตรฐาน)
        # Note: In new logic, Ast is derived from layers directly in capacity check.
        # We might not have reverse calc As_req stored unless we do it.
        # But for comparison, let's use the provided vs min.
        
        as_prov = design_res.get('as_prov_bot', 0.0)
        # As_req isn't strictly calculated in capacity-check workflow, but we can display Min vs Prov
        # Or if we want As_req, we need to reverse calc. 
        # Let's display Min vs Provided which is the code check requirement.
        
        st.write("**Bottom Steel (Mid-span)**")
        st.write(f"Min Required: `{as_min:.0f}` $mm^2$ | Provided: `{as_prov:.0f}` $mm^2$")
        
        if as_min > 0:
            ratio = min(as_prov / as_min, 1.0)
            st.progress(ratio)
            if as_prov >= as_min:
                st.success(f"✅ Area > Min")
            else:
                st.error(f"❌ Area < Min")

    with as_col2:
        as_prov_t = design_res.get('as_prov_top', 0.0)
        
        st.write("**Top Steel (Support)**")
        st.write(f"Min Required: `{as_min:.0f}` $mm^2$ | Provided: `{as_prov_t:.0f}` $mm^2$")
        
        if as_min > 0:
            ratio_t = min(as_prov_t / as_min, 1.0)
            st.progress(ratio_t)
            if as_prov_t >= as_min:
                st.success(f"✅ Area > Min")
            else:
                st.error(f"❌ Area < Min")

    # --- ส่วนที่ 2: Strength Check (Mu, Vu) ---
    st.markdown("---")
    st.markdown("#### ⚡ Section Strength ($\phi M_n, \phi V_n$)")

    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Positive Moment (+M)**")
        phi_mn_pos = design_res.get('phi_Mn_pos', 0.0)
        st.metric("Demand $M_u^+$", f"{mu_pos:.2f} kNm")
        st.metric("Capacity $\phi M_n^+$", f"{phi_mn_pos:.2f} kNm", 
                  delta=f"{(phi_mn_pos - mu_pos):.2f}", delta_color="normal")
        st.success("✅ PASS") if phi_mn_pos >= mu_pos else st.error("❌ FAIL")

    with col2:
        st.markdown("**Negative Moment (-M)**")
        phi_mn_neg = design_res.get('phi_Mn_neg', 0.0)
        mu_neg_abs = abs(mu_neg)
        st.metric("Demand $M_u^-$", f"{mu_neg_abs:.2f} kNm")
        st.metric("Capacity $\phi M_n^-$", f"{phi_mn_neg:.2f} kNm",
                  delta=f"{(phi_mn_neg - mu_neg_abs):.2f}", delta_color="normal")
        st.success("✅ PASS") if phi_mn_neg >= mu_neg_abs else st.error("❌ FAIL")

    with col3:
        st.markdown("**Shear Force (V)**")
        phi_vn = design_res.get('phi_Vn', 0.0)
        st.metric("Demand $V_u$", f"{vu:.2f} kN")
        st.metric("Capacity $\phi V_n$", f"{phi_vn:.2f} kN",
                  delta=f"{(phi_vn - vu):.2f}", delta_color="normal")
        st.success("✅ PASS") if phi_vn >= vu else st.error("❌ FAIL")
            
    # Helper to format layers string
    def fmt_layers(layers):
        if not layers: return "None"
        parts = []
        for i, l in enumerate(layers):
            parts.append(f"L{i+1}:{l['n']}DB{l['db']}")
        return ", ".join(parts)

    st.info(f"💡 **Final Detailing:** Top [{fmt_layers(design_res.get('top_layers'))}] | "
            f"Bottom [{fmt_layers(design_res.get('bot_layers'))}] | "
            f"Stirrup RB{design_res.get('stir_db')}@{design_res.get('shear',{}).get('s')} mm")

def render_design_view(solver_results, params, spans, sup_df, loads_df):
    """
    Main Entry Point for Tab 2
    Integrates Analysis Plots and Interactive Multi-Layer Design
    """
    x_vals, m_vals, v_vals, d_vals, reactions = solver_results
    
    # 1. Create DataFrame for Plotting
    res_df = pd.DataFrame({
        'x': x_vals,
        'moment': m_vals,
        'shear': v_vals,
        'deflection': d_vals * 1000 # m to mm
    })
    
    st.markdown("### 📊 Structural Analysis & Design")
    
    # --- Show Main Analysis Plots ---
    with st.expander("📈 View Full Analysis Diagrams (SFD, BMD, Deflection)", expanded=True):
        fig = plot_analysis_results(res_df, spans, sup_df, loads_df, reactions)
        st.plotly_chart(fig, use_container_width=True)

    # --- Interactive Design Section ---
    st.markdown("---")
    st.markdown("### 🏗️ Member Design (Multi-Layer Supported)")

    # 1. Select Span
    col_sel, col_param = st.columns([1, 2])
    with col_sel:
        st.info("👇 **Select Beam Span**")
        n_spans = len(spans)
        span_options = [f"Span {i+1} (L={spans[i]:.2f}m)" for i in range(n_spans)]
        selected_span_label = st.radio("Select Span:", span_options)
        selected_span_idx = span_options.index(selected_span_label)
        
        # Get Span Forces
        L_span = spans[selected_span_idx]
        x_start = sum(spans[:selected_span_idx])
        x_end = x_start + L_span
        
        mask = (x_vals >= x_start) & (x_vals <= x_end)
        span_m = m_vals[mask]
        span_v = v_vals[mask]
        
        mu_pos = np.max(span_m)/1000 if len(span_m) > 0 else 0
        mu_pos = max(0, mu_pos)
        
        # Approximate Negative Moment (Left/Right of span)
        # Find M at x_start and x_end
        idx_start = (np.abs(x_vals - x_start)).argmin()
        idx_end = (np.abs(x_vals - x_end)).argmin()
        m_left = m_vals[idx_start]/1000
        m_right = m_vals[idx_end]/1000
        mu_neg = min(m_left, m_right) # Usually negative, take the most negative
        
        vu_max = np.max(np.abs(span_v))/1000 if len(span_v) > 0 else 0

    with col_param:
        st.info("⚙️ **Material & Section**")
        c1, c2, c3, c4 = st.columns(4)
        fc = c1.number_input("f'c (MPa)", value=28.0, step=1.0)
        fy = c2.number_input("fy (MPa)", value=420.0, step=10.0)
        b = c3.number_input("b (mm)", value=float(params['b']*1000), step=50.0)
        h = c4.number_input("h (mm)", value=float(params['h']*1000), step=50.0)
        cover = st.number_input("Cover (mm)", value=25.0)

    # 2. Reinforcement Inputs (Multi-layer)
    st.markdown("---")
    st.subheader("✏️ Reinforcement Detailing")
    
    col_input, col_view = st.columns([1, 1])
    
    with col_input:
        # --- Top Steel ---
        st.markdown("**Top Reinforcement (Negative Moment)**")
        n_top_layers = st.selectbox("Top Layers", [1, 2, 3], index=0, key="top_L_cnt")
        top_layers = []
        for i in range(n_top_layers):
            c_t1, c_t2 = st.columns(2)
            nt = c_t1.number_input(f"Top L{i+1} Bars", 2, 10, 2, key=f"nt_{i}")
            dt = c_t2.selectbox(f"Top L{i+1} Size", [12, 16, 20, 25, 28, 32], index=1, key=f"dt_{i}")
            top_layers.append({'n': nt, 'db': dt})
            
        st.markdown("---")
        # --- Bottom Steel ---
        st.markdown("**Bottom Reinforcement (Positive Moment)**")
        n_bot_layers = st.selectbox("Bottom Layers", [1, 2, 3], index=0, key="bot_L_cnt")
        bot_layers = []
        for i in range(n_bot_layers):
            c_b1, c_b2 = st.columns(2)
            nb = c_b1.number_input(f"Bot L{i+1} Bars", 2, 10, (3 if i==0 else 2), key=f"nb_{i}")
            db = c_b2.selectbox(f"Bot L{i+1} Size", [12, 16, 20, 25, 28, 32], index=2, key=f"db_{i}")
            bot_layers.append({'n': nb, 'db': db})
            
        st.markdown("---")
        # --- Shear ---
        st.markdown("**Shear Reinforcement**")
        cs1, cs2 = st.columns(2)
        stir_db = cs1.selectbox("Stirrup Ø", [6, 9, 10, 12], index=1)
        stir_s = cs2.number_input("Spacing (mm)", value=150.0, step=25.0)

    # 3. Calculation Logic
    # 3.1 Bottom Capacity
    phi_Mn_pos, Ast_pos, a_pos, Mn_pos, c_pos, st_pos = get_phi_Mn_details(
        bot_layers, b, h, fc, fy, cover, stir_db
    )
    
    # 3.2 Top Capacity
    phi_Mn_neg, Ast_neg, a_neg, Mn_neg, c_neg, st_neg = get_phi_Mn_details(
        top_layers, b, h, fc, fy, cover, stir_db
    )
    
    # 3.3 Shear Capacity (Use d from bottom layers approx)
    _, d_shear, _ = calculate_layer_properties(bot_layers, b, h, cover, stir_db)
    if d_shear == 0: d_shear = h - cover - 20 # Fallback
    shear_status, phi_Vn, phi_Vc, phi_Vs, Vc, Vs = check_shear_details(
        vu_max, b, d_shear, fc, fy, stir_db, stir_s
    )
    
    # Prepare Result Dictionary
    design_res = {
        'fc': fc, 'fy': fy, 'b': b, 'h': h,
        'phi_Mn_pos': phi_Mn_pos,
        'phi_Mn_neg': phi_Mn_neg,
        'phi_Vn': phi_Vn,
        'as_prov_bot': Ast_pos,
        'as_prov_top': Ast_neg,
        'top_layers': top_layers,
        'bot_layers': bot_layers,
        'stir_db': stir_db,
        'shear': {'s': stir_s},
        'cover': cover
    }

    # 4. Display Results
    with col_view:
        st.markdown("##### 🖼️ Cross Section Preview")
        svg_xml = plot_cross_section(design_res)
        st.image(svg_xml, use_container_width=False, width=400)
    
    # Call Comparison Function
    display_design_comparison(mu_pos, mu_neg, vu_max, design_res)
    
    # 5. Longitudinal View (Optional but nice)
    st.markdown("---")
    st.markdown("##### 📏 Longitudinal View (Current Span Design Applied to All)")
    # Replicate design for all spans just for visualization
    all_spans_res = [design_res] * len(spans)
    _, png_long = plot_longitudinal_section_detailed(spans, sup_df, all_spans_res, h/1000, cover)
    st.image(png_long, use_container_width=True)
