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
    # ป้องกัน Error: ตรวจสอบว่า spans เป็น List หรือไม่
    if not isinstance(spans, (list, tuple, np.ndarray)):
        spans = [spans] if spans is not None else []
        
    total_L = sum(spans) if len(spans) > 0 else 0
    cum_dist = [0] + list(np.cumsum(spans))
    
    # Beam Line
    fig.add_trace(go.Scatter(
        x=[0, total_L], y=[0, 0], 
        mode='lines', line=dict(color='black', width=4), hoverinfo='skip'
    ), row=1, col=1)
    
    # Supports (Defensive Check)
    if isinstance(supports, pd.DataFrame):
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

    # Loads (FIXED: Strict Type Checking to prevent 'int is not iterable')
    load_iter = []
    if isinstance(loads, pd.DataFrame):
        load_iter = loads.to_dict('records')
    elif isinstance(loads, (list, tuple)):
        load_iter = loads
    # ถ้า loads เป็น int หรือ None, load_iter จะเป็น [] ทำให้ไม่เกิด Error ตอนวนลูป

    for l in load_iter:
        # ป้องกัน Data ไม่ครบ
        if not isinstance(l, dict): continue

        span_idx = int(l.get('span_index', 0))
        if span_idx >= len(cum_dist): continue # ป้องกัน Index Out of Range
        
        start_x = cum_dist[span_idx]
        mag_kN = float(l.get('mag', 0)) / 1000.0
        
        if l.get('type') == 'P':
            x_loc = start_x + float(l.get('d_start', 0)) 
            fig.add_annotation(
                x=x_loc, y=0, ax=0, ay=-50,
                xref="x1", yref="y1",
                showarrow=True, arrowhead=2, arrowsize=1.2, arrowwidth=2, arrowcolor="#c0392b",
                text=f"<b>P={mag_kN:.2f} kN</b>", yshift=5, row=1, col=1
            )
        elif l.get('type') == 'U':
            x_s = start_x + float(l.get('d_start', 0))
            x_e = x_s + float(l.get('dist', 0))
            h_vis = 0.25
            fig.add_trace(go.Scatter(
                x=[x_s, x_e], y=[h_vis, h_vis],
                mode='lines', line=dict(color='#2980b9', width=2), hoverinfo='skip'
            ), row=1, col=1)
            
            # ป้องกัน Error ตอนคำนวณลูกศร
            dist_val = float(l.get('dist', 1))
            n_arrows = max(3, int(dist_val * 3)) 
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
    if 'shear' in res_df.columns:
        fig.add_trace(go.Scatter(
            x=res_df['x'], y=res_df['shear']/1000, 
            mode='lines', name='Shear (kN)', line=dict(color='#e74c3c', width=2),
            fill='tozeroy', fillcolor='rgba(231, 76, 60, 0.1)'
        ), row=2, col=1)
        
        # Annotations (Check empty)
        if not res_df.empty:
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
    if 'moment' in res_df.columns:
        fig.add_trace(go.Scatter(
            x=res_df['x'], y=res_df['moment']/1000, 
            mode='lines', name='Moment (kN-m)', line=dict(color='#27ae60', width=2),
            fill='tozeroy', fillcolor='rgba(39, 174, 96, 0.1)'
        ), row=3, col=1)

        if not res_df.empty:
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
    if 'deflection' in res_df.columns:
        fig.add_trace(go.Scatter(
            x=res_df['x'], y=res_df['deflection'], 
            mode='lines', name='Deflection (mm)', line=dict(color='#8e44ad', width=2)
        ), row=4, col=1)
        
        if not res_df.empty:
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
    
    fc = design_res.get('fc', 24)
    fy = design_res.get('fy', 400)
    b = design_res.get('b', 200)
    h = design_res.get('h', 400)
    
    # Check Min Steel (Approx d)
    d = h - 50 
    as_min = max((0.25 * np.sqrt(fc) / fy) * b * d, (1.4 / fy) * b * d)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Bottom Reinforcement (Mid-span)**")
        as_prov = design_res.get('as_prov_bot', 0.0)
        st.write(f"Min Req: `{as_min:.0f}` mm² | Provided: `{as_prov:.0f}` mm²")
        if as_prov >= as_min: st.success("✅ Area OK")
        else: st.error("❌ Area < Min")
        
    with col2:
        st.markdown("**Top Reinforcement (Support)**")
        as_prov_t = design_res.get('as_prov_top', 0.0)
        st.write(f"Min Req: `{as_min:.0f}` mm² | Provided: `{as_prov_t:.0f}` mm²")
        if as_prov_t >= as_min: st.success("✅ Area OK")
        else: st.error("❌ Area < Min")

    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.metric("Capacity φMn(+)", f"{design_res['phi_Mn_pos']:.2f} kNm", 
                  delta=f"{design_res['phi_Mn_pos'] - mu_pos:.2f}")
    with c2:
        st.metric("Capacity φMn(-)", f"{design_res['phi_Mn_neg']:.2f} kNm", 
                  delta=f"{design_res['phi_Mn_neg'] - abs(mu_neg):.2f}")
    with c3:
        st.metric("Capacity φVn", f"{design_res['phi_Vn']:.2f} kN", 
                  delta=f"{design_res['phi_Vn'] - vu:.2f}")

def render_design_view(solver_results, params, spans, sup_df, loads_df):
    """
    Main Logic for Tab 2
    """
    # ป้องกัน Crash หาก solver_results เป็น None หรือ unpack ไม่ได้
    if not solver_results or len(solver_results) != 5:
        st.error("Solver did not return valid results.")
        return

    x_vals, m_vals, v_vals, d_vals, reactions = solver_results
    
    # Defensive DataFrame Creation
    res_df = pd.DataFrame({
        'x': x_vals,
        'moment': m_vals,
        'shear': v_vals,
        'deflection': d_vals * 1000
    })
    
    st.markdown("### 📊 Structural Analysis & Design")
    with st.expander("📈 View Analysis Diagrams", expanded=True):
        # เรียก function วาดกราฟ (ที่แก้บั๊กแล้ว)
        fig = plot_analysis_results(res_df, spans, sup_df, loads_df, reactions)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("### 🏗️ Member Design")

    # 1. Select Span
    col_sel, col_param = st.columns([1, 2])
    with col_sel:
        # ป้องกัน n_spans เป็น 0 หรือ error
        if not isinstance(spans, list): spans = []
        n_spans = len(spans)
        
        if n_spans > 0:
            span_options = [f"Span {i+1} (L={spans[i]:.2f}m)" for i in range(n_spans)]
            selected_span_label = st.radio("Select Span:", span_options)
            try:
                selected_span_idx = span_options.index(selected_span_label)
            except:
                selected_span_idx = 0
            
            # Get Span Forces
            L_span = spans[selected_span_idx]
            x_start = sum(spans[:selected_span_idx])
            x_end = x_start + L_span
            mask = (x_vals >= x_start) & (x_vals <= x_end)
            
            if any(mask):
                mu_pos = np.max(m_vals[mask])/1000
                mu_pos = max(0, mu_pos)
                vu_max = np.max(np.abs(v_vals[mask]))/1000
                
                # Simple neg moment approx
                idx_s = (np.abs(x_vals - x_start)).argmin()
                idx_e = (np.abs(x_vals - x_end)).argmin()
                m_start = abs(m_vals[idx_s]/1000)
                m_end = abs(m_vals[idx_e]/1000)
                mu_neg = max(m_start, m_end)
            else:
                mu_pos, mu_neg, vu_max = 0, 0, 0
        else:
            st.warning("No spans defined.")
            return

    with col_param:
        c1, c2, c3, c4 = st.columns(4)
        fc = c1.number_input("f'c (MPa)", value=28.0, step=1.0)
        fy = c2.number_input("fy (MPa)", value=420.0, step=10.0)
        b = c3.number_input("b (mm)", value=float(params.get('b', 0.2)*1000), step=50.0)
        h = c4.number_input("h (mm)", value=float(params.get('h', 0.4)*1000), step=50.0)
        cover = st.number_input("Cover (mm)", value=25.0)

    # 2. Reinforcement Inputs
    st.markdown("---")
    c_in, c_view = st.columns([1, 1])
    
    with c_in:
        st.subheader("✏️ Reinforcement")
        
        # Top Layers
        st.markdown("**Top Steel (Negative)**")
        n_top = st.selectbox("Top Layers", [1, 2], key="top_l")
        top_layers = []
        for i in range(n_top):
            c1, c2 = st.columns(2)
            nt = c1.number_input(f"Top L{i+1} Bars", 2, 8, 2, key=f"t_n_{i}")
            dt = c2.selectbox(f"Top L{i+1} Size", [12, 16, 20, 25], index=1, key=f"t_d_{i}")
            top_layers.append({'n': nt, 'db': dt})
            
        # Bottom Layers
        st.markdown("**Bottom Steel (Positive)**")
        n_bot = st.selectbox("Bot Layers", [1, 2, 3], key="bot_l")
        bot_layers = []
        for i in range(n_bot):
            c1, c2 = st.columns(2)
            nb = c1.number_input(f"Bot L{i+1} Bars", 2, 8, 2, key=f"b_n_{i}")
            db = c2.selectbox(f"Bot L{i+1} Size", [12, 16, 20, 25], index=2, key=f"b_d_{i}")
            bot_layers.append({'n': nb, 'db': db})
            
        # Stirrup
        st.markdown("**Shear Stirrup**")
        cs1, cs2 = st.columns(2)
        stir_db = cs1.selectbox("Stirrup Ø", [6, 9, 10, 12], index=1)
        stir_s = cs2.number_input("Spacing (mm)", value=150.0, step=25.0)

    # 3. Calculation (Passed stir_db correctly)
    # ---------------------------------------------------
    # ตรวจสอบว่ามีข้อมูล layers ส่งไปจริง ไม่ใช่ int หรือ None
    if not isinstance(bot_layers, list): bot_layers = []
    if not isinstance(top_layers, list): top_layers = []

    phi_Mn_pos, Ast_pos, a_pos, Mn_pos, c_pos, st_pos = get_phi_Mn_details(
        bot_layers, b, h, fc, fy, cover, stir_db
    )
    
    phi_Mn_neg, Ast_neg, a_neg, Mn_neg, c_neg, st_neg = get_phi_Mn_details(
        top_layers, b, h, fc, fy, cover, stir_db
    )
    
    # Calculate d for shear
    _, d_shear, _ = calculate_layer_properties(bot_layers, b, h, cover, stir_db)
    if d_shear <= 0: d_shear = h - cover - 20 # Fallback
    
    shear_status, phi_Vn, phi_Vc, phi_Vs, Vc, Vs = check_shear_details(
        vu_max, b, d_shear, fc, fy, stir_db, stir_s
    )
    # ---------------------------------------------------

    design_res = {
        'fc': fc, 'fy': fy, 'b': b, 'h': h,
        'phi_Mn_pos': phi_Mn_pos, 'phi_Mn_neg': phi_Mn_neg, 'phi_Vn': phi_Vn,
        'as_prov_bot': Ast_pos, 'as_prov_top': Ast_neg,
        'top_layers': top_layers, 'bot_layers': bot_layers,
        'stir_db': stir_db, 'shear': {'s': stir_s}, 'cover': cover
    }

    with c_view:
        st.markdown("##### Section Preview")
        # ป้องกัน Error ใน section plotter
        try:
            svg_xml = plot_cross_section(design_res)
            st.image(svg_xml, width=350)
        except Exception as e:
            st.error(f"Cannot plot section: {e}")

    display_design_comparison(mu_pos, mu_neg, vu_max, design_res)
    
    # Longitudinal
    st.markdown("---")
    st.markdown("##### Longitudinal View")
    all_spans_res = [design_res] * len(spans)
    
    try:
        _, png_long = plot_longitudinal_section_detailed(spans, sup_df, all_spans_res, h/1000, cover)
        st.image(png_long, use_container_width=True)
    except Exception as e:
        st.info(f"Longitudinal view not available: {e}")
