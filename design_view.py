# design_view.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
# Import engine
from rc_design_engine import get_phi_Mn_details, check_shear_details, calculate_layer_properties
# Import plotter (assume exists)
from section_plotter import plot_longitudinal_section_detailed, plot_cross_section

# --- HELPER: กันเหนียวทุกกรณี (Defensive Programming) ---
def sanitize_input_list(val):
    """แปลง input ใดๆ ให้เป็น list เสมอ เพื่อแก้ปัญหา 'int' object is not iterable"""
    if val is None:
        return []
    if isinstance(val, (int, float, np.number)):
        # ถ้ามาเป็นตัวเลขตัวเดียว ให้มองเป็น list ที่มีสมาชิก 1 ตัว
        return [float(val)]
    if isinstance(val, (list, tuple, np.ndarray)):
        return list(val)
    return []

# -------------------------------------------------------

def plot_analysis_results(res_df, spans, supports, loads, reactions):
    """
    สร้างกราฟวิเคราะห์โครงสร้าง (Textbook-style)
    """
    # 1. Sanitize Inputs (แก้บั๊กตัวเลขไม่ใช่ list)
    spans = sanitize_input_list(spans)
    
    # --- Create Subplots ---
    fig = make_subplots(
        rows=4, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.08,
        subplot_titles=(
            "<b>1. Free Body Diagram (FBD)</b>", 
            "<b>2. Shear Force Diagram (SFD)</b>", 
            "<b>3. Bending Moment Diagram (BMD)</b>",
            "<b>4. Elastic Curve (Deflection)</b>"
        ),
        row_heights=[0.20, 0.25, 0.25, 0.30]
    )

    # ==========================================
    # ROW 1: LOAD MODEL (FBD)
    # ==========================================
    total_L = sum(spans) if spans else 0
    cum_dist = [0] + list(np.cumsum(spans))
    
    # Beam Line
    fig.add_trace(go.Scatter(
        x=[0, total_L], y=[0, 0], 
        mode='lines', line=dict(color='black', width=4), hoverinfo='skip'
    ), row=1, col=1)
    
    # Supports
    if isinstance(supports, pd.DataFrame) and not supports.empty:
        for idx, row in supports.iterrows():
            sym = "triangle-up"
            if row.get('type') == 'Fixed': sym = "square"
            elif row.get('type') == 'Roller': sym = "circle"
            
            fig.add_trace(go.Scatter(
                x=[row.get('x', 0)], y=[-0.08], 
                mode='markers+text',
                marker=dict(symbol=sym, size=14, color='white', line=dict(width=2, color='black')),
                text=[str(row.get('type','S'))[0]], textposition="bottom center",
                hoverinfo='name', name=f"Support"
            ), row=1, col=1)

    # Loads (Sanitize loads)
    load_iter = []
    if isinstance(loads, pd.DataFrame):
        load_iter = loads.to_dict('records')
    else:
        load_iter = sanitize_input_list(loads)

    for l in load_iter:
        if not isinstance(l, dict): continue # ข้ามถ้าข้อมูลไม่ใช่ dict

        # ดึงค่าแบบปลอดภัย (Default 0)
        span_idx = int(l.get('span_index', 0))
        if span_idx >= len(spans): span_idx = 0 # กัน index เกิน
        
        start_x = cum_dist[span_idx] if span_idx < len(cum_dist) else 0
        mag_kN = float(l.get('mag', 0)) / 1000.0
        d_start = float(l.get('d_start', 0))
        
        if l.get('type') == 'P':
            x_loc = start_x + d_start
            fig.add_annotation(
                x=x_loc, y=0, ax=0, ay=-50,
                xref="x1", yref="y1",
                showarrow=True, arrowhead=2, arrowsize=1.2, arrowwidth=2, arrowcolor="#c0392b",
                text=f"<b>P={mag_kN:.2f} kN</b>", yshift=5, row=1, col=1
            )
        elif l.get('type') == 'U':
            dist = float(l.get('dist', 0))
            x_s = start_x + d_start
            x_e = x_s + dist
            h_vis = 0.25
            
            fig.add_trace(go.Scatter(
                x=[x_s, x_e], y=[h_vis, h_vis],
                mode='lines', line=dict(color='#2980b9', width=2), hoverinfo='skip'
            ), row=1, col=1)
            
            # Arrows
            n_arrows = max(3, int(dist * 3)) 
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
    if not res_df.empty and 'shear' in res_df.columns:
        fig.add_trace(go.Scatter(
            x=res_df['x'], y=res_df['shear']/1000, 
            mode='lines', name='Shear (kN)', line=dict(color='#e74c3c', width=2),
            fill='tozeroy', fillcolor='rgba(231, 76, 60, 0.1)'
        ), row=2, col=1)
        
        # Max/Min Labels
        v_vals = res_df['shear']/1000
        v_max, v_min = v_vals.max(), v_vals.min()
        for val in [v_max, v_min]:
            if abs(val) > 0.01:
                idx = (v_vals - val).abs().idxmin()
                fig.add_annotation(
                    x=res_df['x'].iloc[idx], y=val,
                    text=f"<b>{val:.2f}</b>", showarrow=False, yshift=15 if val>0 else -15,
                    font=dict(color='#e74c3c', size=10), row=2, col=1
                )

    # ==========================================
    # ROW 3: BENDING MOMENT (BMD)
    # ==========================================
    fig.add_hline(y=0, line_color="black", line_width=1, row=3, col=1)
    if not res_df.empty and 'moment' in res_df.columns:
        fig.add_trace(go.Scatter(
            x=res_df['x'], y=res_df['moment']/1000, 
            mode='lines', name='Moment (kN-m)', line=dict(color='#27ae60', width=2),
            fill='tozeroy', fillcolor='rgba(39, 174, 96, 0.1)'
        ), row=3, col=1)

        m_vals = res_df['moment']/1000
        m_max, m_min = m_vals.max(), m_vals.min()
        for val in [m_max, m_min]:
            if abs(val) > 0.01:
                idx = (m_vals - val).abs().idxmin()
                fig.add_annotation(
                    x=res_df['x'].iloc[idx], y=val,
                    text=f"<b>{val:.2f}</b>", 
                    showarrow=True, arrowhead=1, ay=30 if val>0 else -30,
                    font=dict(color='#27ae60', size=10), row=3, col=1
                )

    # ==========================================
    # ROW 4: DEFLECTION
    # ==========================================
    fig.add_hline(y=0, line_color="black", line_width=1, row=4, col=1)
    if not res_df.empty and 'deflection' in res_df.columns:
        fig.add_trace(go.Scatter(
            x=res_df['x'], y=res_df['deflection'], 
            mode='lines', name='Deflection (mm)', line=dict(color='#8e44ad', width=2)
        ), row=4, col=1)
        
        idx_max = res_df['deflection'].abs().idxmax()
        max_d = res_df['deflection'].iloc[idx_max]
        if abs(max_d) > 0.001:
            fig.add_annotation(
                x=res_df['x'].iloc[idx_max], y=max_d,
                text=f"<b>Max: {max_d:.2f} mm</b>",
                showarrow=True, arrowhead=1, ay=40 if max_d < 0 else -40,
                font=dict(color='#8e44ad', size=10), row=4, col=1
            )

    # Layout Updates
    for x_pos in cum_dist:
        fig.add_vline(x=x_pos, line_width=1, line_dash="dash", line_color="gray", opacity=0.3)

    fig.update_layout(
        title="<b>Structural Analysis Results</b>",
        height=900, showlegend=False, template="plotly_white", hovermode="x unified",
        margin=dict(t=60, b=40, l=60, r=20)
    )
    
    fig.update_yaxes(visible=False, range=[-0.5, 0.8], row=1, col=1)
    fig.update_yaxes(title_text="V (kN)", showgrid=True, row=2, col=1)
    fig.update_yaxes(title_text="M (kN-m)", autorange="reversed", showgrid=True, row=3, col=1)
    fig.update_yaxes(title_text="δ (mm)", showgrid=True, zeroline=True, row=4, col=1)
    fig.update_xaxes(title_text="Length (m)", row=4, col=1)

    return fig

def display_design_comparison(mu_pos, mu_neg, vu, design_res):
    st.markdown("---")
    st.subheader("🛠 RC Design Verification")
    
    fc = design_res.get('fc', 24)
    fy = design_res.get('fy', 400)
    b = design_res.get('b', 200)
    h = design_res.get('h', 400)
    
    d = h - 50 
    as_min = max((0.25 * np.sqrt(fc) / fy) * b * d, (1.4 / fy) * b * d)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Bottom Reinf. (Mid-span)**")
        as_prov = design_res.get('as_prov_bot', 0.0)
        st.write(f"Req Min: `{as_min:.0f}` | Prov: `{as_prov:.0f}` mm²")
        if as_prov >= as_min: st.success("✅ Area OK")
        else: st.error("❌ Area < Min")
        
    with col2:
        st.markdown("**Top Reinf. (Support)**")
        as_prov_t = design_res.get('as_prov_top', 0.0)
        st.write(f"Req Min: `{as_min:.0f}` | Prov: `{as_prov_t:.0f}` mm²")
        if as_prov_t >= as_min: st.success("✅ Area OK")
        else: st.error("❌ Area < Min")

    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    with c1: st.metric("φMn(+)", f"{design_res['phi_Mn_pos']:.2f} kNm", delta=f"{design_res['phi_Mn_pos'] - mu_pos:.2f}")
    with c2: st.metric("φMn(-)", f"{design_res['phi_Mn_neg']:.2f} kNm", delta=f"{design_res['phi_Mn_neg'] - abs(mu_neg):.2f}")
    with c3: st.metric("φVn", f"{design_res['phi_Vn']:.2f} kN", delta=f"{design_res['phi_Vn'] - vu:.2f}")

def render_design_view(solver_results, params, spans, sup_df, loads_df):
    """
    Main Logic for Tab 2
    """
    # 1. Sanitize Data BEFORE Processing (สำคัญมาก จุดแก้บั๊ก)
    spans = sanitize_input_list(spans)
    
    # ถ้าไม่มี span เลย ให้หยุด
    if not spans:
        st.error("No span data found.")
        return

    # Unpack solver results safely
    if not solver_results or len(solver_results) != 5:
        st.error("Solver Error: Invalid results format.")
        return

    x_vals, m_vals, v_vals, d_vals, reactions = solver_results
    
    # Create DataFrame safely
    res_df = pd.DataFrame({
        'x': x_vals,
        'moment': m_vals,
        'shear': v_vals,
        'deflection': d_vals * 1000
    })
    
    st.markdown("### 📊 Structural Analysis & Design")
    with st.expander("📈 View Analysis Diagrams", expanded=True):
        fig = plot_analysis_results(res_df, spans, sup_df, loads_df, reactions)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("### 🏗️ Member Design")

    # Select Span
    col_sel, col_param = st.columns([1, 2])
    with col_sel:
        n_spans = len(spans)
        span_options = [f"Span {i+1} (L={spans[i]:.2f}m)" for i in range(n_spans)]
        selected_span_label = st.radio("Select Span:", span_options)
        
        try:
            selected_span_idx = span_options.index(selected_span_label)
        except:
            selected_span_idx = 0
        
        # Calculate forces for selected span
        L_span = spans[selected_span_idx]
        x_start = sum(spans[:selected_span_idx])
        x_end = x_start + L_span
        mask = (x_vals >= x_start) & (x_vals <= x_end)
        
        mu_pos, mu_neg, vu_max = 0.0, 0.0, 0.0
        if any(mask):
            mu_pos = max(0, np.max(m_vals[mask])/1000)
            vu_max = np.max(np.abs(v_vals[mask]))/1000
            # Simple neg moment (start/end of span)
            idx_s = (np.abs(x_vals - x_start)).argmin()
            idx_e = (np.abs(x_vals - x_end)).argmin()
            m_s = abs(m_vals[idx_s])/1000
            m_e = abs(m_vals[idx_e])/1000
            mu_neg = max(m_s, m_e)

    with col_param:
        c1, c2, c3, c4 = st.columns(4)
        fc = c1.number_input("f'c (MPa)", value=28.0, step=1.0)
        fy = c2.number_input("fy (MPa)", value=420.0, step=10.0)
        # ป้องกัน params เป็น None
        p_b = params.get('b', 0.2) if params else 0.2
        p_h = params.get('h', 0.4) if params else 0.4
        b = c3.number_input("b (mm)", value=float(p_b*1000), step=50.0)
        h = c4.number_input("h (mm)", value=float(p_h*1000), step=50.0)
        cover = st.number_input("Cover (mm)", value=25.0)

    # Reinforcement Inputs
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

    # 3. Calculation
    phi_Mn_pos, Ast_pos, _, _, _, _ = get_phi_Mn_details(bot_layers, b, h, fc, fy, cover, stir_db)
    phi_Mn_neg, Ast_neg, _, _, _, _ = get_phi_Mn_details(top_layers, b, h, fc, fy, cover, stir_db)
    
    _, d_shear, _ = calculate_layer_properties(bot_layers, b, h, cover, stir_db)
    if d_shear <= 0: d_shear = h - cover - 20
    
    _, phi_Vn, _, _, _, _ = check_shear_details(vu_max, b, d_shear, fc, fy, stir_db, stir_s)

    design_res = {
        'fc': fc, 'fy': fy, 'b': b, 'h': h,
        'phi_Mn_pos': phi_Mn_pos, 'phi_Mn_neg': phi_Mn_neg, 'phi_Vn': phi_Vn,
        'as_prov_bot': Ast_pos, 'as_prov_top': Ast_neg,
        'top_layers': top_layers, 'bot_layers': bot_layers,
        'stir_db': stir_db, 'shear': {'s': stir_s}, 'cover': cover
    }

    with c_view:
        st.markdown("##### Section Preview")
        try:
            svg_xml = plot_cross_section(design_res)
            st.image(svg_xml, width=350)
        except Exception as e:
            st.error(f"Error drawing section: {e}")

    display_design_comparison(mu_pos, mu_neg, vu_max, design_res)
    
    # Longitudinal
    st.markdown("---")
    st.markdown("##### Longitudinal View")
    all_spans_res = [design_res] * len(spans)
    
    try:
        _, png_long = plot_longitudinal_section_detailed(spans, sup_df, all_spans_res, h/1000, cover)
        st.image(png_long, use_container_width=True)
    except Exception as e:
        st.info("Longitudinal view unavailable (Check section plotter logic)")
