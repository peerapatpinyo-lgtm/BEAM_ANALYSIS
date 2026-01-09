# design_view.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
# Import engine
from rc_design_engine import get_phi_Mn_details, check_shear_details, calculate_layer_properties
# Import plotter
from section_plotter import plot_longitudinal_section_detailed, plot_cross_section

# --- HELPER: ฟังก์ชันแปลงข้อมูลให้ปลอดภัย (ป้องกัน Error 'int' not iterable) ---
def sanitize_input_list(val):
    """
    แปลงค่าใดๆ ให้เป็น List เสมอ 
    - ถ้าเป็น None -> []
    - ถ้าเป็น int/float -> [val]
    - ถ้าเป็น list/tuple/array -> list(val)
    """
    if val is None:
        return []
    if isinstance(val, (int, float, np.number)):
        return [float(val)]
    if isinstance(val, (list, tuple, np.ndarray)):
        # กรองค่า None ออกจาก list ถ้ามี
        return [v for v in val if v is not None]
    return []

def plot_analysis_results(res_df, spans, supports, loads, reactions):
    """
    สร้างกราฟวิเคราะห์โครงสร้าง
    """
    # 1. Sanitize Spans (ป้องกัน spans เป็นตัวเลขเดี่ยวๆ)
    spans = sanitize_input_list(spans)
    
    # ถ้าไม่มี span เลย ให้ใส่ค่า Default เพื่อไม่ให้กราฟพัง
    if not spans: spans = [1.0]

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
    # ใช้ sum() อย่างปลอดภัยเพราะ spans เป็น list แน่นอนแล้ว
    total_L = sum(spans)
    cum_dist = [0] + list(np.cumsum(spans))
    
    # Beam Line
    fig.add_trace(go.Scatter(
        x=[0, total_L], y=[0, 0], 
        mode='lines', line=dict(color='black', width=4), hoverinfo='skip'
    ), row=1, col=1)
    
    # Supports (Check type strictly)
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

    # Loads (Handle safely)
    load_iter = []
    if isinstance(loads, pd.DataFrame):
        load_iter = loads.to_dict('records')
    else:
        # ถ้า loads เป็น dict เดียวๆ หรือ list หรือ int
        temp_loads = sanitize_input_list(loads)
        # ตรวจสอบว่าใน list เป็น dict หรือไม่ (ถ้าเป็น int ก็จะถูกกรองออก)
        load_iter = [l for l in temp_loads if isinstance(l, dict)]

    for l in load_iter:
        # ดึงค่าแบบปลอดภัย (Default 0)
        try:
            span_idx = int(l.get('span_index', 0))
            if span_idx >= len(spans): span_idx = 0
            
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
        except Exception:
            continue # ข้าม Load ตัวที่มีปัญหา

    # ==========================================
    # ROW 2-4: RESULTS
    # ==========================================
    # Helper for adding traces safely
    def add_res_trace(col_name, row_num, color, name):
        if not res_df.empty and col_name in res_df.columns:
            y_data = res_df[col_name]
            # Handle Unit Conversion
            if col_name in ['shear', 'moment']: y_data = y_data / 1000.0
            
            fig.add_trace(go.Scatter(
                x=res_df['x'], y=y_data, 
                mode='lines', name=name, line=dict(color=color, width=2),
                fill='tozeroy' if row_num != 4 else None
            ), row=row_num, col=1)
            
            # Max/Min Annotations
            try:
                v_max, v_min = y_data.max(), y_data.min()
                # เลือกโชว์เฉพาะค่าที่เยอะกว่า 0.01 (กันรก)
                to_show = set()
                if abs(v_max) > 0.01: to_show.add(v_max)
                if abs(v_min) > 0.01: to_show.add(v_min)
                
                for val in to_show:
                    # หาตำแหน่ง index
                    idx = (y_data - val).abs().idxmin()
                    fig.add_annotation(
                        x=res_df['x'].iloc[idx], y=val,
                        text=f"<b>{val:.2f}</b>", 
                        showarrow=(row_num != 2), 
                        yshift=15 if val > 0 else -15,
                        row=row_num, col=1
                    )
            except:
                pass

    fig.add_hline(y=0, line_color="black", line_width=1, row=2, col=1)
    add_res_trace('shear', 2, '#e74c3c', 'Shear (kN)')
    
    fig.add_hline(y=0, line_color="black", line_width=1, row=3, col=1)
    add_res_trace('moment', 3, '#27ae60', 'Moment (kN-m)')
    
    fig.add_hline(y=0, line_color="black", line_width=1, row=4, col=1)
    add_res_trace('deflection', 4, '#8e44ad', 'Deflection (mm)')

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
    # 1. Sanitize Inputs (สำคัญที่สุด)
    # แปลง spans ให้เป็น list แน่นอน ป้องกัน sum(int) error
    spans = sanitize_input_list(spans)
    if not spans:
        st.warning("⚠️ Please define at least one span in Tab 1.")
        return

    # 2. Unpack solver results SAFELY
    # ป้องกันกรณี solver_results เป็น int (เช่น error code) ทำให้ len(int) หรือ unpacking int พัง
    if not isinstance(solver_results, (list, tuple)):
        st.error("⚠️ Solver did not return valid results. Please check input in Tab 1.")
        return
        
    if len(solver_results) != 5:
        st.error(f"⚠️ Solver returned incomplete data (Expected 5 items, got {len(solver_results)}).")
        return

    x_vals, m_vals, v_vals, d_vals, reactions = solver_results
    
    # Create DataFrame
    res_df = pd.DataFrame({
        'x': x_vals,
        'moment': m_vals,
        'shear': v_vals,
        'deflection': d_vals * 1000
    })
    
    st.markdown("### 📊 Structural Analysis & Design")
    
    # 3. Plotting with Error Handling
    with st.expander("📈 View Analysis Diagrams", expanded=True):
        try:
            fig = plot_analysis_results(res_df, spans, sup_df, loads_df, reactions)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"⚠️ Could not plot diagrams: {e}")

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
        
        # Calculate forces safely
        try:
            L_span = spans[selected_span_idx]
            x_start = sum(spans[:selected_span_idx])
            x_end = x_start + L_span
            mask = (x_vals >= x_start) & (x_vals <= x_end)
            
            mu_pos, mu_neg, vu_max = 0.0, 0.0, 0.0
            if any(mask):
                mu_pos = max(0, np.max(m_vals[mask])/1000)
                vu_max = np.max(np.abs(v_vals[mask]))/1000
                
                idx_s = (np.abs(x_vals - x_start)).argmin()
                idx_e = (np.abs(x_vals - x_end)).argmin()
                m_s = abs(m_vals[idx_s])/1000
                m_e = abs(m_vals[idx_e])/1000
                mu_neg = max(m_s, m_e)
        except Exception:
            # Fallback values
            mu_pos, mu_neg, vu_max = 0, 0, 0

    with col_param:
        c1, c2, c3, c4 = st.columns(4)
        fc = c1.number_input("f'c (MPa)", value=28.0, step=1.0)
        fy = c2.number_input("fy (MPa)", value=420.0, step=10.0)
        
        # Safe Param Access
        p_b = 0.2
        p_h = 0.4
        if isinstance(params, dict):
            p_b = params.get('b', 0.2)
            p_h = params.get('h', 0.4)
            
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

    # 3. Calculation & Display
    try:
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
                st.info("Section preview unavailable.")

        display_design_comparison(mu_pos, mu_neg, vu_max, design_res)
        
        # Longitudinal
        st.markdown("---")
        st.markdown("##### Longitudinal View")
        all_spans_res = [design_res] * len(spans)
        
        try:
            _, png_long = plot_longitudinal_section_detailed(spans, sup_df, all_spans_res, h/1000, cover)
            st.image(png_long, use_container_width=True)
        except Exception as e:
            st.info("Longitudinal view unavailable.")
            
    except Exception as e:
        st.error(f"Design Calculation Error: {e}")
