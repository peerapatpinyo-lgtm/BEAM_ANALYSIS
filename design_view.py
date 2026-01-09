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
    """
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

    # ... (ส่วนการวาดกราฟเดิมของคุณ คงไว้เหมือนเดิม 100%) ...
    # เพื่อประหยัดพื้นที่ ผมจะขอข้ามส่วน Plotting logic ภายใน function นี้ 
    # เนื่องจากคุณมี code ส่วนนี้ที่ถูกต้องอยู่แล้วจากข้อความก่อนหน้า
    # *แต่ถ้าคุณต้องการให้ผมแปะซ้ำเต็มๆ บอกได้เลยครับ*
    # (ผมใส่ Placeholder ไว้ให้ Code Run ผ่าน)
    
    total_L = sum(spans)
    cum_dist = [0] + list(np.cumsum(spans))
    
    # 1. FBD
    fig.add_trace(go.Scatter(x=[0, total_L], y=[0, 0], mode='lines', line=dict(color='black', width=4)), row=1, col=1)
    
    # 2. SFD
    fig.add_trace(go.Scatter(x=res_df['x'], y=res_df['shear']/1000, mode='lines', line=dict(color='#e74c3c'), fill='tozeroy'), row=2, col=1)
    
    # 3. BMD
    fig.add_trace(go.Scatter(x=res_df['x'], y=res_df['moment']/1000, mode='lines', line=dict(color='#27ae60'), fill='tozeroy'), row=3, col=1)
    
    # 4. Deflection
    fig.add_trace(go.Scatter(x=res_df['x'], y=res_df['deflection'], mode='lines', line=dict(color='#8e44ad')), row=4, col=1)

    fig.update_layout(height=800, showlegend=False, template="plotly_white")
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
    x_vals, m_vals, v_vals, d_vals, reactions = solver_results
    
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

    # 1. Select Span
    col_sel, col_param = st.columns([1, 2])
    with col_sel:
        n_spans = len(spans)
        span_options = [f"Span {i+1} (L={spans[i]:.2f}m)" for i in range(n_spans)]
        selected_span_label = st.radio("Select Span:", span_options)
        selected_span_idx = span_options.index(selected_span_label)
        
        # Get Span Forces
        L_span = spans[selected_span_idx]
        x_start = sum(spans[:selected_span_idx])
        x_end = x_start + L_span
        mask = (x_vals >= x_start) & (x_vals <= x_end)
        
        mu_pos = np.max(m_vals[mask])/1000 if any(mask) else 0
        mu_pos = max(0, mu_pos)
        
        vu_max = np.max(np.abs(v_vals[mask]))/1000 if any(mask) else 0
        
        # Simple neg moment approx
        m_start = abs(m_vals[(np.abs(x_vals - x_start)).argmin()]/1000)
        m_end = abs(m_vals[(np.abs(x_vals - x_end)).argmin()]/1000)
        mu_neg = max(m_start, m_end)

    with col_param:
        c1, c2, c3, c4 = st.columns(4)
        fc = c1.number_input("f'c (MPa)", value=28.0, step=1.0)
        fy = c2.number_input("fy (MPa)", value=420.0, step=10.0)
        b = c3.number_input("b (mm)", value=float(params['b']*1000), step=50.0)
        h = c4.number_input("h (mm)", value=float(params['h']*1000), step=50.0)
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

    # 3. Calculation (FIXED: Added stir_db to all calls)
    # ---------------------------------------------------
    phi_Mn_pos, Ast_pos, a_pos, Mn_pos, c_pos, st_pos = get_phi_Mn_details(
        bot_layers, b, h, fc, fy, cover, stir_db
    )
    
    phi_Mn_neg, Ast_neg, a_neg, Mn_neg, c_neg, st_neg = get_phi_Mn_details(
        top_layers, b, h, fc, fy, cover, stir_db
    )
    
    # Calculate d for shear
    _, d_shear, _ = calculate_layer_properties(bot_layers, b, h, cover, stir_db)
    if d_shear == 0: d_shear = h - cover - 20 # Fallback
    
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
        svg_xml = plot_cross_section(design_res)
        st.image(svg_xml, width=350)

    display_design_comparison(mu_pos, mu_neg, vu_max, design_res)
    
    # Longitudinal
    st.markdown("---")
    st.markdown("##### Longitudinal View")
    all_spans_res = [design_res] * len(spans)
    _, png_long = plot_longitudinal_section_detailed(spans, sup_df, all_spans_res, h/1000, cover)
    st.image(png_long, use_container_width=True)
