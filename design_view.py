import streamlit as st
import plotly.graph_objects as go
import numpy as np
import input_handler as ui  # เรียกใช้ input_handler เดิม
from rc_design import calculate_rc_design

# ==========================================
# HELPER: DRAW REINFORCEMENT PROFILE
# ==========================================
def draw_reinforcement_profile(spans, design_results, m_bar, s_bar):
    total_len = sum(spans)
    cum_len = [0] + list(np.cumsum(spans))
    fig = go.Figure()
    # Beam Body
    fig.add_shape(type="rect", x0=0, y0=-0.5, x1=total_len, y1=0.5, line=dict(color="black", width=2), fillcolor="#F5F5F5", layer="below")

    for i in range(len(spans)):
        x0, x1 = cum_len[i], cum_len[i+1]
        res = design_results[i]
        
        # Bottom Rebar
        if res.get('bot_nb'):
            txt = f"{res['bot_nb']}-{m_bar}"
            fig.add_trace(go.Scatter(x=[x0 + 0.1, x1 - 0.1], y=[-0.35, -0.35], mode='lines+text', line=dict(color='#1565C0', width=5), text=[txt, txt], textposition="bottom center", showlegend=False, hoverinfo='text', hovertext=f"Bottom: {txt}"))
        
        # Top Rebar
        if res.get('top_nb'):
            txt = f"{res['top_nb']}-{m_bar}"
            fig.add_trace(go.Scatter(x=[x0, x1], y=[0.35, 0.35], mode='lines+text', line=dict(color='#D32F2F', width=3), text=[txt, txt], textposition="top center", showlegend=False, hoverinfo='text', hovertext=f"Top: {txt}"))

        # Stirrup Label
        mid = (x0+x1)/2
        stir_txt = f"{s_bar} {res.get('stir_text', '-')}"
        fig.add_annotation(x=mid, y=0, text=f"<b>SPAN {i+1}</b><br>Stir: {stir_txt}", showarrow=False, font=dict(size=10), bgcolor="rgba(255,255,255,0.8)")

    # Supports
    for x in cum_len: 
        fig.add_trace(go.Scatter(x=[x], y=[-0.6], mode='markers', marker=dict(symbol="triangle-up", size=15, color="#333"), showlegend=False, hoverinfo='skip'))
    
    fig.update_layout(title="🏗️ Reinforcement Profile", height=300, xaxis=dict(range=[-0.5, total_len+0.5], showgrid=True), yaxis=dict(visible=False, range=[-1, 1]), margin=dict(l=20,r=20,t=40,b=20), plot_bgcolor='white')
    return fig

# ==========================================
# HELPER: DRAW CROSS SECTION
# ==========================================
def draw_section_real(b_cm, h_cm, cov_cm, nb_bot, nb_top, bd_mm, stir_d_mm, main_name, stir_name, s_val_mm, title):
    fig = go.Figure()
    bd_cm, sd_cm = bd_mm/10.0, stir_d_mm/10.0
    
    # Concrete Face
    fig.add_shape(type="rect", x0=0, y0=0, x1=b_cm, y1=h_cm, line=dict(color="black", width=3), fillcolor="#FAFAFA")
    
    # Stirrup
    sx0, sy0, sx1, sy1 = cov_cm, cov_cm, b_cm - cov_cm, h_cm - cov_cm
    fig.add_shape(type="rect", x0=sx0, y0=sy0, x1=sx1, y1=sy1, line=dict(color="#C62828", width=3), fillcolor="rgba(0,0,0,0)")
    
    # Bottom Bars
    if nb_bot > 0:
        start_x, end_x = cov_cm + sd_cm + bd_cm/2, b_cm - (cov_cm + sd_cm + bd_cm/2)
        y_pos = cov_cm + sd_cm + bd_cm/2
        x_pos = [start_x] if nb_bot==1 else np.linspace(start_x, end_x, nb_bot)
        for xp in x_pos: fig.add_shape(type="circle", x0=xp-bd_cm/2, y0=y_pos-bd_cm/2, x1=xp+bd_cm/2, y1=y_pos+bd_cm/2, line_color="black", fillcolor="#1565C0")
            
    # Top Bars
    if nb_top > 0:
        start_x, end_x = cov_cm + sd_cm + bd_cm/2, b_cm - (cov_cm + sd_cm + bd_cm/2)
        y_top = h_cm - (cov_cm + sd_cm + bd_cm/2)
        x_pos = [start_x] if nb_top==1 else np.linspace(start_x, end_x, nb_top)
        for xp in x_pos: fig.add_shape(type="circle", x0=xp-bd_cm/2, y0=y_top-bd_cm/2, x1=xp+bd_cm/2, y1=y_top+bd_cm/2, line_color="black", fillcolor="#D32F2F")

    fig.add_annotation(x=b_cm/2, y=h_cm/2, text=f"Top: {nb_top}-{main_name}<br>Bot: {nb_bot}-{main_name}<br>{stir_name}@{s_val_mm}mm", showarrow=False, font=dict(size=14, color="#333"))
    fig.update_layout(title=dict(text=title, x=0.5), width=250, height=300, xaxis=dict(visible=False, range=[-b_cm*0.5, b_cm*1.5]), yaxis=dict(visible=False, range=[-h_cm*0.2, h_cm*1.2]), margin=dict(l=10,r=10,t=40,b=10), plot_bgcolor='white')
    return fig

# ==========================================
# MAIN EXPORT FUNCTION
# ==========================================
def render_design_section(df, vis_spans, unit_sys, method):
    """
    ฟังก์ชันหลักสำหรับเรียกส่วน Design ทั้งหมด
    รับค่า Analysis Result (df) และ Geometry (vis_spans) เข้ามาคำนวณต่อ
    """
    st.markdown('<div class="sub-header">2️⃣ Structural Design Results</div>', unsafe_allow_html=True)
    
    # 1. รับค่า Design Parameters (Material & Section)
    fc, fy, b_cm, h_cm, cov_cm, m_bar, s_bar, man_s_cm = ui.render_design_input(unit_sys)
    
    # 2. เตรียมข้อมูลเหล็กเสริม
    bar_areas = {'RB6':0.28, 'RB9':0.64, 'DB10':0.79, 'DB12':1.13, 'DB16':2.01, 'DB20':3.14, 'DB25':4.91, 'DB28':6.16}
    stir_areas = {'RB6':0.28, 'RB9':0.64, 'DB10':0.79, 'DB12':1.13}
    m_area = bar_areas.get(m_bar, 1.13)
    s_area = stir_areas.get(s_bar, 0.28)
    
    # 3. วนลูปคำนวณแต่ละ Span
    n_span = len(vis_spans)
    span_results = []
    
    for i in range(n_span):
        # ดึงค่า Force จาก Analysis Dataframe
        span_df = df[df['span_id'] == i]
        if span_df.empty: continue
        
        m_max_pos = span_df['moment'].max()
        m_max_neg = span_df['moment'].min()
        v_max = span_df['shear'].abs().max()
        
        # Design Logic (เรียก rc_design)
        res_bot = calculate_rc_design(m_max_pos, v_max, fc, fy, b_cm, h_cm, cov_cm, method, unit_sys, m_area, s_area, man_s_cm) if m_max_pos > 0.01 else {'nb': 0, 'logs': []}
        res_top = calculate_rc_design(abs(m_max_neg), v_max, fc, fy, b_cm, h_cm, cov_cm, method, unit_sys, m_area, s_area, man_s_cm) if m_max_neg < -0.01 else {'nb': 0, 'logs': []}
        res_stir = calculate_rc_design(max(abs(m_max_pos), abs(m_max_neg)), v_max, fc, fy, b_cm, h_cm, cov_cm, method, unit_sys, m_area, s_area, man_s_cm)

        span_results.append({
            'id': i+1,
            'Mu_pos': m_max_pos, 'Mu_neg': m_max_neg, 'Vu': v_max,
            'bot_nb': res_bot.get('nb', 0),
            'top_nb': res_top.get('nb', 0),
            'stir_text': res_stir.get('stirrup_text', 'Err'),
            's_val': res_stir.get('s_value_mm', 0),
            'logs_bot': res_bot.get('logs', []),
            'logs_top': res_top.get('logs', []),
            'logs_stir': res_stir.get('logs', [])
        })

    # 4. แสดงผลกราฟ Profile (รวมทั้งคาน)
    st.plotly_chart(draw_reinforcement_profile(vis_spans, span_results, m_bar, s_bar), use_container_width=True, key="profile_plot")

    # 5. แสดงรายละเอียดแต่ละ Span (Tabs)
    st.markdown("#### 🔍 Section Details & Calculations")
    tabs = st.tabs([f"Span {r['id']}" for r in span_results])
    
    for i, tab in enumerate(tabs):
        r = span_results[i]
        with tab:
            c1, c2 = st.columns([1, 2])
            # Draw Section
            with c1: 
                main_d = int(m_bar[2:])
                stir_d = int(s_bar[2:]) if 'DB' in s_bar else int(s_bar[2:])
                st.plotly_chart(draw_section_real(
                    b_cm, h_cm, cov_cm, 
                    r['bot_nb'], r['top_nb'],  
                    main_d, stir_d, m_bar, s_bar, r['s_val'], 
                    f"Section Span {r['id']}"
                ), use_container_width=True)
                
                st.info(f"**Forces Span {i+1}:**\n\n- Mu(+) = {r['Mu_pos']:.2f}\n- Mu(-) = {r['Mu_neg']:.2f}\n- Vu max = {r['Vu']:.2f}")

            # Logs Detail
            with c2:
                with st.expander(f"👇 Bottom Steel Calculation (+M)", expanded=True):
                    if r['bot_nb'] > 0:
                        for l in r['logs_bot']: st.write(l)
                    else: st.write("No positive moment significant enough for design.")
                
                with st.expander(f"👆 Top Steel Calculation (-M)", expanded=False):
                    if r['top_nb'] > 0:
                        for l in r['logs_top']: st.write(l)
                    else: st.write("No negative moment significant enough for design.")
                        
                with st.expander(f"⛓️ Shear Design", expanded=False):
                     for l in r['logs_stir']: st.write(l)
