import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- IMPORT MODULES ---
try:
    from beam_analysis import BeamFiniteElement
    from rc_design import calculate_rc_design
    import input_handler as ui
except ImportError:
    st.error("⚠️ Missing required files.")
    st.stop()

# ==========================================
# 0. SETUP & GRAPHICS FUNCTIONS
# ==========================================
st.set_page_config(page_title="RC Beam Pro V.Final", layout="wide", page_icon="🏗️")

if 'analyzed' not in st.session_state: st.session_state['analyzed'] = False
if 'res_df' not in st.session_state: st.session_state['res_df'] = None
if 'vis_data' not in st.session_state: st.session_state['vis_data'] = None

# CSS Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Sarabun:wght@300;400;600&display=swap');
    html, body, [class*="css"] { font-family: 'Sarabun', sans-serif; }
    .main-header { background: linear-gradient(90deg, #1565C0 0%, #0D47A1 100%); color: white; padding: 15px; border-radius: 10px; text-align: center; margin-bottom: 20px; }
    .section-header { font-size: 1.3rem; font-weight: bold; color: #1565C0; border-bottom: 2px solid #1565C0; padding-bottom: 5px; margin-top: 30px; margin-bottom: 15px; }
    .design-box { background-color: #e3f2fd; border: 2px solid #90caf9; padding: 20px; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

def draw_beam_diagram(spans, supports, loads):
    total_len = sum(spans)
    cum_len = [0] + list(np.cumsum(spans))
    fig = go.Figure()
    
    # 1. Main Beam Line (แก้ชื่อ Trace เป็น Beam และปรับเส้น)
    fig.add_trace(go.Scatter(
        x=[0, total_len], y=[0, 0], 
        mode='lines', 
        name='Beam',  # เปลี่ยนชื่อจาก Trace เป็น Beam
        line=dict(color='black', width=6), 
        hoverinfo='name+x'
    ))

    # 2. Supports
    for i, s in enumerate(supports):
        sx = cum_len[i]
        if s == "None": continue
        
        if s == "Fix": sym, col, off, txt = "square", "#D32F2F", 0, "Fix"
        elif s == "Pin": sym, col, off, txt = "triangle-up", "#2E7D32", -0.02, "Pin"
        else: sym, col, off, txt = "circle", "#F57C00", -0.02, "Roller"

        fig.add_trace(go.Scatter(
            x=[sx], y=[off], mode='markers',
            name=txt,
            marker=dict(symbol=sym, size=16, color=col, line=dict(width=2,color='black')),
            hovertext=f"{txt} @ {sx}m", hoverinfo="text", showlegend=False
        ))
        
        # Ground lines
        if s in ["Roller", "Pin"]:
             fig.add_shape(type="line", x0=sx-0.2, y0=off-0.04, x1=sx+0.2, y1=off-0.04, line=dict(color="black", width=2))
             for dash in range(3):
                 fig.add_shape(type="line", x0=sx-0.15 + dash*0.1, y0=off-0.04, x1=sx-0.2 + dash*0.1, y1=off-0.07, line=dict(color="black", width=1))

    # 3. Loads
    max_h = 0.3
    for load in loads:
        start_x = cum_len[load['span_idx']]
        val = load['display_val']
        
        if load['type'] == 'Uniform':
            end_x = start_x + spans[load['span_idx']]
            fig.add_shape(type="rect", x0=start_x, y0=0, x1=end_x, y1=max_h, 
                          line=dict(width=0), fillcolor="rgba(255, 87, 34, 0.2)")
            fig.add_trace(go.Scatter(x=[start_x, end_x], y=[max_h, max_h], mode='lines', 
                                     name="Load", line=dict(color='#E64A19', width=2), showlegend=False))
            
            mid = (start_x + end_x)/2
            fig.add_annotation(x=mid, y=max_h, text=f"w={val:.2f}", showarrow=True, arrowhead=0, ax=0, ay=-20, font=dict(color="#bf360c"))
            
        elif load['type'] == 'Point':
            lx = start_x + load['pos']
            fig.add_annotation(x=lx, y=0, ax=0, ay=-50, text=f"P={val:.2f}", 
                               showarrow=True, arrowhead=2, arrowwidth=2, arrowcolor="#D32F2F", 
                               font=dict(color="#D32F2F", weight="bold"))
            fig.add_annotation(x=lx, y=0, text=f"x={lx:.2f}", showarrow=False, yshift=-10, font=dict(size=10, color="gray"))

    # 4. Dimensions
    for i, sp in enumerate(spans):
        mid = cum_len[i] + sp/2
        fig.add_annotation(x=mid, y=-0.15, text=f"<b>L={sp}m</b>", showarrow=False, font=dict(color="#1565C0", size=14))
        fig.add_shape(type="line", x0=cum_len[i], y0=-0.1, x1=cum_len[i], y1=-0.2, line=dict(color="gray", dash="dot"))
        fig.add_shape(type="line", x0=cum_len[i+1], y0=-0.1, x1=cum_len[i+1], y1=-0.2, line=dict(color="gray", dash="dot"))

    # Layout Adjustment (เพิ่ม Padding แกน X ไม่ให้เส้นหาย)
    fig.update_layout(height=280, title_text="Structure Model",
                      margin=dict(l=30,r=30,t=40,b=20), 
                      yaxis=dict(visible=False, range=[-0.3, 0.5]), 
                      xaxis=dict(title="Distance (m)", range=[-0.5, total_len + 0.5]), # เผื่อขอบซ้ายขวา
                      plot_bgcolor='white')
    return fig

def draw_section_beautiful(b, h, cov, nb, bd_mm, sd_mm, main_name, stir_name):
    fig = go.Figure()
    bd = bd_mm / 10 
    sd = sd_mm / 10 
    
    # Concrete
    fig.add_shape(type="rect", x0=0, y0=0, x1=b, y1=h, line=dict(color="black", width=3), fillcolor="#EEEEEE")
    
    # Stirrup
    inset = cov
    path_x = [inset, b-inset, b-inset, inset, inset]
    path_y = [inset, inset, h-inset, h-inset, inset]
    fig.add_trace(go.Scatter(x=path_x, y=path_y, mode='lines', 
                             line=dict(color="#C62828", width=3), name="Stirrup", hoverinfo='skip'))
    
    # Main Bars
    if nb > 0:
        eff_w = b - 2*cov - 2*sd - bd
        gap = eff_w / (nb - 1) if nb > 1 else 0
        y_pos = cov + sd + bd/2
        x_positions = [cov + sd + bd/2 + i*gap for i in range(nb)]
        
        fig.add_trace(go.Scatter(
            x=x_positions, y=[y_pos]*nb, mode='markers',
            marker=dict(size=bd_mm*1.8, color='#1565C0', line=dict(width=2, color='black')),
            name="Main Bar"
        ))
        # Hanger Bars
        y_top = h - (cov + sd + bd/2)
        fig.add_trace(go.Scatter(
            x=[cov + sd + bd/2, b - (cov + sd + bd/2)], y=[y_top, y_top], mode='markers',
            marker=dict(size=bd_mm*1.0, color='#90A4AE', line=dict(width=1, color='black')),
            showlegend=False, hoverinfo='skip'
        ))

    # Dimensions
    fig.add_annotation(x=b/2, y=-4, text=f"b = {b} cm", showarrow=False, font=dict(size=14))
    fig.add_annotation(x=-4, y=h/2, text=f"h = {h} cm", textangle=-90, showarrow=False, font=dict(size=14))
    
    fig.add_annotation(x=b/2, y=y_pos, ax=50, ay=40, text=f"<b>{nb}-{main_name}</b>", 
                       arrowcolor="#1565C0", arrowhead=2, font=dict(color="#1565C0", size=14))
    fig.add_annotation(x=b, y=h/2, ax=40, ay=0, text=f"Stirrup <b>{stir_name}</b>", 
                       arrowcolor="#C62828", arrowhead=2, font=dict(color="#C62828", size=12))

    fig.update_layout(
        title="Cross Section Detail",
        width=350, height=350,
        xaxis=dict(visible=False, range=[-10, b+15]), 
        yaxis=dict(visible=False, range=[-10, h+10]),
        margin=dict(l=10,r=10,t=40,b=10),
        plot_bgcolor='white', showlegend=False
    )
    return fig

# ==========================================
# 1. MAIN APPLICATION
# ==========================================
st.markdown('<div class="main-header"><h2>🏗️ RC Beam Pro V.Final</h2></div>', unsafe_allow_html=True)

# 1.1 Input Handling
design_code, method, fact_dl, fact_ll, unit_sys = ui.render_sidebar()

st.markdown('<div class="section-header">1️⃣ Structure & Loads</div>', unsafe_allow_html=True)
c_geo, c_load = st.columns([1, 1.5])
with c_geo:
    n_span, spans, supports = ui.render_geometry_input()
with c_load:
    loads_input = ui.render_loads_input(n_span, spans, fact_dl, fact_ll, unit_sys)

# 1.2 Analysis
st.markdown('<div class="section-header">2️⃣ Analysis (Finite Element)</div>', unsafe_allow_html=True)
if st.button("🚀 Calculate Analysis", type="primary"):
    solver = BeamFiniteElement(spans, supports, loads_input)
    success, msg = solver.solve()
    
    if success:
        st.session_state['res_df'] = solver.get_internal_forces()
        st.session_state['vis_data'] = (spans, supports, loads_input)
        st.session_state['analyzed'] = True
        st.rerun()
    else:
        st.error(f"Analysis Failed: {msg}")

# 1.3 Result & Design
if st.session_state['analyzed']:
    vis_data = st.session_state['vis_data']
    df = st.session_state['res_df'].copy()
    
    vis_spans = vis_data[0]
    total_len = sum(vis_spans)
    
    u_f, u_m = ("kN", "kN-m") if "kN" in unit_sys else ("kg", "kg-m")
    
    # 1. Draw Beam Diagram
    st.plotly_chart(draw_beam_diagram(*vis_data), use_container_width=True)
    
    # 2. Draw Graphs (Synchronized X-axis & Better Labels)
    v_max_idx, v_min_idx = df['shear'].idxmax(), df['shear'].idxmin()
    m_max_idx, m_min_idx = df['moment'].idxmax(), df['moment'].idxmin()
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                        subplot_titles=(f"Shear Force Diagram ({u_f})", f"Bending Moment Diagram ({u_m})"))
    
    fig.add_trace(go.Scatter(x=df['x'], y=df['shear'], fill='tozeroy', line=dict(color='#E53935'), name="Shear"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['x'], y=df['moment'], fill='tozeroy', line=dict(color='#1E88E5'), name="Moment"), row=2, col=1)
    
    # Add Markers & Smart Annotations (ป้องกันเลขตกขอบ)
    for col, data, idx_list in [(1, df['shear'], [v_max_idx, v_min_idx]), (2, df['moment'], [m_max_idx, m_min_idx])]:
        for idx in idx_list:
            x_val = df.iloc[idx]['x']
            y_val = df.iloc[idx]['shear'] if col==1 else df.iloc[idx]['moment']
            
            # Smart Text Position
            if x_val < total_len * 0.1: text_pos = "top right"
            elif x_val > total_len * 0.9: text_pos = "top left"
            else: text_pos = "top center"

            fig.add_trace(go.Scatter(x=[x_val], y=[y_val], mode='markers+text',
                                     marker=dict(color='black', size=8),
                                     text=[f"{y_val:.2f}"], textposition=text_pos,
                                     name="Max/Min", showlegend=False), row=col, col=1)

    # Force X-axis range (เพิ่มระยะขอบซ้ายขวาเล็กน้อย -0.5 ถึง L+0.5)
    fig.update_xaxes(range=[-0.5, total_len + 0.5], showgrid=True)
    
    # ปรับ Margin กราฟให้กว้างขึ้น เพื่อให้ Text ไม่โดนตัด
    fig.update_layout(height=500, margin=dict(l=50, r=50, t=40, b=40), hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    # 1.4 Design Section
    st.markdown('<div class="section-header">3️⃣ RC Design & Detailing</div>', unsafe_allow_html=True)
    
    max_M = df['moment'].abs().max()
    max_V = df['shear'].abs().max()

    fc, fy, b, h, cov, m_bar, s_bar = ui.render_design_input(unit_sys)
    
    bar_areas = {'DB12':1.13, 'DB16':2.01, 'DB20':3.14, 'DB25':4.91, 'DB28':6.16}
    bar_dias = {k: int(k[2:]) for k in bar_areas}
    stir_areas = {'RB6':0.28, 'RB9':0.64, 'DB10':0.78}
    stir_dias = {'RB6':6, 'RB9':9, 'DB10':10}

    res = calculate_rc_design(max_M, max_V, fc, fy, b, h, cov, method, unit_sys, bar_areas[m_bar], stir_areas[s_bar])
    
    st.markdown('<div class="design-box">', unsafe_allow_html=True)
    c_res, c_draw = st.columns([1.2, 1])
    with c_res:
        st.subheader("📝 Calculation Result")
        st.info(f"Design Load: Mu = {max_M:.2f} {u_m}, Vu = {max_V:.2f} {u_f}")
        
        if res['nb'] == 0: st.error(res['msg_flex'])
        else:
            st.success(f"**Flexure:** {res['msg_flex']}")
            st.markdown(f"""
            * **Required Main Steel:** `{res['nb']} - {m_bar}`
            * Area Required: {res['As_req']:.2f} cm²
            * Area Provided: {res['nb']*bar_areas[m_bar]:.2f} cm²
            """)
            st.divider()
            shear_color = "orange" if "Shear Reinf" in res['msg_shear'] else "green"
            st.markdown(f":{shear_color}[**Shear:** {res['msg_shear']}]")
            st.markdown(f"* **Stirrups:** `{s_bar} {res['stirrup_text']}`")

            with st.expander("🔎 View Full Calculation Log"):
                 st.code("\n".join(res['logs']))

    with c_draw:
        fig_sec = draw_section_beautiful(b, h, cov, res['nb'], bar_dias[m_bar], stir_dias[s_bar], m_bar, s_bar)
        st.plotly_chart(fig_sec, use_container_width=True)
        
    st.markdown('</div>', unsafe_allow_html=True)
