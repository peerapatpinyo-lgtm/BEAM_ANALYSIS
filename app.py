import streamlit as st
import pandas as pd
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
# 0. SETUP
# ==========================================
st.set_page_config(page_title="RC Beam Pro Ultimate", layout="wide", page_icon="🏗️")

if 'analyzed' not in st.session_state: st.session_state['analyzed'] = False
if 'res_df' not in st.session_state: st.session_state['res_df'] = None
if 'vis_data' not in st.session_state: st.session_state['vis_data'] = None

# CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Sarabun:wght@300;400;600&display=swap');
    html, body, [class*="css"] { font-family: 'Sarabun', sans-serif; }
    .main-header { background: linear-gradient(90deg, #2c3e50 0%, #3498db 100%); color: white; padding: 15px; border-radius: 8px; text-align: center; margin-bottom: 20px; }
    .section-header { font-size: 1.4rem; font-weight: bold; color: #2c3e50; border-bottom: 2px solid #bdc3c7; padding-bottom: 5px; margin-top: 30px; margin-bottom: 15px; }
    h4 { color: #1abc9c; }
</style>
""", unsafe_allow_html=True)

# --- GRAPHIC FUNCTIONS ---

def draw_reinforcement_profile(spans, design_results, m_bar, s_bar):
    """ วาดรูปด้านข้างคาน เส้นเหล็กชัดเจน แยกสี """
    total_len = sum(spans)
    cum_len = [0] + list(np.cumsum(spans))
    fig = go.Figure()

    # 1. Concrete Body (Gray Fill)
    fig.add_shape(type="rect", x0=0, y0=-0.5, x1=total_len, y1=0.5, 
                  line=dict(color="black", width=2), fillcolor="#ecf0f1", layer="below")
    
    # 2. Main Bars (Thick Lines)
    # เหล็กบน (Top Bar) - สีแดงเข้ม (Negative Moment Reinf)
    fig.add_trace(go.Scatter(x=[0, total_len], y=[0.35, 0.35], mode='lines', 
                             name="Top Reinf.", line=dict(color='#c0392b', width=5), hoverinfo='name'))
    
    # เหล็กล่าง (Bottom Bar) - สีน้ำเงินเข้ม (Positive Moment Reinf)
    fig.add_trace(go.Scatter(x=[0, total_len], y=[-0.35, -0.35], mode='lines', 
                             name="Bottom Reinf.", line=dict(color='#2980b9', width=5), hoverinfo='name'))

    # 3. Text Labels & Supports
    for i in range(len(cum_len)-1):
        x_start, x_end = cum_len[i], cum_len[i+1]
        mid = (x_start + x_end)/2
        res = design_results[i]
        
        # Text Label Box
        fig.add_annotation(
            x=mid, y=0,
            text=f"<b>SPAN {i+1}</b><br>Bot: {res['nb']}-{m_bar}<br>Stir: {s_bar}@{res['s_value']}",
            bgcolor="rgba(255,255,255,0.8)", bordercolor="#34495e", borderwidth=1,
            font=dict(size=12, color="black")
        )
    
    # Support Triangles
    for x in cum_len:
         fig.add_trace(go.Scatter(x=[x], y=[-0.6], mode='markers', 
                                  marker=dict(symbol="triangle-up", size=15, color="#2c3e50"), 
                                  showlegend=False, hoverinfo='skip'))

    fig.update_layout(
        title="🏗️ Reinforcement Profile (Side View)", 
        height=250, 
        yaxis=dict(visible=False, range=[-1.2, 1.2]), 
        xaxis=dict(title="Distance (m)", showgrid=True, dtick=1, range=[-0.2, total_len+0.2]),
        margin=dict(l=20,r=20,t=40,b=20), plot_bgcolor='white', showlegend=True,
        legend=dict(orientation="h", y=1.1, x=0.5, xanchor='center')
    )
    return fig

def draw_section_engineering(b_cm, h_cm, cov_cm, nb, bd_mm, stir_d_mm, main_name, stir_name, s_val, title):
    """ วาด Cross Section แบบวิศวะ (เส้นปลอก + วงกลมเหล็ก) """
    fig = go.Figure()
    
    # Dimensions in Drawing Units (Let's use cm for coordinates)
    # Convert rebar dia to cm
    bd = bd_mm / 10.0
    sd = stir_d_mm / 10.0
    
    # 1. Concrete Section (Outline)
    fig.add_shape(type="rect", x0=0, y0=0, x1=b_cm, y1=h_cm, 
                  line=dict(color="black", width=3), fillcolor="#f7f9f9")
    
    # 2. Stirrup (Continuous Line Loop) - "The Red Line"
    # Offset by cover
    sx0, sy0 = cov_cm, cov_cm
    sx1, sy1 = b_cm - cov_cm, h_cm - cov_cm
    
    # Draw Stirrup as a path/shape
    fig.add_shape(type="rect", x0=sx0, y0=sy0, x1=sx1, y1=sy1,
                  line=dict(color="#c0392b", width=3), fillcolor="rgba(0,0,0,0)") # Transparent fill
    
    # 3. Main Bars (Solid Circles with Outline)
    if nb > 0:
        eff_w = (b_cm - 2*cov_cm - 2*sd - bd)
        gap = eff_w / (nb - 1) if nb > 1 else 0
        y_pos = cov_cm + sd + bd/2
        
        # Generate X positions
        xs = [cov_cm + sd + bd/2 + i*gap for i in range(nb)]
        
        fig.add_trace(go.Scatter(
            x=xs, y=[y_pos]*nb, mode='markers',
            marker=dict(size=bd_mm*1.5, color='#2980b9', line=dict(width=2, color='black')),
            name="Main Bar", showlegend=False
        ))
        
    # 4. Hanger Bars (Dummy Top Bars - usually 2)
    y_top = h_cm - (cov_cm + sd + bd/2)
    fig.add_trace(go.Scatter(
        x=[cov_cm+sd+bd/2, b_cm-(cov_cm+sd+bd/2)], y=[y_top, y_top], mode='markers',
        marker=dict(size=bd_mm*1.0, color='#95a5a6', line=dict(width=1, color='black')),
        name="Hanger Bar", showlegend=False
    ))

    # Annotations (Dimensions)
    fig.add_annotation(x=b_cm/2, y=-h_cm*0.1, text=f"b = {b_cm*10:.0f} mm", showarrow=False)
    fig.add_annotation(x=-b_cm*0.15, y=h_cm/2, text=f"h = {h_cm*10:.0f} mm", textangle=-90, showarrow=False)
    
    # Stirrup Label
    fig.add_annotation(
        x=b_cm/2, y=h_cm/2, 
        text=f"<b>{stir_name} @ {s_val} cm</b>", 
        font=dict(color="#c0392b", size=14), bgcolor="white"
    )

    fig.update_layout(
        title=dict(text=title, x=0.5), width=300, height=350, 
        xaxis=dict(visible=False, range=[-b_cm*0.3, b_cm*1.3]), 
        yaxis=dict(visible=False, range=[-h_cm*0.2, h_cm*1.2]),
        margin=dict(l=10,r=10,t=40,b=10), plot_bgcolor='white'
    )
    return fig

# ==========================================
# 1. MAIN APPLICATION
# ==========================================
st.markdown('<div class="main-header"><h2>🏗️ RC Beam Pro Ultimate</h2></div>', unsafe_allow_html=True)

# 1.1 Inputs
design_code, method, fact_dl, fact_ll, unit_sys = ui.render_sidebar()
c_geo, c_load = st.columns([1, 1.5])
with c_geo: n_span, spans, supports = ui.render_geometry_input()
with c_load: loads_input = ui.render_loads_input(n_span, spans, fact_dl, fact_ll, unit_sys)

# 1.2 Analysis Button
if st.button("🚀 Calculate Analysis & Design", type="primary"):
    solver = BeamFiniteElement(spans, supports, loads_input)
    success, msg = solver.solve()
    if success:
        st.session_state['res_df'] = solver.get_internal_forces()
        st.session_state['vis_data'] = (spans, supports, loads_input)
        st.session_state['analyzed'] = True
    else:
        st.error(f"Analysis Failed: {msg}")

# 1.3 Results
if st.session_state['analyzed']:
    vis_data = st.session_state['vis_data']
    df = st.session_state['res_df'].copy()
    vis_spans = vis_data[0]
    total_len = sum(vis_spans)
    cum_len = [0] + list(np.cumsum(vis_spans))
    
    u_force = "kN" if "kN" in unit_sys else "kg"
    u_moment = "kN-m" if "kN" in unit_sys else "kg-m"

    # --- PART 1: ADVANCED GRAPHS ---
    st.markdown('<div class="section-header">1️⃣ Analysis Results</div>', unsafe_allow_html=True)
    
    # ใช้ make_subplots พร้อมเพิ่ม vertical_spacing
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, 
        vertical_spacing=0.1,  # เพิ่มระยะห่าง กันตัวหนังสือซ้อน
        subplot_titles=("Beam Model", "", ""), # ไม่ใส่ Title ในนี้ เดี๋ยวใส่ที่แกนแทน
        row_heights=[0.15, 0.425, 0.425]
    )

    # Graph 1: Beam Model (วาดเป็นแท่ง Rectangle แทนเส้น)
    fig.add_shape(type="rect", x0=0, y0=-0.1, x1=total_len, y1=0.1, 
                  fillcolor="lightgray", line=dict(color="black", width=2), row=1, col=1)
    
    # Add Supports Markers
    for i, s in enumerate(supports):
        if s != "None":
            sx = cum_len[i]
            fig.add_trace(go.Scatter(x=[sx], y=[-0.15], mode='markers', 
                                     marker=dict(symbol="triangle-up", size=12, color="green"), 
                                     name="Support", hoverinfo='skip'), row=1, col=1)

    # Graph 2: Shear Force
    fig.add_trace(go.Scatter(x=df['x'], y=df['shear'], fill='tozeroy', 
                             line=dict(color='#e74c3c', width=2), name="Shear Force"), row=2, col=1)

    # Graph 3: Bending Moment
    fig.add_trace(go.Scatter(x=df['x'], y=df['moment'], fill='tozeroy', 
                             line=dict(color='#3498db', width=2), name="Bending Moment"), row=3, col=1)

    # Layout Adjustments
    fig.update_layout(
        height=800,  # เพิ่มความสูงรวม
        hovermode="x unified",
        plot_bgcolor='white',
        showlegend=False,
        margin=dict(l=60, r=20, t=20, b=40)
    )

    # X-Axis Settings (Ticks & Labels)
    tick_vals = sorted(list(set([0, total_len] + cum_len + list(np.arange(0, total_len+1, 1)))))
    fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True, row=1, col=1, visible=False) # ซ่อนแกน x รูปคาน
    fig.update_xaxes(showgrid=True, gridcolor='#eee', row=2, col=1)
    
    # แกน X กราฟล่างสุด (แสดงตัวเลขชัดเจน)
    fig.update_xaxes(
        title_text="Distance (m)", 
        showgrid=True, gridcolor='#eee', 
        tickmode='array', tickvals=tick_vals, # บังคับโชว์จุดสำคัญ
        showline=True, linewidth=1, linecolor='black', mirror=True,
        row=3, col=1
    )

    # Y-Axis Titles (ใส่ตรงนี้แทน Subplot Title จะได้ไม่ซ้อน)
    fig.update_yaxes(title_text="Beam", showgrid=False, zeroline=False, visible=False, row=1, col=1)
    fig.update_yaxes(title_text=f"Shear V<br>({u_force})", showgrid=True, gridcolor='#eee', zeroline=True, zerolinecolor='black', row=2, col=1)
    fig.update_yaxes(title_text=f"Moment M<br>({u_moment})", showgrid=True, gridcolor='#eee', zeroline=True, zerolinecolor='black', row=3, col=1)

    st.plotly_chart(fig, use_container_width=True)

    # --- PART 2: DESIGN & DETAILING ---
    st.markdown('<div class="section-header">2️⃣ Design & Detailing</div>', unsafe_allow_html=True)
    
    fc, fy, b_cm, h_cm, cov_cm, m_bar, s_bar, man_s = ui.render_design_input(unit_sys)
    
    # Database
    bar_areas = {'DB12':1.13, 'DB16':2.01, 'DB20':3.14, 'DB25':4.91, 'DB28':6.16}
    bar_dias_mm = {k: int(k[2:]) for k in bar_areas}
    stir_areas = {'RB6':0.28, 'RB9':0.64, 'DB10':0.78, 'DB12':1.13}
    stir_dias_mm = {k: (int(k[2:]) if 'DB' in k else int(k[2:])) for k in stir_areas}
    
    span_results = []
    for i in range(n_span):
        x_start, x_end = cum_len[i], cum_len[i+1]
        span_df = df[(df['x'] >= x_start) & (df['x'] <= x_end)]
        m_span = span_df['moment'].abs().max()
        v_span = span_df['shear'].abs().max()
        
        # Calculate Logic
        res = calculate_rc_design(m_span, v_span, fc, fy, b_cm, h_cm, cov_cm, method, unit_sys, bar_areas[m_bar], stir_areas[s_bar], man_s)
        res['id'] = i+1
        span_results.append(res)

    # 2.1 Profile Visualization
    st.plotly_chart(draw_reinforcement_profile(vis_spans, span_results, m_bar, s_bar), use_container_width=True)

    # 2.2 Detailed Calculation
    st.markdown("#### 📝 Detailed Section Design")
    tabs = st.tabs([f"Span {res['id']}" for res in span_results])
    
    for i, tab in enumerate(tabs):
        res = span_results[i]
        with tab:
            c1, c2 = st.columns([1, 1.5])
            with c1:
                st.markdown(f"**Section Span {res['id']}**")
                # เรียกฟังก์ชันวาดรูปแบบ Engineering
                fig_sec = draw_section_engineering(
                    b_cm, h_cm, cov_cm, res['nb'], 
                    bar_dias_mm[m_bar], stir_dias_mm[s_bar], 
                    m_bar, s_bar, res['s_value'], ""
                )
                st.plotly_chart(fig_sec, use_container_width=True, key=f"sec_{i}")
                
            with c2:
                with st.expander("Show Calculation Log", expanded=True):
                    for line in res['logs']:
                        if "###" in line: st.markdown(line)
                        elif "Conversion" in line or "Design Load" in line: st.info(line.replace(">", ""))
                        elif "❌" in line: st.error(line)
                        else: st.markdown(line)
