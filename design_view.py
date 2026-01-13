import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
import streamlit as st

# ==========================================
# 1. HELPER: LOAD TABLE (ตาราง Load Combination)
# ==========================================
def render_load_table(params):
    """
    แสดงตาราง Load Combination แยกออกมาเพื่อให้เรียกใช้ง่ายและจัด layout ได้สวยงาม
    """
    st.markdown("### 📋 Design Load Parameters")
    
    dl_f = params.get('dl_factor', 1.4)
    ll_f = params.get('ll_factor', 1.7)
    inc_sw = params.get('include_sw', True)
    
    data = [
        {
            "Load Case": "Dead Load (DL)", 
            "Factor": f"{dl_f:.2f}", 
            "Description": "Superimposed Dead Load"
        },
        {
            "Load Case": "Live Load (LL)", 
            "Factor": f"{ll_f:.2f}", 
            "Description": "Live Load (Occupancy)"
        }
    ]
    
    if inc_sw:
        data.insert(0, {
            "Load Case": "Self-Weight (SW)", 
            "Factor": f"{dl_f:.2f}", 
            "Description": "Beam Self-Weight (approx. 2400 kg/m³)"
        })
        
    df = pd.DataFrame(data)
    
    st.dataframe(
        df, 
        use_container_width=True, 
        hide_index=True,
        column_config={
            "Load Case": st.column_config.TextColumn("Case", width="small"),
            "Factor": st.column_config.TextColumn("Safety Factor", width="small"),
            "Description": st.column_config.TextColumn("Detail", width="large")
        }
    )
    st.caption(f"ℹ️ **Ultimate Load Equation:** U = {dl_f}DL + {ll_f}LL")
    st.divider()

# ==========================================
# 2. PLOTLY ANALYSIS GRAPH (WORLD-CLASS TEXTBOOK STYLE)
# ==========================================
def plot_analysis_results(res_df, spans, supports, loads, reactions):
    """
    World-Class Engineering FBD:
    1. All loads touch the beam (y=0).
    2. High contrast between Point Loads (Thick) and UDL (Thin/Transparent).
    3. Smart label positioning to avoid overlap.
    """
    fig = make_subplots(
        rows=4, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.08,
        subplot_titles=(
            "<b>1. Free Body Diagram (FBD)</b>", 
            "<b>2. Shear Force Diagram (SFD)</b>", 
            "<b>3. Bending Moment Diagram (BMD)</b>",
            "<b>4. Deflection Diagram</b>"
        ),
        row_heights=[0.3, 0.23, 0.23, 0.24]
    )

    # --- ROW 1: FREE BODY DIAGRAM ---
    total_L = sum(spans)
    cum_dist = [0] + list(np.cumsum(spans))
    
    # 1.1 Beam Line (Thick Black Line at y=0)
    fig.add_trace(go.Scatter(
        x=[0, total_L], y=[0, 0], 
        mode='lines', line=dict(color='black', width=6), 
        hoverinfo='skip'
    ), row=1, col=1)
    
    # 1.2 Supports
    for idx, row in supports.iterrows():
        sym = "triangle-up"
        if row['type'] == 'Fixed': sym = "square"
        elif row['type'] == 'Roller': sym = "circle"
        
        fig.add_trace(go.Scatter(
            x=[row['x']], y=[-0.05], # Support อยู่ใต้คานเล็กน้อย
            mode='markers+text',
            marker=dict(symbol=sym, size=14, color='white', line=dict(width=2, color='black')),
            text=[row['type'][0]], textposition="bottom center",
            hoverinfo='name', name="Support"
        ), row=1, col=1)

    # 1.3 Load Data Preparation (Robust Conversion)
    if isinstance(loads, pd.DataFrame):
        load_list = loads.to_dict('records')
    elif isinstance(loads, list):
        load_list = loads
    else:
        load_list = []
        
    # Visual Constants (Pixel Based for consistency)
    UDL_HEIGHT_VISUAL = 0.5   # ความสูงกราฟิกของ UDL
    ARROW_LEN_P = 70          # ความยาวลูกศร Point Load (ยาวและเด่น)
    ARROW_LEN_U = 35          # ความยาวลูกศร UDL (สั้นและบาง)

    # --- LAYER 1: UDL (Background Context) ---
    # วาด UDL ก่อน เพื่อให้เป็นพื้นหลัง
    for l in load_list:
        if l['type'] == 'U':
            span_idx = int(l['span_index'])
            start_x = cum_dist[span_idx] + float(l.get('d_start', 0))
            dist_val = float(l['dist'])
            end_x = start_x + dist_val
            mag_label = l['mag'] / 1000.0
            
            # Color Logic: Live Load = Red, Dead/SW = Blue
            color = '#e74c3c' if l.get('case') == 'LL' else '#2980b9'
            
            # 1. Fill Area (จางๆ)
            fig.add_trace(go.Scatter(
                x=[start_x, end_x, end_x, start_x], 
                y=[0, 0, UDL_HEIGHT_VISUAL, UDL_HEIGHT_VISUAL],
                fill='toself', fillcolor=color, opacity=0.1, line=dict(width=0),
                hoverinfo='skip', showlegend=False
            ), row=1, col=1)
            
            # 2. Top Line (เส้นขอบบน)
            fig.add_trace(go.Scatter(
                x=[start_x, end_x], y=[UDL_HEIGHT_VISUAL, UDL_HEIGHT_VISUAL],
                mode='lines', line=dict(color=color, width=1, dash='dot'), hoverinfo='skip'
            ), row=1, col=1)
            
            # 3. Small Arrows (ชี้ลงมาที่ y=0)
            n_arrows = max(3, int(dist_val * 2.5)) # จำนวนลูกศรถี่หน่อยให้ดูเป็นแผง
            for ax_x in np.linspace(start_x, end_x, n_arrows + 2)[1:-1]:
                fig.add_annotation(
                    x=ax_x, y=0, 
                    ax=0, ay=-ARROW_LEN_U, 
                    ayref='pixel', xref="x1", yref="y1",
                    showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1, arrowcolor=color,
                    row=1, col=1
                )
            
            # 4. Label (วางตรงกลาง สูงเหนือกล่องเล็กน้อย)
            label_txt = f"w={mag_label:.2f}"
            if l.get('case') == 'SW': label_txt = f"SW={mag_label:.2f}"
            fig.add_annotation(
                x=(start_x+end_x)/2, y=UDL_HEIGHT_VISUAL, 
                text=label_txt, showarrow=False, yshift=10, 
                font=dict(color=color, size=9), row=1, col=1
            )

    # --- LAYER 2: POINT LOAD (Foreground Emphasis) ---
    # วาด Point Load ทีหลัง ทับลงไปเลย
    for l in load_list:
        if l['type'] == 'P':
            span_idx = int(l['span_index'])
            x_loc = cum_dist[span_idx] + float(l['d_start'])
            mag_label = l['mag'] / 1000.0
            color = '#c0392b' if l.get('case') == 'LL' else '#2980b9'
            
            # ตรวจสอบว่าจุดนี้มี UDL ซ้อนทับไหม? (แบบง่าย) เพื่อดัน Label หนี
            # ถ้ามี UDL เราจะดัน Label สูงขึ้นไปอีก
            is_overlap_udl = False
            for ul in load_list:
                if ul['type'] == 'U':
                    u_start = cum_dist[int(ul['span_index'])] + float(ul.get('d_start', 0))
                    u_end = u_start + float(ul['dist'])
                    if u_start <= x_loc <= u_end:
                        is_overlap_udl = True
                        break
            
            text_yshift = ARROW_LEN_P + 10
            if is_overlap_udl:
                text_yshift += 25 # ดันขึ้นอีกถ้าชนกับ UDL Label
            
            # วาดลูกศร Point Load (ใหญ่ หนา เด่น)
            fig.add_annotation(
                x=x_loc, 
                y=0, # แตะคานเป๊ะ
                ax=0, ay=-ARROW_LEN_P, 
                ayref='pixel', xref="x1", yref="y1",
                showarrow=True, arrowhead=2, arrowsize=1.2, arrowwidth=3, arrowcolor=color, # หนา 3px
                text=f"<b>P={mag_label:.2f}</b>", 
                yshift=text_yshift, 
                font=dict(color=color, size=11, family="Arial Black"), 
                row=1, col=1
            )

    # --- ROW 2-4: DIAGRAMS (Standard) ---
    # SFD
    fig.add_hline(y=0, line_color="black", line_width=1, row=2, col=1)
    fig.add_trace(go.Scatter(
        x=res_df['x'], y=res_df['shear']/1000, 
        mode='lines', name='Shear', line=dict(color='#e74c3c', width=2),
        fill='tozeroy', fillcolor='rgba(231, 76, 60, 0.1)'
    ), row=2, col=1)
    
    # SFD Labels
    v_vals = res_df['shear']/1000
    for val in [v_vals.max(), v_vals.min()]:
        if abs(val) > 0.01:
            idx = (v_vals - val).abs().idxmin()
            fig.add_annotation(x=res_df['x'].iloc[idx], y=val, text=f"<b>{val:.2f}</b>", showarrow=False, yshift=10 if val>0 else -10, font=dict(color='#e74c3c', size=11), row=2, col=1)

    # BMD
    fig.add_hline(y=0, line_color="black", line_width=1, row=3, col=1)
    fig.add_trace(go.Scatter(
        x=res_df['x'], y=res_df['moment']/1000, 
        mode='lines', name='Moment', line=dict(color='#27ae60', width=2),
        fill='tozeroy', fillcolor='rgba(39, 174, 96, 0.1)'
    ), row=3, col=1)
    
    # BMD Labels
    m_vals = res_df['moment']/1000
    for val in [m_vals.max(), m_vals.min()]:
        if abs(val) > 0.01:
            idx = (m_vals - val).abs().idxmin()
            fig.add_annotation(x=res_df['x'].iloc[idx], y=val, text=f"<b>{val:.2f}</b>", showarrow=True, arrowhead=1, ay=20 if val>0 else -20, font=dict(color='#27ae60', size=11), row=3, col=1)

    # Deflection
    fig.add_hline(y=0, line_color="black", line_width=1, row=4, col=1)
    fig.add_trace(go.Scatter(
        x=res_df['x'], y=res_df['deflection'], 
        mode='lines', name='Deflection', line=dict(color='#8e44ad', width=2)
    ), row=4, col=1)
    
    # Deflection Labels
    if not res_df['deflection'].empty:
        idx_max_def = res_df['deflection'].abs().idxmax()
        max_def = res_df['deflection'].iloc[idx_max_def]
        if abs(max_def) > 0.001:
             fig.add_annotation(x=res_df['x'].iloc[idx_max_def], y=max_def, text=f"<b>Max: {max_def:.2f} mm</b>", showarrow=True, arrowhead=1, ay=30 if max_def < 0 else -30, font=dict(color='#8e44ad', size=11), row=4, col=1)

    # --- LAYOUT SETTINGS ---
    for x_pos in cum_dist:
        fig.add_vline(x=x_pos, line_width=1, line_dash="dash", line_color="gray", opacity=0.3)

    fig.update_layout(
        height=1000, showlegend=False, template="plotly_white", hovermode="x unified",
        margin=dict(t=50, b=40, l=60, r=20)
    )
    
    # Lock Scale for FBD Headroom (ให้ที่ว่างด้านบนเยอะหน่อยสำหรับลูกศร)
    fig.update_yaxes(range=[-0.5, 2.0], showgrid=False, visible=False, row=1, col=1)
    
    fig.update_yaxes(title_text="Shear (kN)", showgrid=True, row=2, col=1)
    fig.update_yaxes(title_text="Moment (kNm)", autorange="reversed", showgrid=True, row=3, col=1)
    fig.update_yaxes(title_text="Def. (mm)", showgrid=True, row=4, col=1)
    fig.update_xaxes(title_text="Length (m)", row=4, col=1)

    return fig
