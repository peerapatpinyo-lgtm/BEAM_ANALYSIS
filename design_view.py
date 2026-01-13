import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
import streamlit as st
import math

# ==========================================
# 1. HELPER: LOAD TABLE (ตาราง Load Combination)
# ==========================================
def render_load_table(params):
    """
    แสดงตาราง Load Combination และ Safety Factors แบบมืออาชีพ
    """
    st.markdown("### 📋 Design Load Parameters")
    
    # ดึงค่าจาก params
    dl_f = params.get('dl_factor', 1.4)
    ll_f = params.get('ll_factor', 1.7)
    inc_sw = params.get('include_sw', True)
    
    # สร้างข้อมูลตาราง
    data = [
        {
            "Load Type": "Dead Load (DL)", 
            "Factor": f"{dl_f:.2f}", 
            "Description": "Superimposed Dead Load (น้ำหนักบรรทุกคงที่)"
        },
        {
            "Load Type": "Live Load (LL)", 
            "Factor": f"{ll_f:.2f}", 
            "Description": "Live Load (น้ำหนักบรรทุกจร)"
        }
    ]
    
    # เพิ่ม Self-weight
    if inc_sw:
        data.insert(0, {
            "Load Type": "Self-Weight (SW)", 
            "Factor": f"{dl_f:.2f}", 
            "Description": "Beam Self-Weight (น้ำหนักคาน 2400 kg/m³)"
        })
        
    df = pd.DataFrame(data)
    
    # แสดงผลตาราง
    st.dataframe(
        df, 
        use_container_width=True, 
        hide_index=True,
        column_config={
            "Load Type": st.column_config.TextColumn("Load Case", width="medium"),
            "Factor": st.column_config.TextColumn("Safety Factor", width="small"),
            "Description": st.column_config.TextColumn("Detail", width="large")
        }
    )
    
    # แสดงสมการ Load Combination
    eqn = f"**Design Load (U)** = {dl_f}DL + {ll_f}LL"
    st.info(f"ℹ️ {eqn}")
    st.divider()

# ==========================================
# 2. PLOTLY ANALYSIS GRAPH (PIXEL SCALING FIX)
# ==========================================
def plot_analysis_results(res_df, spans, supports, loads, reactions):
    """
    Uses Pixel-Based scaling for arrows to guarantee visual consistency.
    """
    # Create Subplots
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
    
    # 1.1 Beam Line (Thick Black Line)
    fig.add_trace(go.Scatter(
        x=[0, total_L], y=[0, 0], 
        mode='lines', line=dict(color='black', width=6), 
        hoverinfo='skip'
    ), row=1, col=1)
    
    # 1.2 Supports
    for idx, row in supports.iterrows():
        # Choose symbol based on type
        sym = "triangle-up"
        if row['type'] == 'Fixed': sym = "square"
        elif row['type'] == 'Roller': sym = "circle"
        
        fig.add_trace(go.Scatter(
            x=[row['x']], y=[-0.05], # วางใต้เส้นคานเล็กน้อย
            mode='markers+text',
            marker=dict(symbol=sym, size=14, color='white', line=dict(width=2, color='black')),
            text=[row['type'][0]], textposition="bottom center",
            hoverinfo='name', name="Support"
        ), row=1, col=1)

    # 1.3 Loads (The Critical Fix: Pixel Scaling)
    # เราจะไม่ใช้ค่าแกน Y จริงในการวาดความยาวลูกศร แต่จะใช้ "Pixel" หน้าจอแทน
    
    load_iter = loads if isinstance(loads, list) else []
    
    for l in load_iter:
        span_idx = int(l['span_index'])
        start_x_span = cum_dist[span_idx]
        mag_val = l['mag']
        mag_label = mag_val / 1000.0  # Convert to kN for label
        
        # Color coding
        case_type = l.get('case', 'DL')
        color = '#c0392b' if case_type == 'LL' else '#2980b9'  # Red for LL, Blue for DL/SW
        
        # --- POINT LOAD (P) ---
        if l['type'] == 'P':
            x_loc = start_x_span + float(l['d_start'])
            
            # ใช้ ayref='pixel' เพื่อกำหนดความยาวลูกศรเป็น Pixel (เช่น 50px)
            # ไม่ว่าค่า Load จะเป็น 100 หรือ 100000 ลูกศรจะยาวเท่ากัน
            fig.add_annotation(
                x=x_loc, y=0,
                ax=0, ay=-50,      # หางลูกศรอยู่สูงขึ้นไป 50 pixels
                ayref='pixel',     # สำคัญมาก! ใช้หน่วย Pixel
                xref="x1", yref="y1",
                showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor=color,
                text=f"<b>P={mag_label:.2f}</b>", 
                yshift=55,         # ขยับ Text ขึ้นไปเหนือหางลูกศร
                font=dict(color=color, size=10),
                row=1, col=1
            )
            
        # --- UNIFORM LOAD (U) ---
        elif l['type'] == 'U':
            x_s = start_x_span + float(l.get('d_start', 0))
            dist_val = float(l['dist'])
            x_e = x_s + dist_val
            
            # วาดสี่เหลี่ยมแทน UDL (ความสูงสมมติ 0.5 หน่วย เพื่อความสวยงาม)
            h_vis = 0.5 
            
            # เส้นบน
            fig.add_trace(go.Scatter(
                x=[x_s, x_e], y=[h_vis, h_vis],
                mode='lines', line=dict(color=color, width=1.5), hoverinfo='skip'
            ), row=1, col=1)
            
            # ถมสีจางๆ
            fig.add_trace(go.Scatter(
                x=[x_s, x_e, x_e, x_s], y=[0, 0, h_vis, h_vis],
                fill='toself', fillcolor=color, opacity=0.1, line=dict(width=0),
                hoverinfo='skip', showlegend=False
            ), row=1, col=1)
            
            # วาดลูกศรเล็กๆ เรียงกัน (ใช้ Pixel Scaling เช่นกัน)
            n_arrows = max(2, int(dist_val * 2.0)) # จำนวนลูกศรตามความยาว
            arrow_x_positions = np.linspace(x_s, x_e, n_arrows + 2)[1:-1]
            
            for ax_x in arrow_x_positions:
                fig.add_annotation(
                    x=ax_x, y=0, 
                    ax=0, ay=-30,  # ลูกศร UDL สั้นกว่า P (30px)
                    ayref='pixel',
                    xref="x1", yref="y1",
                    showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1, arrowcolor=color,
                    row=1, col=1
                )
            
            # Label ตรงกลาง
            label_txt = f"<b>w={mag_label:.2f}</b>"
            if case_type == 'SW': label_txt = f"SW={mag_label:.2f}"
            
            fig.add_annotation(
                x=(x_s+x_e)/2, y=h_vis,
                text=label_txt, showarrow=False, yshift=10,
                font=dict(color=color, size=10), row=1, col=1
            )

    # --- ROW 2: SHEAR (SFD) ---
    fig.add_hline(y=0, line_color="black", line_width=1, row=2, col=1)
    fig.add_trace(go.Scatter(
        x=res_df['x'], y=res_df['shear']/1000, 
        mode='lines', name='Shear', line=dict(color='#e74c3c', width=2),
        fill='tozeroy', fillcolor='rgba(231, 76, 60, 0.1)'
    ), row=2, col=1)
    
    # Max/Min Shear Labels
    v_vals = res_df['shear']/1000
    for val in [v_vals.max(), v_vals.min()]:
        if abs(val) > 0.01:
            idx = (v_vals - val).abs().idxmin()
            fig.add_annotation(
                x=res_df['x'].iloc[idx], y=val,
                text=f"<b>{val:.2f}</b>", showarrow=False, yshift=10 if val>0 else -10,
                font=dict(color='#e74c3c', size=11), row=2, col=1
            )

    # --- ROW 3: MOMENT (BMD) ---
    fig.add_hline(y=0, line_color="black", line_width=1, row=3, col=1)
    fig.add_trace(go.Scatter(
        x=res_df['x'], y=res_df['moment']/1000, 
        mode='lines', name='Moment', line=dict(color='#27ae60', width=2),
        fill='tozeroy', fillcolor='rgba(39, 174, 96, 0.1)'
    ), row=3, col=1)
    
    # Max/Min Moment Labels
    m_vals = res_df['moment']/1000
    for val in [m_vals.max(), m_vals.min()]:
        if abs(val) > 0.01:
            idx = (m_vals - val).abs().idxmin()
            fig.add_annotation(
                x=res_df['x'].iloc[idx], y=val,
                text=f"<b>{val:.2f}</b>", showarrow=True, arrowhead=1, ay=20 if val>0 else -20,
                font=dict(color='#27ae60', size=11), row=3, col=1
            )

    # --- ROW 4: DEFLECTION ---
    fig.add_hline(y=0, line_color="black", line_width=1, row=4, col=1)
    fig.add_trace(go.Scatter(
        x=res_df['x'], y=res_df['deflection'], 
        mode='lines', name='Deflection', line=dict(color='#8e44ad', width=2)
    ), row=4, col=1)
    
    # Max Deflection Label
    if not res_df['deflection'].empty:
        idx_max_def = res_df['deflection'].abs().idxmax()
        max_def = res_df['deflection'].iloc[idx_max_def]
        if abs(max_def) > 0.001:
             fig.add_annotation(
                x=res_df['x'].iloc[idx_max_def], y=max_def,
                text=f"<b>Max: {max_def:.2f} mm</b>",
                showarrow=True, arrowhead=1, ay=30 if max_def < 0 else -30,
                font=dict(color='#8e44ad', size=11), row=4, col=1
            )

    # --- LAYOUT SETTINGS ---
    # Grid Lines at Supports
    for x_pos in cum_dist:
        fig.add_vline(x=x_pos, line_width=1, line_dash="dash", line_color="gray", opacity=0.3)

    fig.update_layout(
        height=1000, 
        showlegend=False, 
        template="plotly_white", 
        hovermode="x unified",
        margin=dict(t=50, b=40, l=60, r=20)
    )
    
    # *** CRITICAL SCALING FIX ***
    # Lock FBD Y-Axis Range: กำหนดแกน Y ของ FBD ให้คงที่ (-0.5 ถึง 1.5)
    # เพื่อให้เส้นคานไม่ถูกบีบอัด ส่วนลูกศรจะวาดทับด้วย Pixel Scaling เอง
    fig.update_yaxes(range=[-0.5, 1.5], showgrid=False, visible=False, row=1, col=1)
    
    # Axis Labels
    fig.update_yaxes(title_text="Shear (kN)", showgrid=True, row=2, col=1)
    fig.update_yaxes(title_text="Moment (kNm)", autorange="reversed", showgrid=True, row=3, col=1)
    fig.update_yaxes(title_text="Def. (mm)", showgrid=True, row=4, col=1)
    fig.update_xaxes(title_text="Length (m)", row=4, col=1)

    return fig
