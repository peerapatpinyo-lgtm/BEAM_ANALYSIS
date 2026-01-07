import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

def plot_analysis_results(res_df, spans, supports, loads, reactions):
    """
    Creates a detailed structural analysis plot (4 Rows):
    1. Load Model (FBD)
    2. Shear Force (SFD)
    3. Bending Moment (BMD)
    4. Deflection
    """
    
    # --- 1. PREPARE SUMMARY DATA ---
    # คำนวณค่า Max เพื่อนำไปทำสรุปหัวตาราง
    max_shear = res_df['shear'].abs().max() / 1000
    
    # Moment: แยกค่าบวก (Sagging) และลบ (Hogging)
    m_max_val = res_df['moment'].max() / 1000
    m_min_val = res_df['moment'].min() / 1000
    max_moment_pos = m_max_val if m_max_val > 0 else 0.0
    max_moment_neg = abs(m_min_val) if m_min_val < 0 else 0.0
    
    # Deflection: หาค่าการแอ่นตัวสูงสุด (ไม่ว่าจะขึ้นหรือลง)
    max_def_val = res_df['deflection'].abs().max()
    
    # สร้างข้อความสรุป (Summary Text) ไว้บนหัวกราฟ
    summary_title = (
        f"<b>ANALYSIS SUMMARY:</b> "
        f"V<sub>max</sub>={max_shear:.2f} kN | "
        f"M<sub>pos</sub>={max_moment_pos:.2f} kNm | "
        f"M<sub>neg</sub>={max_moment_neg:.2f} kNm | "
        f"Δ<sub>max</sub>={max_def_val:.2f} mm"
    )

    # --- 2. CREATE SUBPLOTS ---
    fig = make_subplots(
        rows=4, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.08,
        subplot_titles=(
            "<b>1. Free Body Diagram (Load Model)</b>", 
            "<b>2. Shear Force Diagram (SFD)</b>", 
            "<b>3. Bending Moment Diagram (BMD)</b>",
            "<b>4. Deflection (Elastic Curve)</b>"
        ),
        row_heights=[0.20, 0.25, 0.25, 0.30]
    )

    # ==========================================
    # ROW 1: LOAD MODEL
    # ==========================================
    total_L = sum(spans)
    cum_dist = [0] + list(np.cumsum(spans))
    
    # Beam Line
    fig.add_trace(go.Scatter(
        x=[0, total_L], y=[0, 0], 
        mode='lines', line=dict(color='black', width=5), hoverinfo='skip'
    ), row=1, col=1)
    
    # Supports
    for idx, row in supports.iterrows():
        sym = "triangle-up"
        if row['type'] == 'Fixed': sym = "square"
        elif row['type'] == 'Roller': sym = "circle"
        
        # วาง Support ไว้ใต้คานเล็กน้อย (y=-0.1)
        fig.add_trace(go.Scatter(
            x=[row['x']], y=[-0.1], 
            mode='markers+text',
            marker=dict(symbol=sym, size=14, color='#34495e', line=dict(width=2, color='black')),
            text=[row['type'][0]], textposition="bottom center", # Show first letter of support type
            hoverinfo='name', name=f"Support: {row['type']}"
        ), row=1, col=1)

    # Loads
    max_mag = 1.0 # ใช้สำหรับปรับสเกลความสูงกราฟ
    if loads: max_mag = max([l['mag'] for l in loads]) / 1000.0

    for l in loads:
        start_x = cum_dist[l['span_index']]
        mag_kN = l['mag'] / 1000.0
        
        if l['type'] == 'P':
            x_loc = start_x + l['dist']
            # ลูกศร Point Load: หางอยู่สูง หัวปักที่ y=0
            fig.add_annotation(
                x=x_loc, y=0,
                ax=0, ay=-50, # หางสูง 50px
                xref="x1", yref="y1",
                showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor="#c0392b",
                text=f"<b>{mag_kN:.1f}</b>",
                yshift=5, row=1, col=1
            )
        elif l['type'] == 'U':
            x_s = start_x
            x_e = x_s + l['dist']
            h_vis = 0.3 # ความสูง Visual คงที่สำหรับ UDL
            
            # Shaded Block
            fig.add_trace(go.Scatter(
                x=[x_s, x_e, x_e, x_s], y=[h_vis, h_vis, 0, 0], 
                fill='toself', fillcolor='rgba(52, 152, 219, 0.2)', mode='none', hoverinfo='skip'
            ), row=1, col=1)
            # Top Line
            fig.add_trace(go.Scatter(
                x=[x_s, x_e], y=[h_vis, h_vis],
                mode='lines', line=dict(color='#2980b9', width=2), hoverinfo='skip'
            ), row=1, col=1)
            # Arrow & Label
            mid_x = (x_s + x_e) / 2
            fig.add_annotation(
                x=mid_x, y=h_vis,
                ax=0, ay=-20,
                xref="x1", yref="y1",
                showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1.5, arrowcolor="#2980b9",
                text=f"{mag_kN:.1f} kN/m",
                yshift=5, row=1, col=1
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

    # ==========================================
    # ROW 3: BENDING MOMENT (BMD)
    # ==========================================
    fig.add_hline(y=0, line_color="black", line_width=1, row=3, col=1)
    fig.add_trace(go.Scatter(
        x=res_df['x'], y=res_df['moment']/1000, 
        mode='lines', name='Moment (kNm)', line=dict(color='#27ae60', width=2),
        fill='tozeroy', fillcolor='rgba(39, 174, 96, 0.1)'
    ), row=3, col=1)
    
    # ==========================================
    # ROW 4: DEFLECTION (+Up, -Down)
    # ==========================================
    fig.add_hline(y=0, line_color="black", line_width=1, row=4, col=1)
    
    # Plot ตามค่าจริง (Real Value Plotting)
    # ค่าลบ = ลงต่ำ (Sagging/Gravity), ค่าบวก = ขึ้นบน (Uplift)
    fig.add_trace(go.Scatter(
        x=res_df['x'], y=res_df['deflection'], 
        mode='lines', name='Deflection (mm)', line=dict(color='#8e44ad', width=2)
    ), row=4, col=1)

    # Annotate Max Deflection point
    idx_max_def = res_df['deflection'].abs().idxmax()
    val_max_def = res_df['deflection'].iloc[idx_max_def]
    
    fig.add_annotation(
        x=res_df['x'].iloc[idx_max_def], y=val_max_def,
        text=f"Max: {val_max_def:.2f} mm",
        showarrow=True, arrowhead=1, 
        ay=30 if val_max_def < 0 else -30, # ถ้าค่าลบ(ลง) ให้ลูกศรชี้ขึ้นมาจากด้านล่าง
        font=dict(color='#8e44ad', size=10),
        row=4, col=1
    )

    # ==========================================
    # LAYOUT SETTINGS
    # ==========================================
    # Grid Lines (Vertical at Supports)
    for x_pos in cum_dist:
        fig.add_vline(x=x_pos, line_width=1, line_dash="dash", line_color="gray", opacity=0.3)

    fig.update_layout(
        title=dict(text=summary_title, x=0.5, y=0.98, font=dict(size=14, color="#2c3e50")),
        height=900, 
        showlegend=False, 
        template="plotly_white", 
        hovermode="x unified",
        margin=dict(t=80, b=50, l=60, r=20)
    )
    
    # Axis Configuration
    # Row 1: Load (Set fixed range to make it look clean)
    fig.update_yaxes(visible=False, range=[-0.5, 1.0], row=1, col=1)
    
    # Row 2: Shear
    fig.update_yaxes(title_text="V (kN)", showgrid=True, row=2, col=1)
    
    # Row 3: Moment (Reversed Axis for Civil Eng Convention)
    fig.update_yaxes(title_text="M (kNm)", autorange="reversed", showgrid=True, row=3, col=1)
    
    # Row 4: Deflection (Standard Axis: +Up, -Down)
    # ไม่ใส่ autorange="reversed" เพื่อให้ + อยู่บน, - อยู่ล่างตามจริง
    fig.update_yaxes(title_text="Def (mm)", showgrid=True, zeroline=True, row=4, col=1)

    return fig
