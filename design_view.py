import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

def plot_analysis_results(res_df, spans, supports, loads, reactions):
    """
    Creates a detailed structural analysis plot (4 Rows):
    1. Load Model
    2. Shear Force (SFD)
    3. Bending Moment (BMD)
    4. Deflection
    """
    
    # คำนวณค่า Max เพื่อนำไปทำสรุป
    max_shear = res_df['shear'].abs().max() / 1000
    max_moment_pos = max(0, res_df['moment'].max()) / 1000
    max_moment_neg = abs(min(0, res_df['moment'].min())) / 1000
    max_deflection = res_df['deflection'].abs().max()
    
    # Create Subplots: 4 Rows
    fig = make_subplots(
        rows=4, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.06,
        subplot_titles=(
            "<b>1. Free Body Diagram & Load Model</b>", 
            f"<b>2. Shear Force Diagram (Max: {max_shear:.2f} kN)</b>", 
            f"<b>3. Bending Moment Diagram (Max+: {max_moment_pos:.2f}, Max-: {max_moment_neg:.2f} kNm)</b>",
            f"<b>4. Deflection Diagram (Max: {max_deflection:.2f} mm)</b>"
        ),
        row_heights=[0.2, 0.25, 0.25, 0.3]
    )

    # ==========================================
    # 1. LOAD MODEL (Adjusted Size & Direction)
    # ==========================================
    total_L = sum(spans)
    cum_dist = [0] + list(np.cumsum(spans))
    
    # Beam Line
    fig.add_trace(go.Scatter(
        x=[0, total_L], y=[0, 0], 
        mode='lines', line=dict(color='black', width=4), hoverinfo='skip'
    ), row=1, col=1)
    
    # Supports
    for idx, row in supports.iterrows():
        sym = "triangle-up"
        if row['type'] == 'Fixed': sym = "square"
        elif row['type'] == 'Roller': sym = "circle"
        
        fig.add_trace(go.Scatter(
            x=[row['x']], y=[-0.02], # ขยับให้ชิดคานมากขึ้น
            mode='markers',
            marker=dict(symbol=sym, size=12, color='#2c3e50', line=dict(width=1.5, color='black')),
            hoverinfo='name', name=f"Support: {row['type']}"
        ), row=1, col=1)

    # Loads
    for l in loads:
        start_x = cum_dist[l['span_index']]
        mag_kN = l['mag'] / 1000.0
        
        if l['type'] == 'P':
            # Point Load: ลูกศรเล็กลงและชี้ลง
            x_loc = start_x + l['dist']
            fig.add_annotation(
                x=x_loc, y=0,      # หัวลูกศรแตะคาน (y=0)
                ax=0, ay=-40,      # หางลูกศรอยู่สูงขึ้นไป 40px (ทำให้ลูกศรชี้ลง)
                xref="x1", yref="y1",
                showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor="#c0392b",
                text=f"<b>{mag_kN:.1f} kN</b>",
                yshift=5, row=1, col=1
            )

        elif l['type'] == 'U':
            # UDL: ลดความสูงแถบโหลดลง (h_vis)
            x_s = start_x
            x_e = x_s + l['dist']
            h_vis = 0.15 # ลดความสูงลงให้ดูสมส่วน (Load scale)
            
            # Shaded Block
            fig.add_trace(go.Scatter(
                x=[x_s, x_e, x_e, x_s], y=[h_vis, h_vis, 0, 0], 
                fill='toself', fillcolor='rgba(52, 152, 219, 0.2)', mode='none', hoverinfo='skip'
            ), row=1, col=1)
            
            # Top Line
            fig.add_trace(go.Scatter(
                x=[x_s, x_e], y=[h_vis, h_vis],
                mode='lines', line=dict(color='#2980b9', width=1.5), hoverinfo='skip'
            ), row=1, col=1)
            
            # Center Arrow (ชี้ลง)
            mid_x = (x_s + x_e) / 2
            fig.add_annotation(
                x=mid_x, y=h_vis/2, # ชี้ลงไปกลางบล็อก
                ax=0, ay=-20,       # หางอยู่สูงขึ้นไป
                xref="x1", yref="y1",
                showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1.5, arrowcolor="#2980b9",
                text=f"{mag_kN:.1f} kN/m",
                yshift=5, row=1, col=1
            )

    # ==========================================
    # 2. SHEAR FORCE DIAGRAM (SFD)
    # ==========================================
    fig.add_hline(y=0, line_color="black", line_width=1, row=2, col=1)
    fig.add_trace(go.Scatter(
        x=res_df['x'], y=res_df['shear']/1000, 
        mode='lines', name='Shear', line=dict(color='#e74c3c', width=2),
        fill='tozeroy', fillcolor='rgba(231, 76, 60, 0.1)'
    ), row=2, col=1)

    # ==========================================
    # 3. BENDING MOMENT DIAGRAM (BMD)
    # ==========================================
    fig.add_hline(y=0, line_color="black", line_width=1, row=3, col=1)
    fig.add_trace(go.Scatter(
        x=res_df['x'], y=res_df['moment']/1000, 
        mode='lines', name='Moment', line=dict(color='#27ae60', width=2),
        fill='tozeroy', fillcolor='rgba(39, 174, 96, 0.1)'
    ), row=3, col=1)
    
    # ==========================================
    # 4. DEFLECTION DIAGRAM (New!)
    # ==========================================
    fig.add_hline(y=0, line_color="black", line_width=1, row=4, col=1)
    
    # Deflection usually goes down (negative), we plot as is.
    fig.add_trace(go.Scatter(
        x=res_df['x'], y=res_df['deflection'], 
        mode='lines', name='Deflection', line=dict(color='#8e44ad', width=2)
    ), row=4, col=1)
    
    # Mark Max Deflection
    idx_def_max = res_df['deflection'].abs().idxmax()
    def_val = res_df['deflection'].iloc[idx_def_max]
    fig.add_annotation(
        x=res_df['x'].iloc[idx_def_max], y=def_val,
        text=f"Max: {def_val:.2f} mm",
        showarrow=True, arrowhead=1, ay=30 if def_val < 0 else -30,
        font=dict(color='#8e44ad'), row=4, col=1
    )

    # ==========================================
    # LAYOUT & SUMMARY BOX
    # ==========================================
    # Vertical Grid Lines
    for x_pos in cum_dist:
        fig.add_vline(x=x_pos, line_width=1, line_dash="dash", line_color="gray", opacity=0.5)

    # Summary Text Box (Top Right)
    summary_text = (
        f"<b>ANALYSIS SUMMARY</b><br>"
        f"V<sub>max</sub> = {max_shear:.2f} kN<br>"
        f"M<sub>max(+)</sub> = {max_moment_pos:.2f} kNm<br>"
        f"M<sub>max(-)</sub> = {max_moment_pos:.2f} kNm<br>"
        f"Δ<sub>max</sub> = {max_deflection:.2f} mm"
    )
    
    # Add summary as annotation in the first plot area (top right corner)
    fig.add_annotation(
        text=summary_text,
        xref="paper", yref="paper",
        x=1.0, y=1.0, showarrow=False,
        align="right", bgcolor="rgba(255, 255, 255, 0.8)", bordercolor="black", borderwidth=1,
        row=1, col=1
    )

    fig.update_layout(height=1000, showlegend=False, template="plotly_white", hovermode="x unified")
    
    # Axis Updates
    fig.update_yaxes(visible=False, row=1, col=1) # Load
    fig.update_yaxes(title_text="V (kN)", row=2, col=1) # Shear
    fig.update_yaxes(title_text="M (kNm)", autorange="reversed", row=3, col=1) # Moment (Reversed)
    fig.update_yaxes(title_text="Def (mm)", autorange="reversed", row=4, col=1) # Deflection (Reversed: Down is positive visual)

    return fig
