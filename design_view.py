import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

def plot_analysis_results(res_df, spans, supports, loads, reactions):
    """
    Creates a Textbook-style structural analysis plot.
    Features: Distributed arrows for UDL, explicit Max/Min labels, and clear axis names.
    """
    
    # --- 1. PREPARE SUMMARY DATA ---
    max_shear_pos = res_df['shear'].max() / 1000
    max_shear_neg = res_df['shear'].min() / 1000
    
    m_max_val = res_df['moment'].max() / 1000
    m_min_val = res_df['moment'].min() / 1000
    
    # Deflection: Find absolute max
    idx_max_def = res_df['deflection'].abs().idxmax()
    max_def_val = res_df['deflection'].iloc[idx_max_def]
    
    summary_title = (
        f"<b>ANALYSIS RESULTS</b><br>"
        f"V<sub>max</sub> = {max(abs(max_shear_pos), abs(max_shear_neg)):.2f} kN | "
        f"M<sub>max</sub> = {max(abs(m_max_val), abs(m_min_val)):.2f} kNm | "
        f"Δ<sub>max</sub> = {max_def_val:.2f} mm"
    )

    # --- 2. CREATE SUBPLOTS ---
    fig = make_subplots(
        rows=4, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.08,
        subplot_titles=(
            "<b>1. Free Body Diagram (Textbook Style)</b>", 
            "<b>2. Shear Force Diagram (SFD)</b>", 
            "<b>3. Bending Moment Diagram (BMD)</b>",
            "<b>4. Elastic Curve (Deflection)</b>"
        ),
        row_heights=[0.25, 0.25, 0.25, 0.25]
    )

    # ==========================================
    # ROW 1: LOAD MODEL (Textbook Style)
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
            x=[row['x']], y=[-0.08], 
            mode='markers+text',
            marker=dict(symbol=sym, size=14, color='white', line=dict(width=2, color='black')),
            text=[row['type'][0]], textposition="bottom center",
            hoverinfo='name', name=f"Support"
        ), row=1, col=1)

    # Loads
    for l in loads:
        start_x = cum_dist[l['span_index']]
        mag_kN = l['mag'] / 1000.0
        
        if l['type'] == 'P':
            x_loc = start_x + l['dist']
            # ลูกศรเดี่ยว (Single Arrow)
            fig.add_annotation(
                x=x_loc, y=0, ax=0, ay=-50,
                xref="x1", yref="y1",
                showarrow=True, arrowhead=2, arrowsize=1.2, arrowwidth=2, arrowcolor="#c0392b",
                text=f"<b>P={mag_kN:.2f}</b>", yshift=5, row=1, col=1
            )

        elif l['type'] == 'U':
            x_s = start_x
            x_e = x_s + l['dist']
            h_vis = 0.25
            
            # 1. Top Bar Line
            fig.add_trace(go.Scatter(
                x=[x_s, x_e], y=[h_vis, h_vis],
                mode='lines', line=dict(color='#2980b9', width=2), hoverinfo='skip'
            ), row=1, col=1)
            
            # 2. Distributed Arrows (Loop วาดลูกศรเรียงกัน)
            # คำนวณจำนวนลูกศรตามความยาว (ให้สวยงาม ไม่ถี่เกินไป)
            n_arrows = max(3, int(l['dist'] * 3)) 
            arrow_x = np.linspace(x_s, x_e, n_arrows)
            
            for ax_x in arrow_x:
                fig.add_annotation(
                    x=ax_x, y=0, # หัวลูกศรที่คาน
                    ax=0, ay=-30, # หางลูกศรสูงขึ้นไป (สั้นกว่า P นิดหน่อย)
                    xref="x1", yref="y1",
                    showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1, arrowcolor="#2980b9",
                    row=1, col=1
                )
            
            # Label ตรงกลาง
            fig.add_annotation(
                x=(x_s+x_e)/2, y=h_vis,
                text=f"<b>w={mag_kN:.2f} kN/m</b>",
                showarrow=False, yshift=10, font=dict(color="#2980b9"),
                row=1, col=1
            )

    # ==========================================
    # ROW 2: SHEAR FORCE (SFD)
    # ==========================================
    fig.add_hline(y=0, line_color="black", line_width=1, row=2, col=1)
    fig.add_trace(go.Scatter(
        x=res_df['x'], y=res_df['shear']/1000, 
        mode='lines', name='Shear', line=dict(color='#e74c3c', width=2),
        fill='tozeroy', fillcolor='rgba(231, 76, 60, 0.1)'
    ), row=2, col=1)
    
    # Annotate Max/Min Shear
    for val, name in [(max_shear_pos, "Max"), (max_shear_neg, "Min")]:
        if abs(val) > 0.001:
            idx = (res_df['shear']/1000 - val).abs().idxmin()
            fig.add_annotation(
                x=res_df['x'].iloc[idx], y=val,
                text=f"{val:.2f}", showarrow=False, yshift=10 if val>0 else -10,
                font=dict(color='#e74c3c', size=10), row=2, col=1
            )

    # ==========================================
    # ROW 3: BENDING MOMENT (BMD)
    # ==========================================
    fig.add_hline(y=0, line_color="black", line_width=1, row=3, col=1)
    fig.add_trace(go.Scatter(
        x=res_df['x'], y=res_df['moment']/1000, 
        mode='lines', name='Moment', line=dict(color='#27ae60', width=2),
        fill='tozeroy', fillcolor='rgba(39, 174, 96, 0.1)'
    ), row=3, col=1)

    # Annotate Max/Min Moment
    for val in [m_max_val, m_min_val]:
        if abs(val) > 0.001:
            idx = (res_df['moment']/1000 - val).abs().idxmin()
            fig.add_annotation(
                x=res_df['x'].iloc[idx], y=val,
                text=f"<b>{val:.2f}</b>", 
                showarrow=True, arrowhead=1, ay=20 if val>0 else -20, # BMD กลับหัวใน Layout
                font=dict(color='#27ae60'), row=3, col=1
            )

    # ==========================================
    # ROW 4: DEFLECTION
    # ==========================================
    fig.add_hline(y=0, line_color="black", line_width=1, row=4, col=1)
    fig.add_trace(go.Scatter(
        x=res_df['x'], y=res_df['deflection'], 
        mode='lines', name='Deflection', line=dict(color='#8e44ad', width=2)
    ), row=4, col=1)
    
    # Annotate Max Deflection
    fig.add_annotation(
        x=res_df['x'].iloc[idx_max_def], y=max_def_val,
        text=f"<b>Max: {max_def_val:.3f} mm</b>",
        showarrow=True, arrowhead=1, 
        ay=30 if max_def_val < 0 else -30,
        font=dict(color='#8e44ad'), row=4, col=1
    )

    # ==========================================
    # LAYOUT SETTINGS
    # ==========================================
    # Vertical Grid Lines
    for x_pos in cum_dist:
        fig.add_vline(x=x_pos, line_width=1, line_dash="dash", line_color="gray", opacity=0.3)

    fig.update_layout(
        title=dict(text=summary_title, x=0.5, y=0.98, font=dict(size=14, color="#2c3e50")),
        height=1000, 
        showlegend=False, 
        template="plotly_white", 
        hovermode="x unified",
        margin=dict(t=80, b=60, l=60, r=20)
    )
    
    # Axis Names & Config
    fig.update_yaxes(visible=False, range=[-0.5, 0.8], row=1, col=1) # Load Model fixed range
    fig.update_yaxes(title_text="Shear V (kN)", showgrid=True, row=2, col=1)
    fig.update_yaxes(title_text="Moment M (kNm)", autorange="reversed", showgrid=True, row=3, col=1)
    fig.update_yaxes(title_text="Deflection δ (mm)", showgrid=True, zeroline=True, row=4, col=1)
    
    # X-Axis Label (Bottom only)
    fig.update_xaxes(title_text="Distance x (m)", row=4, col=1)

    return fig
