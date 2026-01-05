import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

def draw_interactive_diagrams(df, reac, spans, sup_df, raw_loads):
    """
    Ultimate Professional View (Final Fixed)
    - Unified Scaling: 2000 is visually larger than 1000.
    - True Scale X-Axis: Beam aligns perfectly with graphs below.
    - Engineering Paper Style: Clean grids, precise markers.
    """
    # --- 1. GEOMETRY & COORDINATES ---
    nodes = [0] + list(np.cumsum(spans))
    total_len = nodes[-1]
    
    # --- 2. UNIFIED MAGNITUDE SCALING (The "Common Sense" Scale) ---
    # รวมค่าทุกอย่าง (P และ U) เพื่อหาค่า Max ตัวเดียว
    # เพื่อให้ P=2000 สูงเป็น 2 เท่าของ w=1000 จริงๆ ในทางสายตา
    all_mags = [l['mag'] for l in raw_loads]
    global_max = max(all_mags) if all_mags else 1000.0
    
    # ความสูง Reference สำหรับกราฟฟิก (หน่วย Visual แกน Y)
    REF_HEIGHT = 2.0 

    # --- 3. PLOT SETUP ---
    fig = make_subplots(
        rows=4, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.06,
        row_heights=[0.3, 0.23, 0.25, 0.22],
        subplot_titles=(
            "📌 Free Body Diagram", 
            "⚡ Shear Force (V)", 
            "🔄 Bending Moment (Tension Side)", 
            "📉 Deflection (δ)"
        )
    )

    # ==========================================
    # ROW 1: BEAM & LOADS (True Scale)
    # ==========================================
    beam_y_top = 0.25
    beam_y_bot = -0.25
    
    # 1.1 The Beam (Classic Engineering Look)
    fig.add_shape(type="rect",
        x0=0, x1=total_len, y0=beam_y_bot, y1=beam_y_top,
        fillcolor="#FAFAFA", line=dict(color="#212121", width=2.5),
        layer="below", row=1, col=1
    )
    # Centerline
    fig.add_trace(go.Scatter(
        x=[0, total_len], y=[0, 0], mode="lines",
        line=dict(color="#B0BEC5", width=1.5, dash="dashdot"),
        hoverinfo="skip", showlegend=False
    ), row=1, col=1)

    # 1.2 Supports
    if not sup_df.empty:
        for _, s in sup_df.iterrows():
            node_idx = int(s['id'])
            if node_idx < len(nodes):
                x = nodes[node_idx]
                stype = s['type']
                
                if stype == "Pin":
                    fig.add_trace(go.Scatter(
                        x=[x], y=[beam_y_bot], mode="markers",
                        marker=dict(symbol="triangle-up", size=15, color="#455A64"),
                        hoverinfo="text", text="Pin", showlegend=False
                    ), row=1, col=1)
                    fig.add_shape(type="line", x0=x-0.2, x1=x+0.2, y0=beam_y_bot-0.1, y1=beam_y_bot-0.1,
                                  line=dict(color="#455A64", width=2), row=1, col=1)
                
                elif stype == "Roller":
                    fig.add_trace(go.Scatter(
                        x=[x], y=[beam_y_bot - 0.1], mode="markers",
                        marker=dict(symbol="circle", size=12, color="#455A64", line=dict(width=1.5, color="white")),
                        showlegend=False
                    ), row=1, col=1)
                    fig.add_shape(type="line", x0=x-0.2, x1=x+0.2, y0=beam_y_bot-0.25, y1=beam_y_bot-0.25,
                                  line=dict(color="#455A64", width=2), row=1, col=1)

                elif stype == "Fixed":
                    fig.add_shape(type="line", x0=x, x1=x, y0=beam_y_bot-0.3, y1=beam_y_top+0.3,
                        line=dict(color="#455A64", width=4), row=1, col=1)
                    hatch_dir = 1 if x == 0 else -1
                    for h in np.linspace(beam_y_bot-0.3, beam_y_top+0.3, 7):
                        fig.add_shape(type="line", 
                            x0=x, y0=h, x1=x - (0.3 * hatch_dir), y1=h - 0.15,
                            line=dict(color="#455A64", width=1), row=1, col=1)

    # 1.3 Loads (Unified Scale)
    for l in raw_loads:
        span_idx = int(l['span_idx'])
        if span_idx >= len(spans): continue 
        
        # Calculate Coordinates
        x_start = nodes[span_idx] + l['x']
        
        # Calculate Visual Height (Unified)
        # 2000 จะสูงเป็น 2 เท่าของ 1000 ไม่ว่าจะเป็น P หรือ U
        raw_h = (l['mag'] / global_max) * REF_HEIGHT
        vis_h = max(0.6, raw_h) # ขั้นต่ำต้องเห็นชัด (0.6 unit)
        
        color = "#D32F2F" if l['case'] == 'LL' else "#1976D2" # Red/Blue Standard
        
        if l['type'] == 'P':
            fig.add_annotation(
                x=x_start, y=beam_y_top,
                ax=0, ay=-vis_h*40, # Pixel scaling
                xref="x1", yref="y1",
                text=f"<b>P={l['mag']:,.0f}</b>",
                showarrow=True, arrowhead=2, arrowwidth=2, arrowcolor=color,
                font=dict(color=color, size=11, weight="bold"),
                bgcolor="white", bordercolor=color, borderwidth=1, borderpad=2,
                row=1, col=1
            )
            
        elif l['type'] == 'U':
            span_L = spans[span_idx]
            x_end = nodes[span_idx] + l.get('end', span_L)
            
            # Draw Load Block
            fig.add_trace(go.Scatter(
                x=[x_start, x_end, x_end, x_start],
                y=[beam_y_top, beam_y_top, beam_y_top + vis_h, beam_y_top + vis_h],
                fill='toself', fillcolor=f"rgba{tuple(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + (0.15,)}",
                mode='none', hoverinfo='skip', showlegend=False
            ), row=1, col=1)
            
            # Top Line
            fig.add_trace(go.Scatter(
                x=[x_start, x_end], y=[beam_y_top + vis_h, beam_y_top + vis_h],
                mode='lines', line=dict(color=color, width=2), hoverinfo='skip', showlegend=False
            ), row=1, col=1)
            
            # Arrows (Distributed)
            n_arrows = max(3, int((x_end - x_start) * 2))
            for ax in np.linspace(x_start, x_end, n_arrows):
                fig.add_shape(type="line",
                    x0=ax, x1=ax, y0=beam_y_top, y1=beam_y_top + vis_h,
                    line=dict(color=color, width=1), layer="below", row=1, col=1
                )
                fig.add_trace(go.Scatter(
                    x=[ax], y=[beam_y_top],
                    mode='markers', marker=dict(symbol="triangle-down", size=6, color=color),
                    hoverinfo='skip', showlegend=False
                ), row=1, col=1)
                
            # Label
            fig.add_annotation(
                x=(x_start + x_end)/2, y=beam_y_top + vis_h,
                text=f"<b>w={l['mag']:,.0f}</b>",
                yshift=10, showarrow=False,
                font=dict(color=color, size=11, weight="bold"),
                bgcolor="white", bordercolor=color, borderwidth=1, borderpad=2,
                row=1, col=1
            )

    # Fix Range Row 1 (Visual Space)
    fig.update_yaxes(range=[-1.0, REF_HEIGHT + 1.0], fixedrange=True, visible=False, row=1, col=1)


    # ==========================================
    # ROW 2: SHEAR (Technical Yellow)
    # ==========================================
    fig.add_trace(go.Scatter(
        x=df['x'], y=df['shear'], mode='lines', line_shape='hv',
        fill='tozeroy', line=dict(color='#FBC02D', width=2), # Darker Yellow
        fillcolor='rgba(255, 235, 59, 0.3)', name="Shear"
    ), row=2, col=1)
    
    # Max Labels
    v_max, v_min = df['shear'].max(), df['shear'].min()
    fig.add_annotation(x=df.loc[df['shear'].idxmax(), 'x'], y=v_max, text=f"<b>{v_max:.0f}</b>", 
        showarrow=False, yshift=15, bgcolor="white", font=dict(color="#F57F17", size=10), row=2, col=1)
    fig.add_annotation(x=df.loc[df['shear'].idxmin(), 'x'], y=v_min, text=f"<b>{v_min:.0f}</b>", 
        showarrow=False, yshift=-15, bgcolor="white", font=dict(color="#F57F17", size=10), row=2, col=1)

    # ==========================================
    # ROW 3: MOMENT (Blue/Red Standard)
    # ==========================================
    df['m_pos'] = df['moment'].apply(lambda x: x if x >= 0 else 0)
    df['m_neg'] = df['moment'].apply(lambda x: x if x < 0 else 0)

    fig.add_trace(go.Scatter(x=df['x'], y=df['m_pos'], mode='lines', line=dict(width=0), 
        fill='tozeroy', fillcolor='rgba(33, 150, 243, 0.3)'), row=3, col=1) # Blue
    fig.add_trace(go.Scatter(x=df['x'], y=df['m_neg'], mode='lines', line=dict(width=0), 
        fill='tozeroy', fillcolor='rgba(233, 30, 99, 0.3)'), row=3, col=1)  # Red
    fig.add_trace(go.Scatter(x=df['x'], y=df['moment'], mode='lines', 
        line=dict(color='#37474F', width=2), showlegend=False), row=3, col=1)

    # Critical Values
    m_max, m_min = df['moment'].max(), df['moment'].min()
    if m_max > 1:
        fig.add_annotation(x=df.loc[df['moment'].idxmax(), 'x'], y=m_max, text=f"<b>{m_max:,.0f}</b>", 
            showarrow=True, arrowcolor="#1976D2", yshift=20, font=dict(color="#1976D2"), bgcolor="white", row=3, col=1)
    if abs(m_min) > 1:
        fig.add_annotation(x=df.loc[df['moment'].idxmin(), 'x'], y=m_min, text=f"<b>{m_min:,.0f}</b>", 
            showarrow=True, arrowcolor="#C2185B", yshift=-20, font=dict(color="#C2185B"), bgcolor="white", row=3, col=1)

    # ==========================================
    # ROW 4: DEFLECTION (Green Dashed)
    # ==========================================
    defl_mm = df['deflection'] * 1000
    fig.add_trace(go.Scatter(
        x=df['x'], y=defl_mm, mode='lines',
        line=dict(color='#2E7D32', width=2, dash='dot'), name="Deflection"
    ), row=4, col=1)
    
    d_max = defl_mm.abs().max()
    if d_max > 0.01:
        d_idx = defl_mm.abs().idxmax()
        fig.add_annotation(x=df.loc[d_idx, 'x'], y=defl_mm[d_idx], text=f"{defl_mm[d_idx]:.2f} mm",
            showarrow=True, arrowhead=1, row=4, col=1, bgcolor="white", font=dict(color="#2E7D32"))

    # ==========================================
    # GLOBAL LAYOUT (Pro Grid)
    # ==========================================
    grid_style = dict(
        showgrid=True, gridcolor='#E0E0E0', gridwidth=1,
        showline=True, linecolor='black', linewidth=1,
        mirror=True, zeroline=True, zerolinecolor='#9E9E9E'
    )
    
    fig.update_layout(
        height=1300,
        template="plotly_white",
        margin=dict(l=50, r=30, t=50, b=50),
        font=dict(family="Arial, sans-serif", size=12),
        showlegend=False
    )
    
    # Force Consistent X-Axis Range (แก้ปัญหาโหลดไม่ตามระยะ)
    # บังคับให้ทุกกราฟเริ่มที่ 0 และจบที่ความยาวคานรวมเสมอ
    fig.update_xaxes(range=[-0.5, total_len + 0.5], **grid_style)
    
    # Y Axes Titles
    fig.update_yaxes(visible=False, showgrid=False, row=1, col=1) # Row 1 No grid
    fig.update_yaxes(title="<b>V (kg)</b>", **grid_style, row=2, col=1)
    fig.update_yaxes(title="<b>M (kg-m)</b>", autorange="reversed", **grid_style, row=3, col=1)
    fig.update_yaxes(title="<b>δ (mm)</b>", **grid_style, row=4, col=1)
    
    st.plotly_chart(fig, use_container_width=True)

def render_result_tables(df, reac, spans, u_force, u_len):
    st.markdown("---")
    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown("##### 📍 Support Reactions")
        if reac is not None:
            r_data = []
            for i in range(len(reac)//2):
                if abs(reac[2*i]) > 0.01 or abs(reac[2*i+1]) > 0.01:
                    r_data.append({"Node": i, "Ry": f"{reac[2*i]:,.2f}", "M": f"{reac[2*i+1]:,.2f}"})
            st.table(pd.DataFrame(r_data))
    with c2:
        st.markdown("##### 📐 Design Forces (Max values)")
        s_data = []
        nodes = [0] + list(np.cumsum(spans))
        for i in range(len(spans)):
            sub = df[(df['x'] >= nodes[i]) & (df['x'] <= nodes[i+1])]
            s_data.append({
                "Span": i+1,
                "Vmax": f"{sub['shear'].abs().max():,.0f}",
                "M(+)": f"{sub['moment'].max():,.0f}",
                "M(-)": f"{sub['moment'].min():,.0f}",
                "Defl(mm)": f"{sub['deflection'].abs().max()*1000:.2f}"
            })
        st.dataframe(pd.DataFrame(s_data), hide_index=True, use_container_width=True)
