import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

def draw_interactive_diagrams(df, reac, spans, sup_df, raw_loads):
    """
    Engineering Grade Visualization
    - Fixes Alignment: Forces all X-axes to match perfectly.
    - Fixes Scaling: Locks Visual Aspect Ratio (no more wasted whitespace).
    - Professional Look: Clean lines, clear labels, no clipping.
    """
    
    # --- 1. PREPARE DATA ---
    nodes = [0] + list(np.cumsum(spans))
    total_len = nodes[-1]
    
    # Find Global Max for Scaling (เพื่อให้สัดส่วน Load ดูสมจริงเทียบกันทั้งกระดาน)
    all_mags = [l['mag'] for l in raw_loads]
    max_load_val = max(all_mags) if all_mags else 1000.0
    
    # Visual Constants (หน่วยเป็นแกน Y สมมติของรูปบนสุด)
    BEAM_Y_ZERO = 0         # ระดับหลังคาน
    BEAM_THICK = 0.5        # ความหนาคาน
    MAX_VISUAL_HEIGHT = 2.5 # ความสูงสูงสุดที่ยอมให้ Load พุ่งขึ้นไป (Lock ไว้เลย)

    # --- 2. SETUP PLOT LAYOUT ---
    fig = make_subplots(
        rows=4, cols=1, 
        shared_xaxes=True, # สำคัญ: แชร์แกน X
        vertical_spacing=0.04,
        row_heights=[0.3, 0.23, 0.25, 0.22], # ให้พื้นที่รูปคานเยอะสุด
        subplot_titles=(
            "", # เว้นว่างไว้ เดี๋ยวใส่ Annotation แทนจะได้ไม่ทับ
            "", 
            "", 
            ""
        )
    )

    # ==========================================
    # ROW 1: FREE BODY DIAGRAM (Engineering Style)
    # ==========================================
    
    # 1.1 The Beam (วาดเป็น Shape สี่เหลี่ยมตายตัว)
    fig.add_shape(type="rect",
        x0=0, x1=total_len, 
        y0=-BEAM_THICK, y1=BEAM_Y_ZERO, # คานอยู่ใต้แกน 0
        fillcolor="#EEEEEE", line=dict(color="#424242", width=2.5),
        row=1, col=1
    )
    
    # 1.2 Supports (วาดตามตำแหน่ง Node จริง)
    if not sup_df.empty:
        for _, s in sup_df.iterrows():
            idx = int(s['id'])
            if idx < len(nodes):
                x_pos = nodes[idx]
                stype = s['type']
                
                # Support Graphics
                if stype == "Pin":
                    # สามเหลี่ยม
                    fig.add_trace(go.Scatter(
                        x=[x_pos], y=[-BEAM_THICK], mode="markers",
                        marker=dict(symbol="triangle-up", size=14, color="#37474F"),
                        hoverinfo="skip", showlegend=False
                    ), row=1, col=1)
                    # ฐานรอง
                    fig.add_shape(type="line", x0=x_pos-0.2, x1=x_pos+0.2, y0=-BEAM_THICK-0.2, y1=-BEAM_THICK-0.2,
                        line=dict(color="#37474F", width=2), row=1, col=1)

                elif stype == "Roller":
                    fig.add_trace(go.Scatter(
                        x=[x_pos], y=[-BEAM_THICK-0.15], mode="markers",
                        marker=dict(symbol="circle", size=12, color="#37474F", line=dict(width=1.5, color="white")),
                        showlegend=False, hoverinfo="skip"
                    ), row=1, col=1)
                    fig.add_shape(type="line", x0=x_pos-0.2, x1=x_pos+0.2, y0=-BEAM_THICK-0.4, y1=-BEAM_THICK-0.4,
                        line=dict(color="#37474F", width=2), row=1, col=1)
                
                elif stype == "Fixed":
                     fig.add_shape(type="line", x0=x_pos, x1=x_pos, y0=-BEAM_THICK-0.3, y1=BEAM_Y_ZERO+0.5,
                        line=dict(color="#37474F", width=4), row=1, col=1)

    # 1.3 Loads (Unified Scaling)
    for l in raw_loads:
        span_idx = int(l['span_idx'])
        if span_idx >= len(spans): continue
        
        x_start = nodes[span_idx] + l['x']
        
        # Calculate Visual Height (Normalized)
        # Load 100% = สูง 2.5 หน่วย (Visual Units)
        ratio = l['mag'] / max_load_val
        vis_h = max(0.8, ratio * MAX_VISUAL_HEIGHT) 

        color = "#C62828" if l['case'] == 'LL' else "#1565C0" # Red/Blue

        if l['type'] == 'P':
            # Point Load
            fig.add_annotation(
                x=x_start, y=BEAM_Y_ZERO,
                ax=0, ay=-vis_h*35, # Vector length based on visual height
                xref="x1", yref="y1",
                text=f"<b>P={l['mag']:,.0f}</b>",
                showarrow=True, arrowhead=2, arrowwidth=2, arrowcolor=color, arrowsize=1,
                font=dict(color=color, size=11), bgcolor="rgba(255,255,255,0.8)",
                row=1, col=1
            )
        elif l['type'] == 'U':
            # Distributed Load
            span_len = spans[span_idx]
            x_end = nodes[span_idx] + l.get('end', span_len)
            
            # Shaded Box
            fig.add_trace(go.Scatter(
                x=[x_start, x_end, x_end, x_start],
                y=[BEAM_Y_ZERO, BEAM_Y_ZERO, vis_h, vis_h],
                fill='toself', fillcolor=f"rgba{tuple(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + (0.1,)}",
                mode='none', hoverinfo='skip', showlegend=False
            ), row=1, col=1)
            
            # Top Line
            fig.add_trace(go.Scatter(
                x=[x_start, x_end], y=[vis_h, vis_h],
                mode='lines', line=dict(color=color, width=1.5), showlegend=False
            ), row=1, col=1)
            
            # Arrows (Distributed)
            n_arr = max(3, int((x_end-x_start)*1.5))
            for ax in np.linspace(x_start, x_end, n_arr):
                fig.add_annotation(
                    x=ax, y=BEAM_Y_ZERO, ax=0, ay=-vis_h*35,
                    xref="x1", yref="y1", showarrow=True, arrowhead=2, arrowwidth=1, arrowcolor=color, arrowsize=0.8,
                    row=1, col=1
                )
            # Label
            fig.add_annotation(
                x=(x_start+x_end)/2, y=vis_h, text=f"<b>w={l['mag']:,.0f}</b>",
                yshift=10, showarrow=False, font=dict(color=color, size=11), bgcolor="white",
                row=1, col=1
            )

    # --- LOCK VIEW ROW 1 (The Fix for Whitespace) ---
    # บังคับแกน Y ของรูปแรกให้ Fix เลย จะได้ไม่เหลือที่ว่างเยอะ
    fig.update_yaxes(range=[-1.5, MAX_VISUAL_HEIGHT + 1.0], fixedrange=True, visible=False, row=1, col=1)


    # ==========================================
    # ROWS 2-4: GRAPHS (Aligned)
    # ==========================================
    
    # Shear
    fig.add_trace(go.Scatter(x=df['x'], y=df['shear'], mode='lines', fill='tozeroy', 
        line=dict(color='#FFA000', width=2), fillcolor='rgba(255, 160, 0, 0.2)', name="V"), row=2, col=1)
    
    # Moment (Inverted Logic)
    fig.add_trace(go.Scatter(x=df['x'], y=df['moment'], mode='lines', 
        line=dict(color='#455A64', width=2), fill='tozeroy', fillcolor='rgba(69, 90, 100, 0.1)', name="M"), row=3, col=1)

    # Deflection
    fig.add_trace(go.Scatter(x=df['x'], y=df['deflection']*1000, mode='lines', 
        line=dict(color='#2E7D32', width=2, dash='dot'), name="Deflection"), row=4, col=1)

    # --- ANNOTATIONS (Smart Labels) ---
    def smart_label(row, col, x, y, txt, color, anchor="bottom"):
        yshift = 15 if anchor == "bottom" else -15
        fig.add_annotation(x=x, y=y, text=f"<b>{txt}</b>", showarrow=False, yshift=yshift,
            font=dict(color=color, size=10), bgcolor="white", bordercolor=color, borderwidth=1, borderpad=2,
            row=row, col=col)

    # Add Max/Min labels (ตัวอย่าง Shear)
    v_max = df['shear'].max()
    smart_label(2, 1, df.loc[df['shear'].idxmax(), 'x'], v_max, f"{v_max:.0f}", "#FFA000")

    # Add Moment Labels
    m_min = df['moment'].min() # Negative moment usually governs design
    smart_label(3, 1, df.loc[df['moment'].idxmin(), 'x'], m_min, f"{m_min:.0f}", "#C2185B", anchor="top")


    # ==========================================
    # GLOBAL LAYOUT (Professional Grid)
    # ==========================================
    
    # *** CRITICAL FIX: FORCED X-RANGE ***
    # บังคับให้ทุกกราฟเริ่มและจบที่เดียวกันเป๊ะๆ เพื่อให้ Grid Line ตรงกัน
    common_x_range = [-0.5, total_len + 0.5]
    
    grid_style = dict(
        showgrid=True, gridcolor='#E0E0E0', gridwidth=1,
        showline=True, linecolor='black', linewidth=1,
        mirror=True
    )

    fig.update_layout(height=1200, template="plotly_white", showlegend=False, 
                      margin=dict(t=40, b=40, l=60, r=40),
                      font=dict(family="Arial", size=12))

    # Apply Styles
    # Row 1: Beam (Hide Grid)
    fig.update_xaxes(range=common_x_range, showgrid=False, visible=False, row=1, col=1)
    
    # Row 2: Shear
    fig.update_xaxes(range=common_x_range, matches='x', **grid_style, row=2, col=1)
    fig.update_yaxes(title="<b>Shear (kg)</b>", **grid_style, row=2, col=1)
    
    # Row 3: Moment
    fig.update_xaxes(range=common_x_range, matches='x', **grid_style, row=3, col=1)
    fig.update_yaxes(title="<b>Moment (kg-m)</b>", autorange="reversed", **grid_style, row=3, col=1)
    
    # Row 4: Deflection
    fig.update_xaxes(title="<b>Distance (m)</b>", range=common_x_range, matches='x', **grid_style, row=4, col=1)
    fig.update_yaxes(title="<b>Deflection (mm)</b>", **grid_style, row=4, col=1)
    
    # Titles inside plots instead of Subplot Titles (ประหยัดที่)
    fig.add_annotation(text="Free Body Diagram", xref="paper", yref="paper", x=0, y=1.01, showarrow=False, font=dict(size=14, color="#333"), row=1, col=1)

    st.plotly_chart(fig, use_container_width=True)

    # ... (Render Tables function remains the same)
