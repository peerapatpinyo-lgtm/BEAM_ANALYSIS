import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

def draw_interactive_diagrams(df, reac, spans, sup_df, raw_loads):
    """
    Final Fix for Alignment & Scaling Issues:
    1. Forced X-Axis Synchronization (Correction for Image 4/5 misalignment)
    2. Unified Magnitude Scaling (P=2000 looks 2x bigger than w=1000)
    3. Clean Engineering Layout
    """
    # --- 1. SETUP GEOMETRY ---
    nodes = [0] + list(np.cumsum(spans))
    total_len = nodes[-1]
    
    # คำนวณ Max Load ของทั้งระบบ เพื่อใช้ Scale ความสูงกราฟฟิก
    all_mags = [l['mag'] for l in raw_loads]
    global_max_mag = max(all_mags) if all_mags else 1000.0
    
    # Constants for Visuals
    BEAM_HEIGHT = 1.0       # ความสูงพื้นที่แสดงคาน (Visual Unit)
    BEAM_THICKNESS = 0.25   # ความหนาตัวคาน
    LOAD_SCALE_FACTOR = 0.8 # ความสูง Max Load เทียบกับพื้นที่
    
    # --- 2. CREATE SUBPLOTS ---
    # ใช้ vertical_spacing น้อยลงเพื่อให้ดูกระชับ
    fig = make_subplots(
        rows=4, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.03,
        row_heights=[0.3, 0.2, 0.25, 0.25], # ให้พื้นที่ Load Diagram เยอะสุด
        subplot_titles=("Loading Diagram (FBD)", "Shear Force (V)", "Bending Moment (M)", "Deflection (δ)")
    )

    # ==========================================
    # ROW 1: THE BEAM & LOADS (Corrected Alignment)
    # ==========================================
    
    # 1.1 วาดตัวคาน (Beam Body)
    fig.add_shape(type="rect",
        x0=0, x1=total_len, 
        y0=-BEAM_THICKNESS/2, y1=BEAM_THICKNESS/2,
        fillcolor="#F5F5F5", line=dict(color="#333", width=3),
        row=1, col=1
    )
    # Centerline
    fig.add_trace(go.Scatter(
        x=[0, total_len], y=[0, 0], mode="lines",
        line=dict(color="#999", width=1, dash="dashdot"),
        hoverinfo="skip", showlegend=False
    ), row=1, col=1)

    # 1.2 วาด Supports
    if not sup_df.empty:
        for _, s in sup_df.iterrows():
            # ใช้ node_idx เพื่อหาตำแหน่ง X ที่แท้จริง
            idx = int(s['id'])
            if idx < len(nodes):
                x_pos = nodes[idx]
                stype = s['type']
                
                # วาดรูป Support
                if stype == "Pin":
                    fig.add_trace(go.Scatter(
                        x=[x_pos], y=[-BEAM_THICKNESS/2], mode="markers",
                        marker=dict(symbol="triangle-up", size=15, color="#424242"),
                        showlegend=False, hoverinfo="skip"
                    ), row=1, col=1)
                elif stype == "Roller":
                    fig.add_trace(go.Scatter(
                        x=[x_pos], y=[-BEAM_THICKNESS/2 - 0.1], mode="markers",
                        marker=dict(symbol="circle", size=12, color="#424242", line=dict(width=1, color="white")),
                        showlegend=False, hoverinfo="skip"
                    ), row=1, col=1)
                elif stype == "Fixed":
                    fig.add_shape(type="line",
                        x0=x_pos, x1=x_pos, 
                        y0=-BEAM_THICKNESS/2 - 0.2, y1=BEAM_THICKNESS/2 + 0.2,
                        line=dict(color="#424242", width=4), row=1, col=1
                    )

    # 1.3 วาด Loads (Unified Scaling & Correct Positioning)
    for l in raw_loads:
        span_idx = int(l['span_idx'])
        if span_idx >= len(spans): continue
        
        # คำนวณตำแหน่ง Global X (แก้เรื่อง Load ไม่ตามระยะ)
        x_start = nodes[span_idx] + l['x']
        
        # คำนวณความสูง Visual (แก้เรื่องสัดส่วนเพี้ยน)
        # Load ค่ามาก = สูงมาก, ค่าน้อย = เตี้ย (Linear Scale)
        vis_h = (l['mag'] / global_max_mag) * LOAD_SCALE_FACTOR
        vis_h = max(0.3, vis_h) # ขั้นต่ำให้เห็นชัดหน่อย
        
        color = "#D32F2F" if l['case'] == 'LL' else "#1976D2"
        fill_c = "rgba(211, 47, 47, 0.15)" if l['case'] == 'LL' else "rgba(25, 118, 210, 0.15)"

        if l['type'] == 'P':
            fig.add_annotation(
                x=x_start, y=BEAM_THICKNESS/2,
                ax=0, ay=-vis_h * 50, # Scale vector length
                xref="x1", yref="y1",
                text=f"<b>P={l['mag']:,.0f}</b>",
                showarrow=True, arrowhead=2, arrowwidth=2, arrowcolor=color, arrowsize=1,
                font=dict(color=color, size=11), bgcolor="white", bordercolor=color, borderwidth=1,
                row=1, col=1
            )
            
        elif l['type'] == 'U':
            span_length = spans[span_idx]
            local_end = l.get('end', span_length)
            x_end = nodes[span_idx] + local_end
            
            y_base = BEAM_THICKNESS/2
            y_top = y_base + vis_h
            
            # Draw Block
            fig.add_trace(go.Scatter(
                x=[x_start, x_end, x_end, x_start],
                y=[y_base, y_base, y_top, y_top],
                fill='toself', fillcolor=fill_c, mode='none', hoverinfo='skip', showlegend=False
            ), row=1, col=1)
            
            # Top Line
            fig.add_trace(go.Scatter(
                x=[x_start, x_end], y=[y_top, y_top],
                mode='lines', line=dict(color=color, width=2), showlegend=False
            ), row=1, col=1)
            
            # Arrows
            n_arrows = max(3, int((x_end - x_start)*2))
            for ax in np.linspace(x_start, x_end, n_arrows):
                fig.add_annotation(
                    x=ax, y=y_base, ax=0, ay=-vis_h*50, # Match P scale roughly
                    xref="x1", yref="y1", showarrow=True, arrowhead=2, arrowwidth=1, arrowcolor=color, arrowsize=0.8,
                    row=1, col=1
                )

            # Label Center
            fig.add_annotation(
                x=(x_start+x_end)/2, y=y_top,
                text=f"<b>w={l['mag']:,.0f}</b>",
                yshift=10, showarrow=False, font=dict(color=color, size=11),
                bgcolor="white", bordercolor=color, borderwidth=1,
                row=1, col=1
            )

    # Lock Y-Axis for Row 1 (Visual Space)
    fig.update_yaxes(range=[-1.0, 2.5], visible=False, fixedrange=True, row=1, col=1)

    # ==========================================
    # ROW 2, 3, 4: GRAPHS (With Clean Grids)
    # ==========================================
    
    # Shear
    fig.add_trace(go.Scatter(
        x=df['x'], y=df['shear'], mode='lines', line_shape='hv',
        fill='tozeroy', line=dict(color='#FFA000', width=2), fillcolor='rgba(255, 160, 0, 0.2)'
    ), row=2, col=1)
    
    # Moment (Inverted Logic handled by autorange reversed later)
    df['m_pos'] = df['moment'].clip(lower=0)
    df['m_neg'] = df['moment'].clip(upper=0)
    fig.add_trace(go.Scatter(x=df['x'], y=df['m_pos'], fill='tozeroy', mode='lines', line=dict(width=0), fillcolor='rgba(33, 150, 243, 0.3)'), row=3, col=1)
    fig.add_trace(go.Scatter(x=df['x'], y=df['m_neg'], fill='tozeroy', mode='lines', line=dict(width=0), fillcolor='rgba(233, 30, 99, 0.3)'), row=3, col=1)
    fig.add_trace(go.Scatter(x=df['x'], y=df['moment'], mode='lines', line=dict(color='#546E7A', width=2)), row=3, col=1)

    # Deflection
    fig.add_trace(go.Scatter(
        x=df['x'], y=df['deflection']*1000, mode='lines', 
        line=dict(color='#43A047', width=2, dash='dot')
    ), row=4, col=1)

    # ==========================================
    # ANNOTATIONS (Values Boxed)
    # ==========================================
    # Helper to add value labels
    def add_label(row, x, y, text, color):
        fig.add_annotation(
            x=x, y=y, text=f"<b>{text}</b>",
            showarrow=False, yshift=15 if y >= 0 else -15,
            font=dict(color=color, size=10),
            bgcolor="white", bordercolor=color, borderwidth=1, borderpad=2,
            row=row, col=1
        )

    # Shear Labels
    v_max, v_min = df['shear'].max(), df['shear'].min()
    add_label(2, df.loc[df['shear'].idxmax(), 'x'], v_max, f"{v_max:.0f}", "#FFA000")
    add_label(2, df.loc[df['shear'].idxmin(), 'x'], v_min, f"{v_min:.0f}", "#FFA000")

    # Moment Labels
    m_max, m_min = df['moment'].max(), df['moment'].min()
    if m_max > 10: add_label(3, df.loc[df['moment'].idxmax(), 'x'], m_max, f"{m_max:.0f}", "#1976D2")
    if abs(m_min) > 10: add_label(3, df.loc[df['moment'].idxmin(), 'x'], m_min, f"{m_min:.0f}", "#C2185B")

    # ==========================================
    # GLOBAL LAYOUT (The Alignment Fix)
    # ==========================================
    
    # กำหนด X-Range ให้เหมือนกันทุกกราฟ!!! (แก้ปัญหา Alignment หลุด)
    x_range = [-0.5, total_len + 0.5]
    
    # Grid Style
    grid_props = dict(
        showgrid=True, gridcolor='#E0E0E0', gridwidth=1,
        showline=True, linecolor='#333', linewidth=1,
        mirror=True, zeroline=True, zerolinecolor='#9E9E9E'
    )

    fig.update_layout(height=1200, template="plotly_white", showlegend=False, margin=dict(t=50, b=50, l=50, r=50))
    
    # Apply Axis Settings
    # Row 1: Beam
    fig.update_xaxes(range=x_range, **grid_props, row=1, col=1) 
    
    # Row 2: Shear
    fig.update_xaxes(range=x_range, matches='x', **grid_props, row=2, col=1)
    fig.update_yaxes(title="V (kg)", **grid_props, row=2, col=1)
    
    # Row 3: Moment
    fig.update_xaxes(range=x_range, matches='x', **grid_props, row=3, col=1)
    fig.update_yaxes(title="M (kg-m)", autorange="reversed", **grid_props, row=3, col=1)
    
    # Row 4: Deflection
    fig.update_xaxes(title="Distance (m)", range=x_range, matches='x', **grid_props, row=4, col=1)
    fig.update_yaxes(title="δ (mm)", **grid_props, row=4, col=1)

    st.plotly_chart(fig, use_container_width=True)

def render_result_tables(df, reac, spans, u_force, u_len):
    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.caption("📍 Support Reactions")
        if reac is not None:
            r_data = [{"Node": i, "Ry (kg)": f"{reac[2*i]:,.2f}", "Mz (kg-m)": f"{reac[2*i+1]:,.2f}"} for i in range(len(reac)//2) if abs(reac[2*i])>0.1 or abs(reac[2*i+1])>0.1]
            st.table(pd.DataFrame(r_data))
    with c2:
        st.caption("📐 Max Values per Span")
        s_data = []
        nodes = [0] + list(np.cumsum(spans))
        for i in range(len(spans)):
            sub = df[(df['x'] >= nodes[i]) & (df['x'] <= nodes[i+1])]
            s_data.append({"Span": i+1, "V_max": f"{sub['shear'].abs().max():,.0f}", "M_max": f"{sub['moment'].abs().max():,.0f}", "Defl_max": f"{sub['deflection'].abs().max()*1000:.2f}"})
        st.table(pd.DataFrame(s_data))
