import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

def draw_interactive_diagrams(df, reac, spans, sup_df, raw_loads):
    """
    Ultimate Visualization v3:
    - Fixed Proportions (Beam always looks solid)
    - Clamped Load Heights (Never too tall)
    - High Visibility Grids
    - Clean Engineering Aesthetics
    """
    nodes = [0] + list(np.cumsum(spans))
    total_len = nodes[-1]
    
    # --- 1. SETTINGS & SCALING ---
    # เราจะไม่ใช้ความสูงคานตามความยาวจริง (เพราะถ้าคานยาวมาก คานจะดูผอมเป็นเส้นด้าย)
    # เราจะใช้ "Visual Coordinate System" สำหรับแถวแรก
    # ให้ Beam หนา 1 หน่วยเสมอ ในแกน Y สมมติ
    VISUAL_BEAM_DEPTH = 1.0 
    beam_top = VISUAL_BEAM_DEPTH / 2
    beam_bot = -VISUAL_BEAM_DEPTH / 2
    
    # Scale Factor สำหรับ Load (เพื่อไม่ให้สูงเกินไป)
    # บังคับให้ Load สูงสุด สูงไม่เกิน 1.5 เท่าของความลึกคาน
    MAX_LOAD_VISUAL_HEIGHT = 1.8 
    
    # หาค่า Max Load จริงเพื่อใช้ Normalize
    p_vals = [l['mag'] for l in raw_loads if l['type']=='P']
    u_vals = [l['mag'] for l in raw_loads if l['type']=='U']
    max_p = max(p_vals) if p_vals else 1.0
    max_u = max(u_vals) if u_vals else 1.0

    # --- 2. CREATE SUBPLOTS ---
    fig = make_subplots(
        rows=4, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.06,
        row_heights=[0.25, 0.25, 0.25, 0.25], # แบ่งความสูงเท่าๆ กันให้ดูสบายตา
        subplot_titles=(
            "📌 Loading Diagram", 
            "⚡ Shear Force (V)", 
            "🔄 Bending Moment (Tension Side)", 
            "📉 Deflection (δ)"
        )
    )

    # ==========================================
    # ROW 1: LOADING DIAGRAM (Visual Scale)
    # ==========================================
    
    # 1.1 The Beam (วาดให้ดูหนา สวยงาม)
    fig.add_shape(type="rect",
        x0=0, x1=total_len, y0=beam_bot, y1=beam_top,
        fillcolor="#F5F5F5", line=dict(color="#424242", width=3),
        layer="below", row=1, col=1
    )
    # Centerline
    fig.add_trace(go.Scatter(
        x=[0, total_len], y=[0, 0], mode="lines",
        line=dict(color="#BDBDBD", width=1, dash="dashdot"),
        hoverinfo="skip", showlegend=False
    ), row=1, col=1)

    # 1.2 Supports
    if not sup_df.empty:
        for _, s in sup_df.iterrows():
            x = nodes[int(s['id'])]
            stype = s['type']
            
            # ขนาด Support สัมพันธ์กับ Visual Beam
            sup_sz = 0.6 
            
            if stype == "Pin":
                fig.add_trace(go.Scatter(
                    x=[x], y=[beam_bot], mode="markers",
                    marker=dict(symbol="triangle-up", size=15, color="#37474F"),
                    hoverinfo="text", text="Pin Support", showlegend=False
                ), row=1, col=1)
                # ฐานรอง
                fig.add_shape(type="line", x0=x-0.2, x1=x+0.2, y0=beam_bot-0.1, y1=beam_bot-0.1,
                              line=dict(color="#37474F", width=3), row=1, col=1)
                
            elif stype == "Roller":
                fig.add_trace(go.Scatter(
                    x=[x], y=[beam_bot - 0.2], mode="markers",
                    marker=dict(symbol="circle", size=12, color="#37474F", line=dict(width=1, color="white")),
                    showlegend=False
                ), row=1, col=1)
                fig.add_shape(type="line", x0=x-0.2, x1=x+0.2, y0=beam_bot-0.4, y1=beam_bot-0.4,
                              line=dict(color="#37474F", width=3), row=1, col=1)

            elif stype == "Fixed":
                fig.add_shape(type="line", x0=x, x1=x, y0=beam_bot-0.4, y1=beam_top+0.4,
                    line=dict(color="#37474F", width=5), row=1, col=1)
                # ลายแรเงา
                hatch_dir = 1 if x == 0 else -1
                for h in np.linspace(beam_bot-0.4, beam_top+0.4, 8):
                    fig.add_shape(type="line", 
                        x0=x, y0=h, x1=x - (0.3 * hatch_dir), y1=h - 0.15,
                        line=dict(color="#37474F", width=1.5), row=1, col=1)

    # 1.3 Loads (Normalized Height)
    for l in raw_loads:
        color = "#D32F2F" if l['case'] == 'LL' else "#1976D2"
        fill_color = "rgba(211, 47, 47, 0.1)" if l['case'] == 'LL' else "rgba(25, 118, 210, 0.1)"
        
        if l['type'] == 'P':
            # Normalize P ความสูงกราฟฟิก
            ratio = l['mag'] / max_p
            # ให้ความสูงอยู่ระหว่าง 0.6 ถึง 1.0 ของ MAX limit
            draw_h = MAX_LOAD_VISUAL_HEIGHT * (0.6 + 0.4*ratio)
            
            x_loc = nodes[int(l['span_idx'])] + l['x']
            
            # วาดลูกศรด้วย Annotation (คุมความยาวได้เป๊ะกว่า)
            fig.add_annotation(
                x=x_loc, y=beam_top,
                ax=0, ay=-draw_h*30, # pixel scaling approx
                xref="x1", yref="y1",
                text=f"<b>P={l['mag']:,.0f}</b>",
                showarrow=True, arrowhead=2, arrowwidth=2, arrowcolor=color, arrowsize=1.0,
                font=dict(color=color, size=11),
                bgcolor="white", bordercolor=color, borderwidth=1, borderpad=3,
                row=1, col=1
            )
            
        elif l['type'] == 'U':
            # Normalize U
            ratio = l['mag'] / max_u
            draw_h = MAX_LOAD_VISUAL_HEIGHT * (0.5 + 0.5*ratio) # U load เตี้ยกว่า P หน่อยเพื่อให้ดูต่าง
            
            start_x = nodes[int(l['span_idx'])] + l['x']
            end_x = nodes[int(l['span_idx'])] + l.get('end', spans[int(l['span_idx'])])
            
            # Block Area
            fig.add_trace(go.Scatter(
                x=[start_x, end_x, end_x, start_x],
                y=[beam_top, beam_top, beam_top + draw_h, beam_top + draw_h],
                fill='toself', fillcolor=fill_color,
                mode='none', hoverinfo='skip', showlegend=False
            ), row=1, col=1)
            
            # Top Bar
            fig.add_trace(go.Scatter(
                x=[start_x, end_x], y=[beam_top + draw_h, beam_top + draw_h],
                mode='lines', line=dict(color=color, width=2),
                hoverinfo='skip', showlegend=False
            ), row=1, col=1)
            
            # Arrows
            n_arrows = max(3, int((end_x - start_x) * 2.5))
            for ax in np.linspace(start_x, end_x, n_arrows):
                fig.add_shape(type="line",
                    x0=ax, x1=ax, y0=beam_top, y1=beam_top + draw_h,
                    line=dict(color=color, width=1.5), layer="below", row=1, col=1
                )
                fig.add_trace(go.Scatter(
                    x=[ax], y=[beam_top],
                    mode='markers', marker=dict(symbol="triangle-down", size=6, color=color),
                    hoverinfo='skip', showlegend=False
                ), row=1, col=1)

            # Label (Boxed)
            fig.add_annotation(
                x=(start_x+end_x)/2, y=beam_top + draw_h,
                text=f"<b>w={l['mag']:,.0f}</b>",
                yshift=12, showarrow=False,
                font=dict(color=color, size=11),
                bgcolor="white", bordercolor=color, borderwidth=1, borderpad=2,
                row=1, col=1
            )

    # FIX RANGE สำหรับ Row 1 เพื่อให้คานไม่เพี้ยน
    # บังคับแกน Y ให้แสดงพื้นที่ว่างด้านบนและล่างพอสมควร
    fig.update_yaxes(range=[-1.5, 3.0], row=1, col=1)


    # ==========================================
    # ROW 2: SHEAR FORCE (Clean Grid)
    # ==========================================
    fig.add_trace(go.Scatter(
        x=df['x'], y=df['shear'], mode='lines', line_shape='hv',
        fill='tozeroy', line=dict(color='#F9A825', width=2), # Darker Yellow
        fillcolor='rgba(253, 216, 53, 0.2)', name="Shear"
    ), row=2, col=1)
    
    # Max/Min Labels (Boxed)
    v_max, v_min = df['shear'].max(), df['shear'].min()
    fig.add_annotation(x=df.loc[df['shear'].idxmax(), 'x'], y=v_max, text=f"{v_max:.0f}", 
        showarrow=False, yshift=15, bgcolor="white", bordercolor="#F9A825", row=2, col=1)
    fig.add_annotation(x=df.loc[df['shear'].idxmin(), 'x'], y=v_min, text=f"{v_min:.0f}", 
        showarrow=False, yshift=-15, bgcolor="white", bordercolor="#F9A825", row=2, col=1)

    # ==========================================
    # ROW 3: BENDING MOMENT (Tension Side + Grid)
    # ==========================================
    df['m_pos'] = df['moment'].apply(lambda x: x if x >= 0 else 0)
    df['m_neg'] = df['moment'].apply(lambda x: x if x < 0 else 0)

    # Plot Lines
    fig.add_trace(go.Scatter(x=df['x'], y=df['m_pos'], mode='lines', line=dict(width=0), fill='tozeroy', fillcolor='rgba(33, 150, 243, 0.3)'), row=3, col=1)
    fig.add_trace(go.Scatter(x=df['x'], y=df['m_neg'], mode='lines', line=dict(width=0), fill='tozeroy', fillcolor='rgba(233, 30, 99, 0.3)'), row=3, col=1)
    fig.add_trace(go.Scatter(x=df['x'], y=df['moment'], mode='lines', line=dict(color='#546E7A', width=2.5), showlegend=False), row=3, col=1)

    # Labels with Backgrounds for readability over grids
    m_max, m_min = df['moment'].max(), df['moment'].min()
    
    if m_max > 1:
        fig.add_annotation(
            x=df.loc[df['moment'].idxmax(), 'x'], y=m_max, 
            text=f"<b>M(+): {m_max:,.0f}</b>", 
            showarrow=True, arrowcolor="#1565C0", yshift=20,
            bgcolor="white", bordercolor="#1565C0", borderwidth=1,
            row=3, col=1
        )
    if abs(m_min) > 1:
        fig.add_annotation(
            x=df.loc[df['moment'].idxmin(), 'x'], y=m_min, 
            text=f"<b>M(-): {m_min:,.0f}</b>", 
            showarrow=True, arrowcolor="#AD1457", yshift=-20,
            bgcolor="white", bordercolor="#AD1457", borderwidth=1,
            row=3, col=1
        )

    # ==========================================
    # ROW 4: DEFLECTION
    # ==========================================
    defl_mm = df['deflection'] * 1000
    fig.add_trace(go.Scatter(
        x=df['x'], y=defl_mm, mode='lines',
        line=dict(color='#43A047', width=2, dash='dot'), name="Deflection"
    ), row=4, col=1)
    
    d_idx = defl_mm.abs().idxmax()
    fig.add_annotation(
        x=df.loc[d_idx, 'x'], y=defl_mm[d_idx],
        text=f"δmax: {defl_mm[d_idx]:.2f} mm",
        showarrow=True, arrowhead=1, bgcolor="white", bordercolor="#43A047", row=4, col=1
    )

    # ==========================================
    # GLOBAL STYLING (The "Pro" Look)
    # ==========================================
    fig.update_layout(
        height=1400, # เพิ่มความสูงรวม
        template="plotly_white",
        hovermode="x unified",
        margin=dict(l=60, r=40, t=60, b=60),
        font=dict(family="Arial, sans-serif", size=13, color="#333"),
        showlegend=False
    )
    
    # GRID STYLING: Make it visible but elegant
    grid_style = dict(
        showgrid=True, 
        gridcolor='#E0E0E0',  # เส้น Grid สีเทาอ่อน-กลาง (เห็นชัดแต่ไม่รก)
        gridwidth=1,
        showline=True, 
        linewidth=1.5, 
        linecolor='#333', 
        mirror=True,
        zeroline=True,
        zerolinecolor='#9E9E9E',
        zerolinewidth=1.5
    )
    
    # Row 1: Hide Y axis labels (visual only), but keep X grid
    fig.update_yaxes(visible=False, row=1, col=1)
    fig.update_xaxes(**grid_style, row=1, col=1) # Show X grid for position ref
    
    # Row 2, 3, 4: Apply Grid Style
    fig.update_yaxes(title="<b>Shear (V)</b>", **grid_style, row=2, col=1)
    fig.update_yaxes(title="<b>Moment (M)</b>", autorange="reversed", **grid_style, row=3, col=1)
    fig.update_yaxes(title="<b>Deflection (mm)</b>", **grid_style, row=4, col=1)
    
    # Bottom X Axis
    fig.update_xaxes(title="<b>Distance (m)</b>", **grid_style, row=4, col=1)
    
    # Link all x-axes grid lines
    fig.update_xaxes(matches='x', showgrid=True, gridcolor='#E0E0E0', row=2, col=1)
    fig.update_xaxes(matches='x', showgrid=True, gridcolor='#E0E0E0', row=3, col=1)

    st.plotly_chart(fig, use_container_width=True)

def render_result_tables(df, reac, spans, u_force, u_len):
    st.markdown("---")
    
    # CSS Styling for clean tables
    st.markdown("""
    <style>
        div[data-testid="stDataFrame"] {font-size: 14px;}
        thead tr th:first-child {display:none}
        tbody th {display:none}
    </style>
    """, unsafe_allow_html=True)
    
    c1, c2 = st.columns([1, 1.5])
    
    with c1:
        st.markdown("##### 📍 Support Reactions")
        if reac is not None:
            r_data = []
            for i in range(len(reac)//2):
                fy = reac[2*i]
                mz = reac[2*i+1]
                if abs(fy)>0.1 or abs(mz)>0.1:
                    r_data.append({
                        "Support Node": f"Node {i}", 
                        "Vertical (kg)": f"{fy:,.2f}", 
                        "Moment (kg-m)": f"{mz:,.2f}"
                    })
            st.dataframe(pd.DataFrame(r_data), use_container_width=True, hide_index=True)
            
    with c2:
        st.markdown("##### 📐 Critical Forces (Envelope)")
        s_data = []
        nodes = [0] + list(np.cumsum(spans))
        for i in range(len(spans)):
            sub = df[(df['x'] >= nodes[i]) & (df['x'] <= nodes[i+1])]
            s_data.append({
                "Span": f"{i+1}",
                "Shear Max": f"{sub['shear'].abs().max():,.0f}",
                "Mom (+)": f"{sub['moment'].max():,.0f}",
                "Mom (-)": f"{sub['moment'].min():,.0f}",
                "Defl (mm)": f"{sub['deflection'].abs().max()*1000:.2f}"
            })
        st.dataframe(pd.DataFrame(s_data), use_container_width=True, hide_index=True)
