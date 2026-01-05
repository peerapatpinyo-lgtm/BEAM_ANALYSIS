import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

def draw_interactive_diagrams(df, reac, spans, sup_df, raw_loads):
    """
    Professional Structural Visualization (Final Fix)
    - Fix: Load positioning now perfectly follows span changes (Global Coordinates).
    - Fix: Visual scaling is locked (Beam always looks solid).
    - Style: Clean Engineering Paper aesthetic.
    """
    # --- 1. PRE-CALCULATE GLOBAL COORDINATES ---
    # สร้าง list จุดต่อ (Nodes) สะสมระยะทาง เช่น [0, 4, 9, 12]
    nodes = [0] + list(np.cumsum(spans))
    total_len = nodes[-1]
    
    # --- 2. VISUAL CONFIGURATION (LOCK SCALES) ---
    # กำหนดหน่วยสมมติในแกน Y เพื่อวาดกราฟิกคาน (ไม่เกี่ยวกับค่า Load จริง)
    # เพื่อให้คานดูสมส่วนตลอดเวลา ไม่ว่าจะยาว 5m หรือ 20m
    VISUAL_Y_RANGE = [-1.5, 3.5] # ล็อคความสูงหน้าจอ Row 1
    BEAM_THICKNESS = 0.6
    BEAM_Y_TOP = BEAM_THICKNESS / 2
    BEAM_Y_BOT = -BEAM_THICKNESS / 2
    
    # Load Scaling Limit (ความสูงกราฟิกสูงสุดของ Load)
    MAX_LOAD_HEIGHT = 1.8 
    
    # หาค่า Max Load เพื่อทำ Normalization (แยกตามประเภท)
    p_vals = [l['mag'] for l in raw_loads if l['type']=='P']
    u_vals = [l['mag'] for l in raw_loads if l['type']=='U']
    max_p = max(p_vals) if p_vals else 1.0
    max_u = max(u_vals) if u_vals else 1.0

    # --- 3. CREATE SUBPLOTS ---
    fig = make_subplots(
        rows=4, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.05,
        row_heights=[0.30, 0.23, 0.25, 0.22], # ให้พื้นที่รูปคานเยอะหน่อย
        subplot_titles=(
            "📌 Free Body Diagram", 
            "⚡ Shear Force (V)", 
            "🔄 Bending Moment (Tension Side)", 
            "📉 Deflection (δ)"
        )
    )

    # ==========================================
    # ROW 1: REALISTIC BEAM & LOADS (The "Pro" Look)
    # ==========================================
    
    # 1.1 วาดคาน (Solid Block)
    fig.add_shape(type="rect",
        x0=0, x1=total_len, y0=BEAM_Y_BOT, y1=BEAM_Y_TOP,
        fillcolor="#E0E0E0", line=dict(color="#263238", width=2.5),
        layer="below", row=1, col=1
    )
    # Centerline (เส้นประกลางคาน)
    fig.add_trace(go.Scatter(
        x=[0, total_len], y=[0, 0], mode="lines",
        line=dict(color="#90A4AE", width=1, dash="dashdot"),
        hoverinfo="skip", showlegend=False
    ), row=1, col=1)

    # 1.2 วาด Supports (Auto Position fix)
    if not sup_df.empty:
        for _, s in sup_df.iterrows():
            # ดึงตำแหน่ง X จาก nodes list โดยตรง (แก้บั๊กตำแหน่งเพี้ยน)
            node_idx = int(s['id'])
            if node_idx < len(nodes):
                x = nodes[node_idx]
                stype = s['type']
                
                if stype == "Pin":
                    # สามเหลี่ยม
                    fig.add_trace(go.Scatter(
                        x=[x], y=[BEAM_Y_BOT], mode="markers",
                        marker=dict(symbol="triangle-up", size=14, color="#37474F"),
                        hoverinfo="text", text="Pin", showlegend=False
                    ), row=1, col=1)
                    # ขีดล่าง
                    fig.add_shape(type="line", x0=x-0.2, x1=x+0.2, y0=BEAM_Y_BOT-0.15, y1=BEAM_Y_BOT-0.15,
                                  line=dict(color="#37474F", width=2), row=1, col=1)
                    
                elif stype == "Roller":
                    # วงกลม
                    fig.add_trace(go.Scatter(
                        x=[x], y=[BEAM_Y_BOT - 0.15], mode="markers",
                        marker=dict(symbol="circle", size=12, color="#37474F", line=dict(width=1.5, color="white")),
                        showlegend=False
                    ), row=1, col=1)
                    # ขีดล่าง
                    fig.add_shape(type="line", x0=x-0.2, x1=x+0.2, y0=BEAM_Y_BOT-0.35, y1=BEAM_Y_BOT-0.35,
                                  line=dict(color="#37474F", width=2), row=1, col=1)

                elif stype == "Fixed":
                    fig.add_shape(type="line", x0=x, x1=x, y0=BEAM_Y_BOT-0.3, y1=BEAM_Y_TOP+0.3,
                        line=dict(color="#37474F", width=4), row=1, col=1)
                    # แรเงา
                    hatch_dir = 1 if x == 0 else -1
                    for h in np.linspace(BEAM_Y_BOT-0.3, BEAM_Y_TOP+0.3, 6):
                        fig.add_shape(type="line", 
                            x0=x, y0=h, x1=x - (0.25 * hatch_dir), y1=h - 0.1,
                            line=dict(color="#37474F", width=1), row=1, col=1)

    # 1.3 วาด Loads (Correct Global Positioning)
    for l in raw_loads:
        span_idx = int(l['span_idx'])
        if span_idx >= len(spans): continue # ป้องกัน Error ถ้าลบ Span แล้ว Load ยังค้าง
        
        # --- CRITICAL FIX: คำนวณ Global X ---
        start_node_x = nodes[span_idx]
        global_x_start = start_node_x + l['x']
        
        color = "#C62828" if l['case'] == 'LL' else "#1565C0" # Red for Live, Blue for Dead
        fill_c = "rgba(198, 40, 40, 0.15)" if l['case'] == 'LL' else "rgba(21, 101, 192, 0.15)"

        if l['type'] == 'P':
            # Height calculation
            ratio = l['mag'] / max_p if max_p > 0 else 1
            h_vis = MAX_LOAD_HEIGHT * (0.5 + 0.5 * ratio)
            
            fig.add_annotation(
                x=global_x_start, y=BEAM_Y_TOP,
                ax=0, ay=-h_vis*35, # Scale pixel length
                xref="x1", yref="y1",
                text=f"<b>P={l['mag']:,.0f}</b>",
                showarrow=True, arrowhead=2, arrowwidth=2, arrowcolor=color,
                font=dict(color=color, size=11, family="Arial"),
                bgcolor="white", bordercolor=color, borderwidth=1, borderpad=2,
                row=1, col=1
            )
            
        elif l['type'] == 'U':
            # Global X End
            span_len = spans[span_idx]
            local_end = l.get('end', span_len)
            global_x_end = start_node_x + local_end
            
            ratio = l['mag'] / max_u if max_u > 0 else 1
            h_vis = MAX_LOAD_HEIGHT * (0.4 + 0.6 * ratio) # U load เตี้ยกว่า P นิดนึง
            
            # 1. Box Area
            fig.add_trace(go.Scatter(
                x=[global_x_start, global_x_end, global_x_end, global_x_start],
                y=[BEAM_Y_TOP, BEAM_Y_TOP, BEAM_Y_TOP + h_vis, BEAM_Y_TOP + h_vis],
                fill='toself', fillcolor=fill_c,
                mode='none', hoverinfo='skip', showlegend=False
            ), row=1, col=1)
            
            # 2. Top Line
            fig.add_trace(go.Scatter(
                x=[global_x_start, global_x_end], 
                y=[BEAM_Y_TOP + h_vis, BEAM_Y_TOP + h_vis],
                mode='lines', line=dict(color=color, width=2),
                hoverinfo='skip', showlegend=False
            ), row=1, col=1)
            
            # 3. Arrows (Smart Density)
            dist = global_x_end - global_x_start
            n_arrows = max(3, int(dist * 1.5)) # ปรับจำนวนลูกศรตามความยาวจริง
            for ax in np.linspace(global_x_start, global_x_end, n_arrows):
                fig.add_shape(type="line",
                    x0=ax, x1=ax, y0=BEAM_Y_TOP, y1=BEAM_Y_TOP + h_vis,
                    line=dict(color=color, width=1), layer="below", row=1, col=1
                )
                fig.add_trace(go.Scatter(
                    x=[ax], y=[BEAM_Y_TOP],
                    mode='markers', marker=dict(symbol="triangle-down", size=6, color=color),
                    hoverinfo='skip', showlegend=False
                ), row=1, col=1)
            
            # Label Center
            fig.add_annotation(
                x=(global_x_start + global_x_end)/2, y=BEAM_Y_TOP + h_vis,
                text=f"<b>w={l['mag']:,.0f}</b>",
                yshift=10, showarrow=False,
                font=dict(color=color, size=11), bgcolor="white", opacity=0.9,
                row=1, col=1
            )

    # --- LOCK ROW 1 SCALING ---
    fig.update_yaxes(range=VISUAL_Y_RANGE, fixedrange=True, visible=False, row=1, col=1)


    # ==========================================
    # ROW 2: SHEAR FORCE (Professional)
    # ==========================================
    fig.add_trace(go.Scatter(
        x=df['x'], y=df['shear'], mode='lines', line_shape='hv',
        fill='tozeroy', line=dict(color='#F57F17', width=2), # Amber Dark
        fillcolor='rgba(255, 179, 0, 0.2)', name="Shear"
    ), row=2, col=1)
    
    # Annotate Max/Min with background box (อ่านง่าย ไม่จม)
    v_max, v_min = df['shear'].max(), df['shear'].min()
    fig.add_annotation(x=df.loc[df['shear'].idxmax(), 'x'], y=v_max, text=f"<b>{v_max:.0f}</b>", 
        showarrow=False, yshift=15, bgcolor="white", bordercolor="#F57F17", borderwidth=1, font=dict(color="#E65100"), row=2, col=1)
    fig.add_annotation(x=df.loc[df['shear'].idxmin(), 'x'], y=v_min, text=f"<b>{v_min:.0f}</b>", 
        showarrow=False, yshift=-15, bgcolor="white", bordercolor="#F57F17", borderwidth=1, font=dict(color="#E65100"), row=2, col=1)


    # ==========================================
    # ROW 3: MOMENT (Tension Side + Clear Zones)
    # ==========================================
    df['m_pos'] = df['moment'].apply(lambda x: x if x >= 0 else 0)
    df['m_neg'] = df['moment'].apply(lambda x: x if x < 0 else 0)

    # Sagging Zone (+M) -> Down
    fig.add_trace(go.Scatter(x=df['x'], y=df['m_pos'], mode='lines', line=dict(width=0), 
        fill='tozeroy', fillcolor='rgba(33, 150, 243, 0.3)', name="Pos M"), row=3, col=1)
    # Hogging Zone (-M) -> Up
    fig.add_trace(go.Scatter(x=df['x'], y=df['m_neg'], mode='lines', line=dict(width=0), 
        fill='tozeroy', fillcolor='rgba(233, 30, 99, 0.3)', name="Neg M"), row=3, col=1)
    # Main Line
    fig.add_trace(go.Scatter(x=df['x'], y=df['moment'], mode='lines', 
        line=dict(color='#455A64', width=2), showlegend=False), row=3, col=1)

    # Labels
    m_max, m_min = df['moment'].max(), df['moment'].min()
    if m_max > 10: # Only show if significant
        fig.add_annotation(
            x=df.loc[df['moment'].idxmax(), 'x'], y=m_max, 
            text=f"<b>M(+): {m_max:,.0f}</b>", 
            showarrow=True, arrowcolor="#1565C0", yshift=20,
            bgcolor="white", bordercolor="#1565C0", borderwidth=1, font=dict(color="#1565C0"), row=3, col=1
        )
    if abs(m_min) > 10:
        fig.add_annotation(
            x=df.loc[df['moment'].idxmin(), 'x'], y=m_min, 
            text=f"<b>M(-): {m_min:,.0f}</b>", 
            showarrow=True, arrowcolor="#C2185B", yshift=-20,
            bgcolor="white", bordercolor="#C2185B", borderwidth=1, font=dict(color="#C2185B"), row=3, col=1
        )


    # ==========================================
    # ROW 4: DEFLECTION
    # ==========================================
    defl_mm = df['deflection'] * 1000
    fig.add_trace(go.Scatter(
        x=df['x'], y=defl_mm, mode='lines',
        line=dict(color='#2E7D32', width=2, dash='dot'), name="Deflection"
    ), row=4, col=1)
    
    d_idx = defl_mm.abs().idxmax()
    if abs(defl_mm[d_idx]) > 0.01:
        fig.add_annotation(
            x=df.loc[d_idx, 'x'], y=defl_mm[d_idx],
            text=f"δmax: {defl_mm[d_idx]:.2f} mm",
            showarrow=True, arrowhead=1, bgcolor="white", bordercolor="#2E7D32", row=4, col=1
        )

    # ==========================================
    # GLOBAL LAYOUT SETTINGS
    # ==========================================
    # Grid Style: Engineering Paper Look
    grid_config = dict(
        showgrid=True, gridcolor='#ECEFF1', gridwidth=1,
        showline=True, linecolor='#546E7A', linewidth=1.5,
        mirror=True, zeroline=True, zerolinecolor='#CFD8DC'
    )

    fig.update_layout(
        height=1400,
        template="plotly_white",
        hovermode="x unified",
        margin=dict(l=60, r=40, t=60, b=60),
        font=dict(family="Roboto, Arial, sans-serif", size=12, color="#37474F"),
        showlegend=False
    )
    
    # Apply Axis Styles
    fig.update_xaxes(title="", **grid_config, row=1, col=1) # แค่โชว์ Grid X ให้รู้ระยะ
    fig.update_yaxes(title="<b>Shear (V)</b>", **grid_config, row=2, col=1)
    fig.update_yaxes(title="<b>Moment (M)</b>", autorange="reversed", **grid_config, row=3, col=1)
    fig.update_yaxes(title="<b>Deflection (mm)</b>", **grid_config, row=4, col=1)
    fig.update_xaxes(title="<b>Distance (m)</b>", **grid_config, row=4, col=1)
    
    # Link X-axes
    fig.update_xaxes(matches='x', row=2, col=1)
    fig.update_xaxes(matches='x', row=3, col=1)

    st.plotly_chart(fig, use_container_width=True)


def render_result_tables(df, reac, spans, u_force, u_len):
    st.markdown("---")
    
    # CSS ให้ตารางดูสะอาดตา
    st.markdown("""
    <style>
        div[data-testid="stDataFrame"] {font-family: 'Roboto', sans-serif;}
    </style>
    """, unsafe_allow_html=True)
    
    c1, c2 = st.columns([1, 1.5])
    
    with c1:
        st.subheader("📍 Support Reactions")
        if reac is not None:
            r_data = []
            for i in range(len(reac)//2):
                fy = reac[2*i]
                mz = reac[2*i+1]
                if abs(fy)>0.01 or abs(mz)>0.01:
                    r_data.append({
                        "Node": f"{i}", 
                        "Fy (kg)": f"{fy:,.2f}", 
                        "Mz (kg-m)": f"{mz:,.2f}"
                    })
            st.dataframe(pd.DataFrame(r_data), use_container_width=True, hide_index=True)
            
    with c2:
        st.subheader("📐 Critical Design Forces")
        s_data = []
        nodes = [0] + list(np.cumsum(spans))
        for i in range(len(spans)):
            sub = df[(df['x'] >= nodes[i]) & (df['x'] <= nodes[i+1])]
            s_data.append({
                "Span": f"{i+1}",
                "V max": f"{sub['shear'].abs().max():,.0f}",
                "M(+)": f"{sub['moment'].max():,.0f}",
                "M(-)": f"{sub['moment'].min():,.0f}",
                "δ (mm)": f"{sub['deflection'].abs().max()*1000:.2f}"
            })
        st.dataframe(pd.DataFrame(s_data), use_container_width=True, hide_index=True)
