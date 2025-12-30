import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- THEME CONSTANTS (ENGINEERING STYLE) ---
C_BEAM = 'black'
C_PIN_ROLLER = 'white' 
C_OUTLINE = 'black'
C_SHEAR_LINE = '#D97706'    # Amber Dark
C_SHEAR_FILL = 'rgba(217, 119, 6, 0.15)'
C_MOMENT_LINE = '#2563EB'   # Blue Dark
C_MOMENT_FILL = 'rgba(37, 99, 235, 0.15)'
C_TEXT_BG = 'white'         # พื้นหลังตัวหนังสือ (กันเส้นทับ)
C_TEXT_BORDER = '#E5E7EB'   # ขอบจางๆ รอบตัวหนังสือ

def draw_interactive_diagrams(df, spans, sup_df, loads, unit_force, unit_len):
    total_len = sum(spans)
    cum_spans = [0] + list(np.cumsum(spans))
    n_nodes = len(cum_spans)
    
    # --- 1. LAYOUT SETUP ---
    # ปรับสัดส่วนใหม่: ลดความสูง Model ลงเล็กน้อย เพื่อให้กราฟผลลัพธ์ใหญ่ขึ้น
    fig = make_subplots(
        rows=3, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.10, 
        row_heights=[0.20, 0.40, 0.40], # เน้นกราฟ SFD/BMD ให้ใหญ่
        subplot_titles=(
            "<b>1. Structural Model (FBD)</b>", 
            "<b>2. Shear Force Diagram (SFD)</b>", 
            "<b>3. Bending Moment Diagram (BMD)</b>"
        )
    )

    # ==========================================
    # ROW 1: STRUCTURAL MODEL
    # ==========================================
    
    # 1.1 Beam Line
    fig.add_trace(go.Scatter(
        x=[0, total_len], y=[0, 0],
        mode='lines', line=dict(color=C_BEAM, width=5),
        hoverinfo='skip'
    ), row=1, col=1)

    # 1.2 Nodes & Supports
    sup_map = {int(r['id']): r['type'] for _, r in sup_df.iterrows()}

    for i in range(n_nodes):
        x = cum_spans[i]
        
        if i in sup_map:
            sType = sup_map[i]
            if sType == 'Pin':
                fig.add_trace(go.Scatter(
                    x=[x], y=[-0.15], 
                    mode='markers',
                    marker=dict(symbol='triangle-up', size=18, color=C_PIN_ROLLER, line=dict(width=2, color=C_OUTLINE)),
                    hoverinfo='name', name=f"Pin"
                ), row=1, col=1)
                # Ground
                fig.add_shape(type="line", x0=x-0.2, y0=-0.25, x1=x+0.2, y1=-0.25, line=dict(color='black', width=2), row=1, col=1)
                for hx in np.linspace(x-0.2, x+0.2, 5):
                    fig.add_shape(type="line", x0=hx, y0=-0.25, x1=hx-0.05, y1=-0.30, line=dict(color='black', width=1), row=1, col=1)

            elif sType == 'Roller':
                fig.add_trace(go.Scatter(
                    x=[x], y=[-0.15],
                    mode='markers',
                    marker=dict(symbol='circle', size=16, color=C_PIN_ROLLER, line=dict(width=2, color=C_OUTLINE)),
                    hoverinfo='name', name=f"Roller"
                ), row=1, col=1)
                # Ground
                fig.add_shape(type="line", x0=x-0.2, y0=-0.25, x1=x+0.2, y1=-0.25, line=dict(color='black', width=2), row=1, col=1)
                for hx in np.linspace(x-0.2, x+0.2, 5):
                    fig.add_shape(type="line", x0=hx, y0=-0.25, x1=hx-0.05, y1=-0.30, line=dict(color='black', width=1), row=1, col=1)

            elif sType == 'Fixed':
                fig.add_shape(type="line", x0=x, y0=-0.3, x1=x, y1=0.3, line=dict(color='black', width=4), row=1, col=1)
                h_dir = -0.15 if x == 0 else 0.15 
                for hy in np.linspace(-0.3, 0.3, 7):
                    fig.add_shape(type="line", x0=x, y0=hy, x1=x+h_dir, y1=hy-0.05, line=dict(color='black', width=1), row=1, col=1)
        else:
            # Internal Node
            fig.add_trace(go.Scatter(
                x=[x], y=[0],
                mode='markers',
                marker=dict(symbol='circle', size=10, color='white', line=dict(width=2, color='black')),
                hoverinfo='name', name=f"Node {i+1}"
            ), row=1, col=1)

    # 1.3 Loads (ใส่กล่องขาวบังเส้นคาน)
    for l in loads:
        x_s = cum_spans[int(l['span_idx'])]
        if l['type'] == 'P':
            # Point Load
            text_label = f"<b>P={l['P']} {unit_force}</b>"
            fig.add_annotation(
                x=x_s + l['x'], y=0, ax=0, ay=-55, # ดึงลูกศรขึ้นสูงหน่อย
                arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor='#DC2626',
                text=text_label, font=dict(color='#DC2626', size=12),
                bgcolor=C_TEXT_BG, bordercolor=C_TEXT_BORDER, borderwidth=1, opacity=1, # Masking
                row=1, col=1
            )
        elif l['type'] == 'U':
            # UDL
            x_e = cum_spans[int(l['span_idx'])+1]
            text_label = f"<b>w={l['w']} {unit_force}/{unit_len}</b>"
            
            fig.add_shape(type="rect", x0=x_s, y0=0.15, x1=x_e, y1=0.3, line_width=0, fillcolor='#DC2626', opacity=0.15, row=1, col=1)
            fig.add_annotation(
                x=(x_s+x_e)/2, y=0.35, showarrow=False, 
                text=text_label, font=dict(color='#DC2626', size=12),
                bgcolor=C_TEXT_BG, bordercolor=C_TEXT_BORDER, borderwidth=1, opacity=1, # Masking
                row=1, col=1
            )

    # ==========================================
    # ROW 2: SHEAR FORCE (SFD)
    # ==========================================
    fig.add_shape(type="line", x0=0, x1=total_len, y0=0, y1=0, line=dict(color='black', width=1.5), row=2, col=1)
    
    fig.add_trace(go.Scatter(
        x=df['x'], y=df['shear'],
        mode='lines', line=dict(color=C_SHEAR_LINE, width=2.5),
        fill='tozeroy', fillcolor=C_SHEAR_FILL,
        name='Shear Force'
    ), row=2, col=1)

    # --- Smart Labeling for Shear ---
    v_max = df['shear'].abs().max()
    if v_max > 0:
        idx = df['shear'].abs().idxmax()
        row_v = df.loc[idx]
        val = row_v['shear']
        x_loc = row_v['x']
        
        # Logic: ถ้าค่าบวก ให้ตัวหนังสืออยู่บน ถ้าลบอยู่ล่าง
        # เพิ่ม yshift ให้เยอะขึ้น เพื่อหนีเส้นกราฟ
        yshift = 25 if val > 0 else -25 
        
        # Logic: จัด Anchor ซ้าย/ขวา เพื่อไม่ให้ตกขอบจอ
        xanchor = 'center'
        if x_loc < total_len * 0.1: xanchor = 'left'
        elif x_loc > total_len * 0.9: xanchor = 'right'

        fig.add_annotation(
            x=x_loc, y=val, 
            text=f"<b>{val:.2f} @ {x_loc:.2f}m</b>",
            showarrow=False, 
            yshift=yshift,
            xanchor=xanchor,
            font=dict(color=C_SHEAR_LINE, size=12), 
            bgcolor=C_TEXT_BG, bordercolor=C_TEXT_BORDER, borderwidth=1, opacity=1.0, # Box Masking
            row=2, col=1
        )

    # ==========================================
    # ROW 3: BENDING MOMENT (BMD)
    # ==========================================
    fig.add_shape(type="line", x0=0, x1=total_len, y0=0, y1=0, line=dict(color='black', width=1.5), row=3, col=1)

    fig.add_trace(go.Scatter(
        x=df['x'], y=df['moment'],
        mode='lines', line=dict(color=C_MOMENT_LINE, width=2.5),
        fill='tozeroy', fillcolor=C_MOMENT_FILL,
        name='Bending Moment'
    ), row=3, col=1)

    # --- Smart Labeling for Moment ---
    m_max = df['moment'].max() # บวก
    m_min = df['moment'].min() # ลบ
    
    # Label Max (+)
    if m_max > 0.01:
        xm = df.loc[df['moment'] == m_max, 'x'].iloc[0]
        xanchor = 'center'
        if xm < total_len * 0.1: xanchor = 'left'
        elif xm > total_len * 0.9: xanchor = 'right'
        
        fig.add_annotation(
            x=xm, y=m_max, 
            text=f"<b>Max(+): {m_max:.2f}</b>", # ตัด @ ออกเพื่อให้สั้นลง หรือจะใส่ก็ได้
            showarrow=False, yshift=25, xanchor=xanchor,
            font=dict(color=C_MOMENT_LINE, size=12),
            bgcolor=C_TEXT_BG, bordercolor=C_TEXT_BORDER, borderwidth=1, opacity=1.0,
            row=3, col=1
        )

    # Label Min (-)
    if m_min < -0.01:
        xm = df.loc[df['moment'] == m_min, 'x'].iloc[0]
        xanchor = 'center'
        if xm < total_len * 0.1: xanchor = 'left'
        elif xm > total_len * 0.9: xanchor = 'right'
        
        fig.add_annotation(
            x=xm, y=m_min, 
            text=f"<b>Max(-): {m_min:.2f}</b>",
            showarrow=False, yshift=-25, xanchor=xanchor,
            font=dict(color=C_MOMENT_LINE, size=12),
            bgcolor=C_TEXT_BG, bordercolor=C_TEXT_BORDER, borderwidth=1, opacity=1.0,
            row=3, col=1
        )

    # ==========================================
    # GLOBAL LAYOUT
    # ==========================================
    fig.update_layout(
        height=950, # เพิ่มความสูงรวม
        margin=dict(l=80, r=40, t=50, b=50),
        plot_bgcolor='white',
        paper_bgcolor='white',
        showlegend=False,
        hovermode="x unified",
        font=dict(family="Sarabun, sans-serif", size=13, color='black')
    )

    # Grid Style (จางๆ)
    ax_style = dict(
        showline=True, linewidth=1.5, linecolor='black',
        showgrid=True, gridcolor='#F3F4F6',
        ticks="outside", tickwidth=1.5, ticklen=5,
        mirror=True
    )

    fig.update_xaxes(**ax_style)
    fig.update_yaxes(**ax_style)

    # Force Titles (ใส่หน่วย)
    fig.update_yaxes(visible=False, row=1, col=1, range=[-0.6, 0.6])
    fig.update_xaxes(visible=True, showticklabels=True, title_text="", row=1, col=1)

    fig.update_yaxes(title_text=f"<b>Shear Force<br>(V) [{unit_force}]</b>", title_standoff=15, row=2, col=1)
    
    fig.update_yaxes(title_text=f"<b>Bending Moment<br>(M) [{unit_force}-{unit_len}]</b>", title_standoff=15, row=3, col=1)
    fig.update_xaxes(title_text=f"<b>Distance (x) [{unit_len}]</b>", row=3, col=1)

    st.plotly_chart(fig, use_container_width=True)

def render_result_tables(df, reac, spans):
    st.markdown("---")
    
    # Summary
    c1, c2, c3 = st.columns(3)
    c1.metric("Max Shear", f"{df['shear'].abs().max():.2f}")
    c2.metric("Max Moment (+)", f"{df['moment'].max():.2f}")
    c3.metric("Max Moment (-)", f"{df['moment'].min():.2f}")

    # Reactions Table
    st.write("#### 📍 Support Reactions")
    data = []
    n_nodes = len(spans) + 1
    for i in range(n_nodes):
        ry = reac[2*i]
        mz = reac[2*i+1]
        data.append({"Node": i+1, "Ry": f"{ry:.2f}", "Mz": f"{mz:.2f}" if abs(mz)>0.001 else "-"})
    st.dataframe(pd.DataFrame(data), hide_index=True, use_container_width=True)
