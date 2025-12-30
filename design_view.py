import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- ENGINEERING THEME COLORS ---
C_BG = 'white'
C_AXIS = '#000000'       # สีแกนดำสนิท
C_GRID = '#E0E0E0'       # Grid เทาจางๆ
C_BEAM = '#000000'       # คานสีดำเข้ม
C_SUP_FILL = '#FFFFFF'   # Support สีขาวขอบดำ
C_SHEAR_LINE = '#D97706' # ส้ม
C_SHEAR_FILL = 'rgba(217, 119, 6, 0.2)'
C_MOMENT_LINE = '#2563EB' # น้ำเงิน
C_MOMENT_FILL = 'rgba(37, 99, 235, 0.2)'

def draw_interactive_diagrams(df, spans, sup_df, loads, unit_force, unit_len):
    total_len = sum(spans)
    cum_spans = [0] + list(np.cumsum(spans))
    
    # Create Subplots (3 Rows)
    fig = make_subplots(
        rows=3, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.1,
        row_heights=[0.3, 0.35, 0.35],
        subplot_titles=(
            "<b>1. Structural Model (FBD)</b>", 
            f"<b>2. Shear Force Diagram (V) [{unit_force}]</b>", 
            f"<b>3. Bending Moment Diagram (M) [{unit_force}-{unit_len}]</b>"
        )
    )

    # ==========================================
    # 1. STRUCTURAL MODEL (FBD)
    # ==========================================
    # 1.1 วาดเส้นคาน (Beam Line)
    fig.add_trace(go.Scatter(
        x=[0, total_len], y=[0, 0],
        mode='lines', line=dict(color=C_BEAM, width=6),
        hoverinfo='skip'
    ), row=1, col=1)

    # 1.2 วาด Supports และ Nodes
    # วนลูปทุก Node ที่มี (ตั้งแต่ 0 ถึง node สุดท้าย)
    n_nodes = len(cum_spans)
    
    # แปลง sup_df เป็น Dict เพื่อค้นหาง่ายๆ
    sup_dict = {int(r['id']): r['type'] for _, r in sup_df.iterrows()}

    for i in range(n_nodes):
        x_pos = cum_spans[i]
        
        if i in sup_dict:
            sType = sup_dict[i]
            # --- Draw Supports ---
            if sType == 'Pin':
                # Pin: Triangle + Ground Line
                fig.add_trace(go.Scatter(
                    x=[x_pos], y=[-0.15], # ขยับลงนิดนึงให้ยอดแตะคาน
                    mode='markers',
                    marker=dict(symbol='triangle-up', size=25, color=C_SUP_FILL, line=dict(width=2, color='black')),
                    hoverinfo='text', text=f"Pin (Node {i+1})", showlegend=False
                ), row=1, col=1)
                # Ground Line
                fig.add_shape(type="line", x0=x_pos-0.2, y0=-0.25, x1=x_pos+0.2, y1=-0.25, line=dict(color='black', width=3), row=1, col=1)

            elif sType == 'Roller':
                # Roller: Circle + Ground Line
                fig.add_trace(go.Scatter(
                    x=[x_pos], y=[-0.15],
                    mode='markers',
                    marker=dict(symbol='circle', size=22, color=C_SUP_FILL, line=dict(width=2, color='black')),
                    hoverinfo='text', text=f"Roller (Node {i+1})", showlegend=False
                ), row=1, col=1)
                # Ground Line
                fig.add_shape(type="line", x0=x_pos-0.2, y0=-0.25, x1=x_pos+0.2, y1=-0.25, line=dict(color='black', width=3), row=1, col=1)

            elif sType == 'Fixed':
                # Fixed: Vertical Line at support
                fig.add_shape(type="line", x0=x_pos, y0=-0.3, x1=x_pos, y1=0.3, line=dict(color='black', width=5), row=1, col=1)
                # Hatch marks (ขีดๆ แสดงกำแพง)
                for h in np.linspace(-0.3, 0.3, 5):
                    direction = 0.15 if x_pos == 0 else -0.15 # ขีดออกด้านนอก
                    fig.add_shape(type="line", x0=x_pos, y0=h, x1=x_pos+direction, y1=h-0.05, line=dict(color='black', width=1), row=1, col=1)
                
        else:
            # --- Draw No Support (Internal Node) ---
            # จุดดำเล็ก แสดงรอยต่อช่วงคาน หรือจุด Load
            fig.add_trace(go.Scatter(
                x=[x_pos], y=[0],
                mode='markers',
                marker=dict(symbol='circle', size=8, color='black'),
                hoverinfo='text', text=f"Node {i+1} (No Support)", showlegend=False
            ), row=1, col=1)

    # 1.3 Loads Annotation
    for l in loads:
        x_s = cum_spans[int(l['span_idx'])]
        if l['type'] == 'P':
            fig.add_annotation(
                x=x_s + l['x'], y=0, ax=0, ay=-50,
                arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor='#DC2626',
                text=f"<b>P={l['P']}</b>", font=dict(color='#DC2626', size=12),
                bgcolor="white", row=1, col=1
            )
        elif l['type'] == 'U':
            x_e = cum_spans[int(l['span_idx'])+1]
            # Draw Load Block
            fig.add_shape(type="rect", x0=x_s, y0=0.1, x1=x_e, y1=0.25, line_width=0, fillcolor='#DC2626', opacity=0.15, row=1, col=1)
            # Label
            fig.add_annotation(x=(x_s+x_e)/2, y=0.35, showarrow=False, text=f"<b>w={l['w']}</b>", font=dict(color='#DC2626'), row=1, col=1)

    # ==========================================
    # 2. SHEAR FORCE (SFD)
    # ==========================================
    # Zero Line (Black & Thick)
    fig.add_shape(type="line", x0=0, x1=total_len, y0=0, y1=0, line=dict(color='black', width=2), row=2, col=1)
    
    fig.add_trace(go.Scatter(
        x=df['x'], y=df['shear'],
        mode='lines', line=dict(color=C_SHEAR_LINE, width=2.5),
        fill='tozeroy', fillcolor=C_SHEAR_FILL,
        name='Shear'
    ), row=2, col=1)

    # Label Logic (Value @ Pos)
    v_max = df['shear'].abs().max()
    if v_max > 0:
        idx = df['shear'].abs().idxmax()
        row_v = df.loc[idx]
        val, lx = row_v['shear'], row_v['x']
        yshift = 20 if val >= 0 else -20
        fig.add_annotation(
            x=lx, y=val, text=f"<b>{val:.2f} @ {lx:.2f}m</b>",
            showarrow=False, yshift=yshift,
            font=dict(color=C_SHEAR_LINE, size=12), bgcolor="rgba(255,255,255,0.8)",
            row=2, col=1
        )

    # ==========================================
    # 3. BENDING MOMENT (BMD)
    # ==========================================
    fig.add_shape(type="line", x0=0, x1=total_len, y0=0, y1=0, line=dict(color='black', width=2), row=3, col=1)

    fig.add_trace(go.Scatter(
        x=df['x'], y=df['moment'],
        mode='lines', line=dict(color=C_MOMENT_LINE, width=2.5),
        fill='tozeroy', fillcolor=C_MOMENT_FILL,
        name='Moment'
    ), row=3, col=1)

    # Labels
    m_max = df['moment'].max()
    m_min = df['moment'].min()
    
    if abs(m_max) > 0.01:
        xm = df.loc[df['moment'] == m_max, 'x'].iloc[0]
        fig.add_annotation(
            x=xm, y=m_max, text=f"<b>M(+): {m_max:.2f} @ {xm:.2f}m</b>",
            showarrow=False, yshift=20,
            font=dict(color=C_MOMENT_LINE, size=12), bgcolor="rgba(255,255,255,0.8)",
            row=3, col=1
        )
    if abs(m_min) > 0.01:
        xm = df.loc[df['moment'] == m_min, 'x'].iloc[0]
        fig.add_annotation(
            x=xm, y=m_min, text=f"<b>M(-): {m_min:.2f} @ {xm:.2f}m</b>",
            showarrow=False, yshift=-20,
            font=dict(color=C_MOMENT_LINE, size=12), bgcolor="rgba(255,255,255,0.8)",
            row=3, col=1
        )

    # ==========================================
    # GLOBAL LAYOUT & AXIS RESTORATION
    # ==========================================
    fig.update_layout(
        height=900,
        margin=dict(l=60, r=40, t=60, b=60),
        paper_bgcolor='white', plot_bgcolor='white',
        showlegend=False,
        hovermode="x unified",
        font=dict(family="Sarabun, sans-serif", size=13, color='black')
    )

    # *** AXIS CONFIGURATION (แกน X Y ต้องชัด) ***
    axis_config = dict(
        visible=True,           # บังคับให้แสดง
        showline=True,          # แสดงเส้นแกน
        linewidth=2,            # เส้นหนา 2px
        linecolor='black',      # สีดำสนิท
        showgrid=True,          # แสดง Grid
        gridcolor=C_GRID,       # สี Grid จางๆ
        gridwidth=1,
        mirror=True,            # ให้มีเส้นกรอบล้อมรอบ (Box Style)
        showticklabels=True,    # แสดงตัวเลข
        ticks="outside",        # ขีดสเกลยื่นออกมาข้างนอก
        tickwidth=2,
        tickcolor='black',
        ticklen=5
    )

    # Apply to all axes first
    fig.update_xaxes(**axis_config)
    fig.update_yaxes(**axis_config)
    
    # Specific Tweaks
    # Row 1 (Model): ซ่อนแกน Y (เพราะเป็นรูปภาพ) แต่แกน X ให้มี Tick Mark ไว้ดูระยะ
    fig.update_yaxes(visible=False, row=1, col=1, range=[-0.8, 0.8]) 
    fig.update_xaxes(title_text="", row=1, col=1) # ไม่ต้องมี Title ซ้ำซ้อน

    # Row 2 & 3: ใส่ Title แกน Y
    fig.update_yaxes(title_text=f"V ({unit_force})", row=2, col=1)
    fig.update_yaxes(title_text=f"M ({unit_force}-{unit_len})", row=3, col=1)
    fig.update_xaxes(title_text=f"Distance ({unit_len})", row=3, col=1)

    st.plotly_chart(fig, use_container_width=True)

def render_result_tables(df, reac, spans):
    st.markdown("---")
    st.subheader("📊 Analysis Summary")
    
    # Cards
    c1, c2, c3 = st.columns(3)
    c1.metric("Max Shear", f"{df['shear'].abs().max():.2f}")
    c2.metric("Max Moment (+)", f"{df['moment'].max():.2f}")
    c3.metric("Max Moment (-)", f"{df['moment'].min():.2f}")

    # Reaction Table
    st.write("##### Support Reactions")
    data = []
    n_nodes = len(spans) + 1
    for i in range(n_nodes):
        ry = reac[2*i]
        mz = reac[2*i+1]
        data.append({
            "Node": i+1,
            "Reaction Ry": f"{ry:.2f}",
            "Reaction Mz": f"{mz:.2f}" if abs(mz) > 0.001 else "-"
        })
    st.dataframe(pd.DataFrame(data), hide_index=True, use_container_width=True)
    
    # Data Table
    st.write("##### Internal Forces Table")
    df_export = df.drop_duplicates(subset=['x'], keep='first').copy()
    st.dataframe(df_export.style.format("{:.2f}"), use_container_width=True, height=300)
