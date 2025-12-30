import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- COLORS PALETTE (Architectural & Engineering) ---
C_BG = 'white'
C_GRID = '#E5E7EB'       # Light Gray Grid
C_AXIS = '#374151'       # Dark Gray Axis
C_BEAM = '#1F2937'       # Dark Beam
C_SHEAR_LINE = '#D97706' # Amber 600
C_SHEAR_FILL = 'rgba(217, 119, 6, 0.15)'
C_MOMENT_LINE = '#2563EB' # Blue 600
C_MOMENT_FILL = 'rgba(37, 99, 235, 0.15)'
C_TEXT = '#111827'

def draw_interactive_diagrams(df, spans, sup_df, loads, unit_force, unit_len):
    total_len = sum(spans)
    cum_spans = [0] + list(np.cumsum(spans))
    
    # Setup Canvas (3 Rows)
    fig = make_subplots(
        rows=3, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.12,
        row_heights=[0.25, 0.375, 0.375],
        subplot_titles=(
            "<b>1. Structural Model (แบบจำลองโครงสร้าง)</b>", 
            f"<b>2. Shear Force Diagram (SFD) [{unit_force}]</b>", 
            f"<b>3. Bending Moment Diagram (BMD) [{unit_force}-{unit_len}]</b>"
        )
    )

    # ==========================================
    # 1. STRUCTURAL MODEL
    # ==========================================
    # Beam Line
    fig.add_trace(go.Scatter(
        x=[0, total_len], y=[0, 0],
        mode='lines', line=dict(color=C_BEAM, width=5),
        hoverinfo='skip'
    ), row=1, col=1)

    # --- Supports Visualization (แยกสัญลักษณ์ตามประเภท) ---
    # เตรียมข้อมูลแยกตาม Type
    for idx, row in sup_df.iterrows():
        x_pos = cum_spans[int(row['id'])]
        sType = row['type']
        
        # Determine Symbol & Style
        symbol = 'circle'
        size = 15
        color = '#4B5563'
        offset_y = 0
        
        if sType == 'Pin':
            symbol = 'triangle-up'
            size = 22
            offset_y = -0.05
        elif sType == 'Roller':
            symbol = 'circle'
            size = 18
            offset_y = -0.05
        elif sType == 'Fixed':
            symbol = 'square'
            size = 20
            # Fixed usually implies wall, let's just mark it clearly
            
        fig.add_trace(go.Scatter(
            x=[x_pos], y=[offset_y],
            mode='markers+text',
            marker=dict(symbol=symbol, size=size, color='white', line=dict(width=2, color=C_BEAM)),
            text=f"<b>{sType}</b>", textposition="bottom center",
            hoverinfo='text', hovertext=f"Node {int(row['id'])+1}: {sType}",
            showlegend=False
        ), row=1, col=1)

    # --- Free Nodes (No Support) ---
    # หา Node ที่ไม่มีใน sup_df
    all_nodes = set(range(len(cum_spans)))
    sup_nodes = set(sup_df['id'].astype(int).tolist())
    free_nodes = all_nodes - sup_nodes
    
    for nid in free_nodes:
        x_pos = cum_spans[nid]
        fig.add_trace(go.Scatter(
            x=[x_pos], y=[0],
            mode='markers',
            marker=dict(symbol='circle', size=8, color=C_BEAM), # จุดเล็กๆ
            hoverinfo='text', hovertext=f"Node {nid+1}: Free",
            showlegend=False
        ), row=1, col=1)

    # --- Loads ---
    for l in loads:
        x_s = cum_spans[int(l['span_idx'])]
        if l['type'] == 'P':
            fig.add_annotation(
                x=x_s + l['x'], y=0, 
                ax=0, ay=-50,
                arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor='#EF4444',
                text=f"<b>P={l['P']}</b>", font=dict(color='#EF4444', size=11),
                bgcolor="white", borderpad=2,
                row=1, col=1
            )
        elif l['type'] == 'U':
            x_e = cum_spans[int(l['span_idx'])+1]
            fig.add_shape(type="rect", x0=x_s, y0=0.1, x1=x_e, y1=0.25, line_width=0, fillcolor='#EF4444', opacity=0.15, row=1, col=1)
            fig.add_annotation(x=(x_s+x_e)/2, y=0.3, showarrow=False, text=f"<b>w={l['w']}</b>", font=dict(color='#EF4444', size=11), row=1, col=1)

    # ==========================================
    # 2. SHEAR FORCE (SFD)
    # ==========================================
    # Zero Line
    fig.add_shape(type="line", x0=0, x1=total_len, y0=0, y1=0, line=dict(color=C_AXIS, width=1.5), row=2, col=1)
    
    # Graph
    fig.add_trace(go.Scatter(
        x=df['x'], y=df['shear'],
        mode='lines', line=dict(color=C_SHEAR_LINE, width=2.5),
        fill='tozeroy', fillcolor=C_SHEAR_FILL,
        name='Shear'
    ), row=2, col=1)

    # Label Max/Min (No Arrow, Use @)
    v_max = df['shear'].abs().max()
    if v_max > 1e-3:
        # Find index
        idx = df['shear'].abs().idxmax()
        row_v = df.loc[idx]
        val = row_v['shear']
        x_loc = row_v['x']
        
        # Position Logic (ถ้าค่าบวก ตัวหนังสืออยู่บน, ค่าลบ ตัวหนังสืออยู่ล่าง)
        yshift = 15 if val >= 0 else -15
        va = "bottom" if val >= 0 else "top"
        
        fig.add_annotation(
            x=x_loc, y=val,
            text=f"<b>{val:.2f} @ {x_loc:.2f}m</b>",
            showarrow=False, # ตัดลูกศรทิ้งตามสั่ง
            yshift=yshift,
            font=dict(color=C_SHEAR_LINE, size=11),
            bgcolor="rgba(255,255,255,0.7)",
            row=2, col=1
        )

    # ==========================================
    # 3. BENDING MOMENT (BMD)
    # ==========================================
    fig.add_shape(type="line", x0=0, x1=total_len, y0=0, y1=0, line=dict(color=C_AXIS, width=1.5), row=3, col=1)

    fig.add_trace(go.Scatter(
        x=df['x'], y=df['moment'],
        mode='lines', line=dict(color=C_MOMENT_LINE, width=2.5),
        fill='tozeroy', fillcolor=C_MOMENT_FILL,
        name='Moment'
    ), row=3, col=1)

    # Label Max/Min (No Arrow, Use @)
    m_max = df['moment'].max()
    m_min = df['moment'].min()
    
    # Max (+)
    if abs(m_max) > 1e-3:
        xm = df.loc[df['moment'] == m_max, 'x'].iloc[0]
        fig.add_annotation(
            x=xm, y=m_max,
            text=f"<b>M(+): {m_max:.2f} @ {xm:.2f}m</b>",
            showarrow=False,
            yshift=15, # Shift Up
            font=dict(color=C_MOMENT_LINE, size=11),
            bgcolor="rgba(255,255,255,0.7)",
            row=3, col=1
        )
    
    # Min (-)
    if abs(m_min) > 1e-3:
        xm = df.loc[df['moment'] == m_min, 'x'].iloc[0]
        fig.add_annotation(
            x=xm, y=m_min,
            text=f"<b>M(-): {m_min:.2f} @ {xm:.2f}m</b>",
            showarrow=False,
            yshift=-15, # Shift Down
            font=dict(color=C_MOMENT_LINE, size=11),
            bgcolor="rgba(255,255,255,0.7)",
            row=3, col=1
        )

    # ==========================================
    # GLOBAL LAYOUT & AXIS
    # ==========================================
    fig.update_layout(
        height=850,
        margin=dict(l=60, r=40, t=50, b=50),
        paper_bgcolor='white',
        plot_bgcolor='white',
        showlegend=False,
        hovermode="x unified",
        font=dict(family="Sarabun, sans-serif", size=13, color=C_TEXT)
    )

    # Axis Styling (กู้คืนแกน XY ให้ชัดเจน)
    axis_style = dict(
        showline=True, linewidth=1.5, linecolor=C_AXIS, # เส้นแกนเข้มขึ้น
        showgrid=True, gridcolor=C_GRID, gridwidth=1,   # Grid บางๆ
        zeroline=False, # เราวาดเส้น 0 เองแล้ว
        showticklabels=True,
        ticks="outside", tickwidth=1.5, tickcolor=C_AXIS
    )
    
    fig.update_xaxes(**axis_style)
    fig.update_yaxes(**axis_style)
    
    # Model Row Adjustments
    fig.update_yaxes(visible=False, range=[-0.8, 0.8], row=1, col=1) # ซ่อนแกน Y เฉพาะ row 1 เพื่อความสวย
    fig.update_xaxes(title_text="Distance (m)", row=3, col=1) # ใส่ Title แกน X อันล่างสุด

    st.plotly_chart(fig, use_container_width=True)

def render_result_tables(df, reac, spans):
    st.markdown("---")
    st.subheader("📊 สรุปผลการวิเคราะห์ (Analysis Summary)")
    
    # Metric Cards with Styling
    c1, c2, c3 = st.columns(3)
    c1.metric("Max Shear (แรงเฉือนสูงสุด)", f"{df['shear'].abs().max():.2f}")
    c2.metric("Max Moment (+) (โมเมนต์บวก)", f"{df['moment'].max():.2f}")
    c3.metric("Max Moment (-) (โมเมนต์ลบ)", f"{df['moment'].min():.2f}")

    # Tabs
    tab1, tab2 = st.tabs(["📍 Support Reactions", "📄 Detailed Data Table"])
    
    with tab1:
        # Prepare Data
        data = []
        n_nodes = len(spans) + 1
        for i in range(n_nodes):
            ry = reac[2*i]
            mz = reac[2*i+1]
            data.append({
                "Node (จุดที่)": i+1,
                "Reaction Ry (แรงแนวดิ่ง)": f"{ry:.2f}",
                "Reaction Mz (โมเมนต์ดัด)": f"{mz:.2f}" if abs(mz) > 0.001 else "-"
            })
        st.dataframe(pd.DataFrame(data), hide_index=True, use_container_width=True)

    with tab2:
        # Exportable Table
        df_export = df.drop_duplicates(subset=['x'], keep='first').copy()
        df_export = df_export[['x', 'shear', 'moment']]
        df_export.columns = ["Position (x)", "Shear Force (V)", "Bending Moment (M)"]
        st.dataframe(df_export.style.format("{:.2f}"), use_container_width=True, height=400)
