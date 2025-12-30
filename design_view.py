import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- COLORS ---
C_SHEAR = '#F59E0B'   # Orange
C_MOMENT = '#2563EB'  # Blue
C_LOAD = '#DC2626'    # Red
C_BEAM = '#374151'    # Dark Grey

def draw_interactive_diagrams(df, spans, sup_df, loads, unit_force, unit_len):
    """
    วาดกราฟ 3 ชั้น (Model, Shear, Moment) โดยโชว์แกนและตัวเลขชัดเจน
    """
    total_len = sum(spans)
    cum_spans = [0] + list(np.cumsum(spans))
    
    # 1. สร้าง Subplots
    fig = make_subplots(
        rows=3, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.1,
        row_heights=[0.25, 0.35, 0.40],
        subplot_titles=(
            "<b>1. Structural Model (แบบจำลองโครงสร้าง)</b>", 
            f"<b>2. Shear Force Diagram (แรงเฉือน - {unit_force})</b>", 
            f"<b>3. Bending Moment Diagram (โมเมนต์ดัด - {unit_force}-{unit_len})</b>"
        )
    )

    # === ROW 1: MODEL ===
    # คาน
    fig.add_trace(go.Scatter(
        x=[0, total_len], y=[0, 0],
        mode='lines', line=dict(color=C_BEAM, width=6),
        hoverinfo='skip', showlegend=False
    ), row=1, col=1)

    # จุดรองรับ
    sup_x = [cum_spans[int(row['id'])] for _, row in sup_df.iterrows()]
    sup_txt = [row['type'] for _, row in sup_df.iterrows()]
    fig.add_trace(go.Scatter(
        x=sup_x, y=[-0.2]*len(sup_x), # ขยับลงนิดหน่อย
        mode='markers+text', 
        marker=dict(symbol='triangle-up', size=18, color=C_BEAM),
        text=sup_txt, textposition="bottom center",
        hoverinfo='text', showlegend=False
    ), row=1, col=1)

    # Loads (ปรับปรุงลูกศร)
    for l in loads:
        x_s = cum_spans[int(l['span_idx'])]
        if l['type'] == 'P':
            # Arrow annotation
            fig.add_annotation(
                x=x_s + l['x'], y=0, 
                ax=0, ay=-50, # ความยาวลูกศรแบบ Pixel (Fixed visual size)
                arrowhead=2, arrowsize=1.5, arrowwidth=2, arrowcolor=C_LOAD,
                text=f"<b>P={l['P']}</b>", 
                font=dict(color=C_LOAD, size=11),
                row=1, col=1
            )
        elif l['type'] == 'U':
            x_e = cum_spans[int(l['span_idx'])+1]
            # UDL Area
            fig.add_shape(
                type="rect", x0=x_s, y0=0.1, x1=x_e, y1=0.4,
                line=dict(width=0), fillcolor=C_LOAD, opacity=0.2,
                row=1, col=1
            )
            # Label
            fig.add_annotation(
                x=(x_s+x_e)/2, y=0.5,
                showarrow=False,
                text=f"<b>w={l['w']}</b>",
                font=dict(color=C_LOAD, size=11),
                row=1, col=1
            )

    # === ROW 2: SHEAR ===
    fig.add_trace(go.Scatter(
        x=df['x'], y=df['shear'],
        mode='lines', line=dict(color=C_SHEAR, width=2),
        fill='tozeroy', fillcolor='rgba(245, 158, 11, 0.1)',
        name='Shear'
    ), row=2, col=1)

    # Annotate Max Shear
    v_max = df['shear'].abs().max()
    if v_max > 0:
        row_v = df.loc[df['shear'].abs() == v_max].iloc[0]
        fig.add_annotation(
            x=row_v['x'], y=row_v['shear'],
            text=f"Vmax: {row_v['shear']:.2f}",
            showarrow=True, arrowhead=1, ax=0, ay=-20 if row_v['shear'] > 0 else 20,
            font=dict(color=C_SHEAR, weight="bold"),
            row=2, col=1
        )

    # === ROW 3: MOMENT ===
    fig.add_trace(go.Scatter(
        x=df['x'], y=df['moment'],
        mode='lines', line=dict(color=C_MOMENT, width=2),
        fill='tozeroy', fillcolor='rgba(37, 99, 235, 0.1)',
        name='Moment'
    ), row=3, col=1)

    # Annotate Max/Min Moment
    m_max = df['moment'].max()
    m_min = df['moment'].min()
    
    # Positive Moment
    if m_max > 1e-3:
        xm = df.loc[df['moment'] == m_max, 'x'].iloc[0]
        fig.add_annotation(
            x=xm, y=m_max, 
            text=f"M(+): {m_max:.2f}",
            showarrow=True, arrowhead=1, ax=0, ay=-20,
            font=dict(color=C_MOMENT, weight="bold"), row=3, col=1
        )
    # Negative Moment
    if m_min < -1e-3:
        xm = df.loc[df['moment'] == m_min, 'x'].iloc[0]
        fig.add_annotation(
            x=xm, y=m_min, 
            text=f"M(-): {m_min:.2f}",
            showarrow=True, arrowhead=1, ax=0, ay=20,
            font=dict(color=C_MOMENT, weight="bold"), row=3, col=1
        )

    # === GLOBAL LAYOUT (AXES & GRIDS) ===
    fig.update_layout(
        height=800, # เพิ่มความสูงเพื่อให้กราฟไม่เบียด
        margin=dict(l=50, r=20, t=50, b=50), # เพิ่มขอบซ้ายให้เลขแกน Y ไม่ตกขอบ
        paper_bgcolor='white',
        plot_bgcolor='#FAFAFA', # พื้นหลังสีเทาจางๆ ให้อ่านง่าย
        hovermode="x unified",
        showlegend=False
    )

    # Config แกน (ให้โชว์เส้น grid และตัวเลขชัดๆ)
    axis_config = dict(
        showline=True, linewidth=1, linecolor='black',
        showgrid=True, gridcolor='#E5E7EB',
        zeroline=True, zerolinecolor='black', zerolinewidth=1,
        showticklabels=True # บังคับโชว์ตัวเลข
    )
    
    # Apply axis config to all subplots
    fig.update_xaxes(**axis_config)
    fig.update_yaxes(**axis_config)
    
    # Hide Y-axis for Model (Row 1)
    fig.update_yaxes(visible=False, row=1, col=1)

    st.plotly_chart(fig, use_container_width=True)


def render_result_tables(df, reac, spans):
    """
    แสดงผลลัพธ์แบบตัวเลขและตารางละเอียด
    """
    # 1. Summary Box
    st.subheader("4. สรุปผลการวิเคราะห์ (Analysis Summary)")
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Max Shear (แรงเฉือนสูงสุด)", f"{df['shear'].abs().max():.2f}")
    with c2:
        st.metric("Max Moment (+) (โมเมนต์บวก)", f"{df['moment'].max():.2f}")
    with c3:
        st.metric("Max Moment (-) (โมเมนต์ลบ)", f"{df['moment'].min():.2f}")

    # 2. Tabs: Reactions & Detailed Data
    tab1, tab2 = st.tabs(["📍 แรงปฏิกิริยา (Reactions)", "📄 ตารางค่าละเอียด (Shear/Moment Table)"])
    
    with tab1:
        r_data = []
        for i in range(len(spans)+1):
            ry = reac[2*i]
            mz = reac[2*i+1]
            txt_mz = f"{mz:.2f}" if abs(mz) > 0.01 else "-"
            r_data.append({"Support Node": f"Node {i+1}", "Ry (Vertical)": f"{ry:.2f}", "Mz (Moment)": txt_mz})
        
        st.dataframe(pd.DataFrame(r_data), use_container_width=True, hide_index=True)
        
    with tab2:
        # ตารางค่าละเอียด (เอาข้อมูลดิบมาโชว์)
        st.caption("ตารางแสดงค่า V และ M ที่ระยะ x ต่างๆ ตลอดความยาวคาน")
        
        # จัด Format ให้สวยงาม
        df_show = df.copy()
        df_show.columns = ["Distance (x)", "Shear (V)", "Moment (M)"]
        st.dataframe(
            df_show.style.format("{:.2f}"), 
            use_container_width=True, 
            height=300 # Scrollable
        )
