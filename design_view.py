import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- COLORS (Clean Palette) ---
C_SHEAR = '#F59E0B'   # Orange
C_MOMENT = '#2563EB'  # Blue
C_LOAD = '#DC2626'    # Red
C_BEAM = '#111827'    # Almost Black
C_SUP = '#4B5563'     # Gray for supports

def draw_interactive_diagrams(df, spans, sup_df, loads, unit_force, unit_len):
    """
    วาดกราฟ 3 ชั้น เน้นความสะอาดตา และความแม่นยำของตำแหน่ง
    """
    total_len = sum(spans)
    cum_spans = [0] + list(np.cumsum(spans))
    
    # 1. Setup Subplots with cleaner titles
    fig = make_subplots(
        rows=3, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.06, # ลดระยะห่างลงนิดหน่อย
        row_heights=[0.2, 0.4, 0.4],
        subplot_titles=(
            "Structural Model", 
            f"Shear Force ({unit_force})", 
            f"Bending Moment ({unit_force}-{unit_len})"
        )
    )

    # === ROW 1: MODEL (Fixing Alignment) ===
    # 1.1 คาน (Beam Line) วางที่ y=0
    fig.add_trace(go.Scatter(
        x=[0, total_len], y=[0, 0],
        mode='lines', line=dict(color=C_BEAM, width=5),
        hoverinfo='skip', showlegend=False
    ), row=1, col=1)

    # 1.2 จุดรองรับ (Supports) วางที่ y=0 เป๊ะๆ
    sup_x = [cum_spans[int(row['id'])] for _, row in sup_df.iterrows()]
    sup_txt = [row['type'] for _, row in sup_df.iterrows()]
    fig.add_trace(go.Scatter(
        x=sup_x, 
        y=[0]*len(sup_x), # y=0 เพื่อให้ยอดสามเหลี่ยมแตะคานพอดี
        mode='markers', 
        marker=dict(symbol='triangle-up', size=18, color=C_SUP, line=dict(width=1, color=C_BEAM)),
        text=sup_txt, hoverinfo='text', showlegend=False
    ), row=1, col=1)

    # 1.3 Loads (ปรับลูกศรให้ดู Clean ขึ้น)
    for l in loads:
        x_s = cum_spans[int(l['span_idx'])]
        if l['type'] == 'P':
            fig.add_annotation(
                x=x_s + l['x'], y=0, 
                ax=0, ay=-60, # ความยาวลูกศรคงที่
                arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor=C_LOAD,
                text=f"<b>P={l['P']}</b>", 
                font=dict(color=C_LOAD, size=12),
                bgcolor="white", bordercolor=C_LOAD, borderpad=2, # ใส่กล่องข้อความให้ชัด
                row=1, col=1
            )
        elif l['type'] == 'U':
            x_e = cum_spans[int(l['span_idx'])+1]
            # UDL Area (ใช้สีจางๆ)
            fig.add_shape(
                type="rect", x0=x_s, y0=0.08, x1=x_e, y1=0.3,
                line=dict(width=0), fillcolor=C_LOAD, opacity=0.15,
                row=1, col=1
            )
            # Label
            fig.add_annotation(
                x=(x_s+x_e)/2, y=0.35, showarrow=False,
                text=f"<b>w={l['w']}</b>", font=dict(color=C_LOAD, size=12),
                 bgcolor="white", bordercolor=C_LOAD, borderpad=2,
                row=1, col=1
            )

    # === ROW 2: SHEAR ===
    fig.add_trace(go.Scatter(
        x=df['x'], y=df['shear'],
        mode='lines', line=dict(color=C_SHEAR, width=2),
        fill='tozeroy', fillcolor='rgba(245, 158, 11, 0.1)', # สีส้มจางๆ
        name='Shear'
    ), row=2, col=1)

    # === ROW 3: MOMENT ===
    fig.add_trace(go.Scatter(
        x=df['x'], y=df['moment'],
        mode='lines', line=dict(color=C_MOMENT, width=2),
        fill='tozeroy', fillcolor='rgba(37, 99, 235, 0.1)', # สีน้ำเงินจางๆ
        name='Moment'
    ), row=3, col=1)

    # === GLOBAL LAYOUT (CLEAN STYLE) ===
    fig.update_layout(
        height=750,
        margin=dict(l=60, r=20, t=60, b=40),
        paper_bgcolor='white', # พื้นหลังขาวจั๊วะ
        plot_bgcolor='white',  # พื้นที่กราฟขาวจั๊วะ
        hovermode="x unified",
        showlegend=False,
        font=dict(family="Sarabun, sans-serif", size=14) # ใช้ฟอนต์อ่านง่าย
    )

    # Config แกนให้ดูสะอาดตา (เส้นบางสีเทาอ่อน)
    axis_config = dict(
        showline=True, linewidth=1, linecolor='#E5E7EB', # เส้นแกนสีเทาอ่อน
        showgrid=True, gridcolor='#F3F4F6', gridwidth=1, # เส้นกริดสีจางๆ
        zeroline=True, zerolinecolor='#9CA3AF', zerolinewidth=1, # เส้นศูนย์สีเทาเข้มขึ้นนิดนึง
        showticklabels=True
    )
    
    fig.update_xaxes(**axis_config)
    fig.update_yaxes(**axis_config)
    
    # *** CRITICAL FIX: ล็อกแกน Y ของ Model ให้คานและ Support อยู่ตรงกลาง ***
    fig.update_yaxes(range=[-0.5, 1.5], visible=False, row=1, col=1) 

    st.plotly_chart(fig, use_container_width=True)


def render_result_tables(df, reac, spans):
    # (ส่วนนี้เหมือนเดิมครับ โค้ดเดิมดีอยู่แล้ว)
    st.subheader("4. สรุปผลการวิเคราะห์ (Analysis Summary)")
    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Max Shear", f"{df['shear'].abs().max():.2f}")
    with c2: st.metric("Max Moment (+)", f"{df['moment'].max():.2f}")
    with c3: st.metric("Max Moment (-)", f"{df['moment'].min():.2f}")

    tab1, tab2 = st.tabs(["📍 Support Reactions", "📄 Detailed Table"])
    
    with tab1:
        r_data = []
        for i in range(len(spans)+1):
            mz_val = reac[2*i+1]
            mz_txt = f"{mz_val:.2f}" if abs(mz_val) > 0.001 else "-"
            r_data.append({"Node": f"{i+1}", "Ry": f"{reac[2*i]:.2f}", "Mz": mz_txt})
        st.dataframe(pd.DataFrame(r_data), use_container_width=True, hide_index=True)
        
    with tab2:
        df_show = df.copy()
        df_show.columns = ["x", "V (Shear)", "M (Moment)"]
        st.dataframe(df_show.style.format("{:.2f}"), use_container_width=True, height=300)
