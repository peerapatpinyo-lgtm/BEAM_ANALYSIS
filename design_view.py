import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

def draw_interactive_diagrams(df, reac, spans, sup_df, raw_loads):
    """
    High-End Structural Visualization
    """
    nodes = [0] + list(np.cumsum(spans))
    total_len = nodes[-1]
    
    # Setup Subplots (Layout แบบ Engineering Report)
    fig = make_subplots(
        rows=4, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.08,
        row_heights=[0.3, 0.23, 0.23, 0.24],
        subplot_titles=(
            "🏗️ Loading Diagram (FBD)", 
            "⚡ Shear Force Diagram (SFD)", 
            "🔄 Bending Moment Diagram (BMD)", 
            "📉 Deflection Diagram (δ)"
        )
    )

    # ==========================================
    # 1. VISUALIZATION (Real Beam Model)
    # ==========================================
    # กำหนดความหนาคานให้ดูสมส่วน (Dynamic Height)
    beam_h = total_len * 0.06 
    beam_h = max(0.3, min(beam_h, 0.8)) # Clamp ไม่ให้เล็กหรือใหญ่เกินไป
    beam_top = beam_h / 2
    beam_bot = -beam_h / 2
    
    # 1.1 วาดตัวคาน (Solid Beam)
    fig.add_shape(type="rect",
        x0=0, x1=total_len, y0=beam_bot, y1=beam_top,
        fillcolor="#eeeeee", line=dict(color="#333333", width=2),
        layer="below", row=1, col=1
    )
    # เส้น Centerline
    fig.add_trace(go.Scatter(
        x=[0, total_len], y=[0, 0], mode="lines",
        line=dict(color="#999999", width=1, dash="dashdot"),
        hoverinfo="skip", showlegend=False
    ), row=1, col=1)

    # 1.2 วาด Supports (จุดรองรับ)
    if not sup_df.empty:
        for _, s in sup_df.iterrows():
            x = nodes[int(s['id'])]
            stype = s['type']
            
            # ขนาด Support แปรผันตามขนาดคาน
            sup_size = beam_h * 0.8
            
            if stype == "Pin":
                # สามเหลี่ยมทึบ
                fig.add_trace(go.Scatter(
                    x=[x], y=[beam_bot], mode="markers",
                    marker=dict(symbol="triangle-up", size=15, color="#333333"),
                    hoverinfo="text", text="Pin Support", showlegend=False
                ), row=1, col=1)
                # ฐาน (Ground)
                fig.add_shape(type="line", x0=x-0.2, x1=x+0.2, y0=beam_bot-0.1, y1=beam_bot-0.1,
                              line=dict(color="#333333", width=2), row=1, col=1)
                
            elif stype == "Roller":
                # วงกลม (ล้อ) + สามเหลี่ยม
                fig.add_trace(go.Scatter(
                    x=[x], y=[beam_bot - sup_size/4], mode="markers",
                    marker=dict(symbol="circle", size=12, color="#333333"),
                    showlegend=False
                ), row=1, col=1)
                # ฐาน
                fig.add_shape(type="line", x0=x-0.2, x1=x+0.2, y0=beam_bot-sup_size/2, y1=beam_bot-sup_size/2,
                              line=dict(color="#333333", width=2), row=1, col=1)

            elif stype == "Fixed":
                # กำแพงแนวตั้ง
                fig.add_shape(type="line", 
                    x0=x, x1=x, y0=beam_bot-0.2, y1=beam_top+0.2,
                    line=dict(color="#333333", width=4), row=1, col=1
                )
                # ขีดแรเงา (Hatching)
                hatch_dir = 1 if x == 0 else -1 # หันลายเส้นออกนอกคาน
                for h in np.linspace(beam_bot-0.2, beam_top+0.2, 6):
                    fig.add_shape(type="line", 
                        x0=x, y0=h, x1=x - (0.3 * hatch_dir), y1=h - 0.1,
                        line=dict(color="#333333", width=1), row=1, col=1
                    )

    # 1.3 วาด Loads (จุดสำคัญที่ต้องแก้!)
    max_load_val = 100.0
    if raw_loads:
        max_load_val = max([l['mag'] for l in raw_loads])
    
    # กำหนดความสูงของกราฟฟิกแรง (Load Height Scaling)
    load_scale_h = beam_h * 1.5 
    
    for l in raw_loads:
        color = "#D32F2F" if l['case'] == 'LL' else "#1976D2" # แดง/น้ำเงิน
        
        if l['type'] == 'P':
            # === Point Load ===
            x_loc = nodes[int(l['span_idx'])] + l['x']
            
            # วาดลูกศร P (ใช้ Annotation จะคมชัดกว่า Shape)
            fig.add_annotation(
                x=x_loc, y=beam_top, # ปลายลูกศรแตะหลังคาน
                ax=0, ay=-60,       # หางลูกศรชี้ขึ้นไป 60px
                xref="x1", yref="y1",
                text=f"<b>P = {l['mag']:,.0f}</b>",
                showarrow=True, arrowhead=2, arrowwidth=2, arrowcolor=color,
                font=dict(color=color, size=12),
                bgcolor="rgba(255,255,255,0.8)", bordercolor=color, borderwidth=1,
                row=1, col=1
            )
            
        elif l['type'] == 'U':
            # === Uniform Load (The Fix) ===
            start_x = nodes[int(l['span_idx'])] + l['x']
            span_len = spans[int(l['span_idx'])]
            end_x = nodes[int(l['span_idx'])] + l.get('end', span_len)
            
            # คำนวณความสูง Block ของแรงตามขนาดแรง
            # ให้แรงน้อยเตี้ย แรงเยอะสูง (แต่มี Min/Max)
            ratio = l['mag'] / max_load_val
            this_load_h = load_scale_h * (0.5 + 0.5*ratio) 
            
            # 1. วาดพื้นที่ระบายสี (Filled Block)
            fig.add_trace(go.Scatter(
                x=[start_x, end_x, end_x, start_x], 
                y=[beam_top, beam_top, beam_top + this_load_h, beam_top + this_load_h],
                fill='toself', fillcolor=color, opacity=0.15,
                line=dict(width=0), showlegend=False, hoverinfo='skip'
            ), row=1, col=1)
            
            # 2. วาดเส้นปิดด้านบน (The "Bar")
            fig.add_trace(go.Scatter(
                x=[start_x, end_x], y=[beam_top + this_load_h, beam_top + this_load_h],
                mode='lines', line=dict(color=color, width=2),
                showlegend=False, hoverinfo='skip'
            ), row=1, col=1)
            
            # 3. วาดลูกศรเรียงกัน (The "Comb")
            # คำนวณจำนวนลูกศรให้เหมาะสมกับความยาว
            dist = end_x - start_x
            n_arrows = max(3, int(dist * 2.5)) # อย่างน้อย 3 ตัว หรือ 2.5 ตัวต่อเมตร
            
            for ax in np.linspace(start_x, end_x, n_arrows):
                fig.add_annotation(
                    x=ax, y=beam_top,         # หัวแตะคาน
                    ax=0, ay=-this_load_h*30, # หางอยู่ที่เส้น Bar ด้านบน (scale pixels approx)
                    # หมายเหตุ: Plotly ay เป็น pixel, เราต้องกะให้มันดูเหมือนมาจากเส้นบน
                    # วิธีที่ดีกว่าคือใช้ ay= -(ระยะ pixel) แต่เนื่องจาก y เป็น unit coordinate 
                    # เราใช้ shape เส้นตรงแนวตั้งแทนจะแม่นยำกว่า
                    showarrow=True, arrowhead=3, arrowwidth=1.5, arrowcolor=color, arrowsize=1,
                    visible=False # Trick: ซ่อน Annotation ปกติ แล้วใช้วาดเส้นแทนข้างล่าง
                )
                # วาดเส้นก้านลูกศร
                fig.add_shape(type="line",
                    x0=ax, x1=ax, 
                    y0=beam_top, y1=beam_top + this_load_h,
                    line=dict(color=color, width=1.5), layer="below",
                    row=1, col=1
                )
                # วาดหัวลูกศร (Marker)
                fig.add_trace(go.Scatter(
                    x=[ax], y=[beam_top],
                    mode='markers', marker=dict(symbol="triangle-down", size=8, color=color),
                    showlegend=False, hoverinfo='skip'
                ), row=1, col=1)

            # Label ตรงกลาง Block
            fig.add_annotation(
                x=(start_x+end_x)/2, y=beam_top + this_load_h,
                text=f"<b>w = {l['mag']:,.0f}</b>",
                yshift=15, showarrow=False, 
                font=dict(color=color, size=12), bgcolor="rgba(255,255,255,0.7)",
                row=1, col=1
            )

    # ==========================================
    # 2. SHEAR FORCE (Step Line)
    # ==========================================
    # Fill Area สีส้มไล่เฉด
    fig.add_trace(go.Scatter(
        x=df['x'], y=df['shear'], mode='lines', line_shape='hv',
        fill='tozeroy', line=dict(color='#FF9800', width=2),
        fillcolor='rgba(255, 152, 0, 0.2)', name="Shear"
    ), row=2, col=1)
    
    # Annotate Max/Min
    v_max, v_min = df['shear'].max(), df['shear'].min()
    fig.add_annotation(x=df.loc[df['shear'].idxmax(), 'x'], y=v_max, text=f"{v_max:.0f}", showarrow=False, yshift=10, row=2, col=1)
    fig.add_annotation(x=df.loc[df['shear'].idxmin(), 'x'], y=v_min, text=f"{v_min:.0f}", showarrow=False, yshift=-10, row=2, col=1)

    # ==========================================
    # 3. BENDING MOMENT (Curve)
    # ==========================================
    # แยกสี บวก/ลบ
    df['m_pos'] = df['moment'].apply(lambda x: x if x >= 0 else 0)
    df['m_neg'] = df['moment'].apply(lambda x: x if x < 0 else 0)

    fig.add_trace(go.Scatter(x=df['x'], y=df['m_pos'], mode='lines', fill='tozeroy', line=dict(width=0), fillcolor='rgba(33, 150, 243, 0.3)', name="+Moment"), row=3, col=1)
    fig.add_trace(go.Scatter(x=df['x'], y=df['m_neg'], mode='lines', fill='tozeroy', line=dict(width=0), fillcolor='rgba(233, 30, 99, 0.3)', name="-Moment"), row=3, col=1)
    fig.add_trace(go.Scatter(x=df['x'], y=df['moment'], mode='lines', line=dict(color='#1565C0', width=2), showlegend=False), row=3, col=1)

    # Annotate Peaks
    m_max, m_min = df['moment'].max(), df['moment'].min()
    if abs(m_max) > 1: fig.add_annotation(x=df.loc[df['moment'].idxmax(), 'x'], y=m_max, text=f"{m_max:.0f}", showarrow=False, yshift=10, font=dict(color="blue"), row=3, col=1)
    if abs(m_min) > 1: fig.add_annotation(x=df.loc[df['moment'].idxmin(), 'x'], y=m_min, text=f"{m_min:.0f}", showarrow=False, yshift=-10, font=dict(color="red"), row=3, col=1)

    # ==========================================
    # 4. DEFLECTION
    # ==========================================
    defl_mm = df['deflection'] * 1000
    fig.add_trace(go.Scatter(
        x=df['x'], y=defl_mm, mode='lines',
        line=dict(color='#4CAF50', width=2, dash='dot'), name="Defl"
    ), row=4, col=1)
    
    d_max = defl_mm.abs().max()
    d_idx = defl_mm.abs().idxmax()
    fig.add_annotation(
        x=df.loc[d_idx, 'x'], y=defl_mm[d_idx], 
        text=f"δ max: {defl_mm[d_idx]:.2f} mm", 
        showarrow=True, arrowhead=1, row=4, col=1
    )

    # ==========================================
    # LAYOUT STYLING
    # ==========================================
    fig.update_layout(
        height=1200, 
        template="plotly_white", 
        showlegend=False, 
        hovermode="x unified",
        margin=dict(l=40, r=40, t=60, b=40),
        font=dict(family="Roboto, sans-serif")
    )
    
    # Hide Y-axis labels for Load Diagram to clean up
    fig.update_yaxes(visible=False, showticklabels=False, row=1, col=1)
    # Range padding
    fig.update_yaxes(title="V (kg)", zeroline=True, zerolinewidth=1, zerolinecolor='black', row=2, col=1)
    fig.update_yaxes(title="M (kg-m)", zeroline=True, zerolinewidth=1, zerolinecolor='black', row=3, col=1)
    fig.update_yaxes(title="δ (mm)", zeroline=True, zerolinewidth=1, zerolinecolor='black', row=4, col=1)
    fig.update_xaxes(title="Distance (m)", row=4, col=1)

    st.plotly_chart(fig, use_container_width=True)

def render_result_tables(df, reac, spans, u_force, u_len):
    # (ใช้โค้ดเดิมส่วนนี้ได้เลย แต่ปรับ Styling เล็กน้อย)
    st.markdown("#### 📊 Engineering Summary")
    c1, c2 = st.columns([1, 2])
    
    with c1:
        st.info("📍 Reactions")
        if reac is not None:
            r_data = []
            for i in range(len(reac)//2):
                fy = reac[2*i]
                mz = reac[2*i+1]
                if abs(fy)>0.1 or abs(mz)>0.1:
                    r_data.append({"Node": i, "Ry": f"{fy:.2f}", "Mz": f"{mz:.2f}"})
            st.table(pd.DataFrame(r_data))
            
    with c2:
        st.success("📐 Max Forces per Span")
        s_data = []
        nodes = [0] + list(np.cumsum(spans))
        for i in range(len(spans)):
            sub = df[(df['x'] >= nodes[i]) & (df['x'] <= nodes[i+1])]
            s_data.append({
                "Span": i+1,
                "V max": f"{sub['shear'].abs().max():.2f}",
                "M max (+)": f"{sub['moment'].max():.2f}",
                "M max (-)": f"{sub['moment'].min():.2f}",
                "Defl (mm)": f"{sub['deflection'].abs().max()*1000:.2f}"
            })
        st.dataframe(pd.DataFrame(s_data), hide_index=True, use_container_width=True)
