import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

def draw_interactive_diagrams(df, reac, spans, sup_df, raw_loads):
    """
    Ultimate Professional Visualization
    - Independent Scaling for P and U loads
    - Correct Tension Side Moment Diagram
    - High-Contrast Engineering Style
    """
    nodes = [0] + list(np.cumsum(spans))
    total_len = nodes[-1]
    
    # --- 1. CONFIGURATION & SCALING ---
    # Setup Geometry
    beam_h = total_len * 0.08  
    beam_h = max(0.4, min(beam_h, 0.7)) # Clamp height
    beam_top = beam_h / 2
    beam_bot = -beam_h / 2
    
    # --- Independent Scaling Logic ---
    # แยกหาค่า Max ของแต่ละประเภท เพื่อให้กราฟฟิกสวยแยกกัน ไม่กดกันเอง
    p_mags = [l['mag'] for l in raw_loads if l['type'] == 'P']
    u_mags = [l['mag'] for l in raw_loads if l['type'] == 'U']
    
    max_p = max(p_mags) if p_mags else 1.0
    max_u = max(u_mags) if u_mags else 1.0
    
    # ความสูง Max ของกราฟฟิก (Visual Height Limit)
    visual_h_limit = beam_h * 1.8 

    # --- 2. PLOT LAYOUT ---
    fig = make_subplots(
        rows=4, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.08,
        row_heights=[0.3, 0.22, 0.26, 0.22],
        subplot_titles=(
            "📌 Free Body Diagram (FBD)", 
            "⚡ Shear Force Diagram (SFD)", 
            "🔄 Bending Moment (Tension Side)", 
            "📉 Deflection (δ)"
        )
    )

    # ==========================================
    # ROW 1: REALISTIC BEAM & LOADS
    # ==========================================
    
    # 1.1 วาดคาน (Solid Concrete Look)
    fig.add_shape(type="rect",
        x0=0, x1=total_len, y0=beam_bot, y1=beam_top,
        fillcolor="#EEEEEE", line=dict(color="#212121", width=2),
        layer="below", row=1, col=1
    )
    # Centerline (Reference)
    fig.add_trace(go.Scatter(
        x=[0, total_len], y=[0, 0], mode="lines",
        line=dict(color="#BDBDBD", width=1, dash="dashdot"),
        hoverinfo="skip", showlegend=False
    ), row=1, col=1)

    # 1.2 วาด Supports
    if not sup_df.empty:
        for _, s in sup_df.iterrows():
            x = nodes[int(s['id'])]
            stype = s['type']
            
            if stype == "Pin":
                fig.add_trace(go.Scatter(
                    x=[x], y=[beam_bot], mode="markers",
                    marker=dict(symbol="triangle-up", size=14, color="#37474F"),
                    hoverinfo="text", text="Pin", showlegend=False
                ), row=1, col=1)
                fig.add_shape(type="line", x0=x-0.2, x1=x+0.2, y0=beam_bot-0.05, y1=beam_bot-0.05,
                              line=dict(color="#37474F", width=2), row=1, col=1)
                
            elif stype == "Roller":
                fig.add_trace(go.Scatter(
                    x=[x], y=[beam_bot - beam_h*0.15], mode="markers",
                    marker=dict(symbol="circle", size=10, color="#37474F", line=dict(width=1, color="white")),
                    showlegend=False
                ), row=1, col=1)
                fig.add_shape(type="line", x0=x-0.2, x1=x+0.2, y0=beam_bot-beam_h*0.3, y1=beam_bot-beam_h*0.3,
                              line=dict(color="#37474F", width=2), row=1, col=1)

            elif stype == "Fixed":
                fig.add_shape(type="line", x0=x, x1=x, y0=beam_bot-0.3, y1=beam_top+0.3,
                    line=dict(color="#37474F", width=4), row=1, col=1)
                hatch_dir = 1 if x == 0 else -1
                for h in np.linspace(beam_bot-0.3, beam_top+0.3, 6):
                    fig.add_shape(type="line", 
                        x0=x, y0=h, x1=x - (0.25 * hatch_dir), y1=h - 0.1,
                        line=dict(color="#37474F", width=1), row=1, col=1)

    # 1.3 วาด Loads (Independent Scaling)
    for l in raw_loads:
        color = "#D32F2F" if l['case'] == 'LL' else "#1976D2"
        fill_color = "rgba(211, 47, 47, 0.1)" if l['case'] == 'LL' else "rgba(25, 118, 210, 0.1)"
        
        if l['type'] == 'P':
            # === Point Load ===
            # Scale เทียบกับ Point Load ด้วยกันเอง
            ratio = (l['mag'] / max_p) if max_p > 0 else 1.0
            # ความสูงขั้นต่ำ 50% ของ Limit เพื่อให้เห็นชัดเสมอ
            this_h = visual_h_limit * (0.5 + 0.5 * ratio)
            
            x_loc = nodes[int(l['span_idx'])] + l['x']
            
            fig.add_annotation(
                x=x_loc, y=beam_top,
                ax=0, ay=-this_h*40, # Pixel scale rough approximation
                xref="x1", yref="y1",
                text=f"<b>P={l['mag']:,.0f}</b>",
                showarrow=True, arrowhead=2, arrowwidth=2, arrowcolor=color, arrowsize=1.2,
                font=dict(color=color, size=11, weight="bold"),
                bgcolor="white", bordercolor=color, borderwidth=1, borderpad=2,
                row=1, col=1
            )
            
        elif l['type'] == 'U':
            # === Uniform Load ===
            # Scale เทียบกับ Uniform Load ด้วยกันเอง (ทำให้ 1000 ดูใหญ่ได้ ถ้ามันคือ Max ของ U)
            ratio = (l['mag'] / max_u) if max_u > 0 else 1.0
            this_h = visual_h_limit * (0.6 + 0.4 * ratio) # ขั้นต่ำ 60% ของความสูง Max
            
            start_x = nodes[int(l['span_idx'])] + l['x']
            end_x = nodes[int(l['span_idx'])] + l.get('end', spans[int(l['span_idx'])])
            
            # 1. Block Area (โปร่งแสง)
            fig.add_trace(go.Scatter(
                x=[start_x, end_x, end_x, start_x],
                y=[beam_top, beam_top, beam_top + this_h, beam_top + this_h],
                fill='toself', fillcolor=fill_color,
                mode='none', hoverinfo='skip', showlegend=False
            ), row=1, col=1)
            
            # 2. Top Bar (เส้นทึบด้านบน)
            fig.add_trace(go.Scatter(
                x=[start_x, end_x], y=[beam_top + this_h, beam_top + this_h],
                mode='lines', line=dict(color=color, width=2),
                hoverinfo='skip', showlegend=False
            ), row=1, col=1)
            
            # 3. Comb Arrows (ลูกศรเรียง)
            n_arrows = max(3, int((end_x - start_x) * 2))
            for ax in np.linspace(start_x, end_x, n_arrows):
                # ก้านลูกศร
                fig.add_shape(type="line",
                    x0=ax, x1=ax, y0=beam_top, y1=beam_top + this_h,
                    line=dict(color=color, width=1), layer="below", row=1, col=1
                )
                # หัวลูกศร
                fig.add_trace(go.Scatter(
                    x=[ax], y=[beam_top],
                    mode='markers', marker=dict(symbol="triangle-down", size=6, color=color),
                    hoverinfo='skip', showlegend=False
                ), row=1, col=1)
                
            # Label
            fig.add_annotation(
                x=(start_x + end_x)/2, y=beam_top + this_h,
                text=f"<b>w={l['mag']:,.0f}</b>",
                yshift=10, showarrow=False,
                font=dict(color=color, size=11, weight="bold"), bgcolor="rgba(255,255,255,0.8)",
                row=1, col=1
            )

    # ==========================================
    # ROW 2: SHEAR (SFD)
    # ==========================================
    # Gradient Fill Style
    fig.add_trace(go.Scatter(
        x=df['x'], y=df['shear'], mode='lines', line_shape='hv',
        fill='tozeroy', line=dict(color='#FFA000', width=2),
        fillcolor='rgba(255, 160, 0, 0.2)', name="Shear"
    ), row=2, col=1)
    
    # Annotate Max/Min
    v_max, v_min = df['shear'].max(), df['shear'].min()
    # Handle overlap text
    fig.add_annotation(x=df.loc[df['shear'].idxmax(), 'x'], y=v_max, text=f"<b>{v_max:.0f}</b>", showarrow=False, yshift=10, font=dict(color="#EF6C00"), row=2, col=1)
    fig.add_annotation(x=df.loc[df['shear'].idxmin(), 'x'], y=v_min, text=f"<b>{v_min:.0f}</b>", showarrow=False, yshift=-10, font=dict(color="#EF6C00"), row=2, col=1)

    # ==========================================
    # ROW 3: MOMENT (BMD) - TENSION SIDE
    # ==========================================
    # Tension Side = Moment Diagram Plot Direction
    # Thai Standard:
    # Negative Moment (Top Tension) -> Plot UP
    # Positive Moment (Bottom Tension) -> Plot DOWN
    
    df['m_pos'] = df['moment'].apply(lambda x: x if x >= 0 else 0)
    df['m_neg'] = df['moment'].apply(lambda x: x if x < 0 else 0)

    # Plot Positive (Down visually due to reversed axis)
    fig.add_trace(go.Scatter(
        x=df['x'], y=df['m_pos'], mode='lines', line=dict(width=0),
        fill='tozeroy', fillcolor='rgba(33, 150, 243, 0.3)', name="Sagging"
    ), row=3, col=1)
    
    # Plot Negative (Up visually due to reversed axis)
    fig.add_trace(go.Scatter(
        x=df['x'], y=df['m_neg'], mode='lines', line=dict(width=0),
        fill='tozeroy', fillcolor='rgba(233, 30, 99, 0.3)', name="Hogging"
    ), row=3, col=1)

    # Main Curve
    fig.add_trace(go.Scatter(
        x=df['x'], y=df['moment'], mode='lines', 
        line=dict(color='#37474F', width=2), showlegend=False
    ), row=3, col=1)

    # Labels
    m_max = df['moment'].max()
    m_min = df['moment'].min()
    
    # Label for Bottom Tension (Pos M)
    if m_max > 1:
        fig.add_annotation(
            x=df.loc[df['moment'].idxmax(), 'x'], y=m_max, 
            text=f"<b>M(+): {m_max:,.0f}</b>", 
            showarrow=True, arrowcolor="#1976D2", yshift=15, 
            font=dict(color="#1976D2"), row=3, col=1
        )
        
    # Label for Top Tension (Neg M)
    if abs(m_min) > 1:
        fig.add_annotation(
            x=df.loc[df['moment'].idxmin(), 'x'], y=m_min, 
            text=f"<b>M(-): {m_min:,.0f}</b>", 
            showarrow=True, arrowcolor="#C2185B", yshift=-15,
            font=dict(color="#C2185B"), row=3, col=1
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
    d_val = defl_mm[d_idx]
    fig.add_annotation(
        x=df.loc[d_idx, 'x'], y=d_val,
        text=f"δ max: {d_val:.2f} mm",
        showarrow=True, arrowhead=1, arrowcolor="#2E7D32", row=4, col=1
    )

    # ==========================================
    # GLOBAL LAYOUT
    # ==========================================
    fig.update_layout(
        height=1300,
        template="plotly_white",
        hovermode="x unified",
        margin=dict(l=50, r=50, t=60, b=50),
        font=dict(family="Roboto, sans-serif", size=12),
        showlegend=False
    )
    
    # Style Axes
    axis_style = dict(showgrid=True, gridcolor='#F0F0F0', showline=True, linewidth=1, linecolor='#333', mirror=True)
    
    fig.update_yaxes(visible=False, showticklabels=False, row=1, col=1)
    fig.update_yaxes(title="<b>Shear (V)</b>", **axis_style, row=2, col=1)
    
    # REVERSED AXIS for Moment (Tension Side)
    # Positive values go DOWN, Negative values go UP
    fig.update_yaxes(title="<b>Moment (M)</b>", autorange="reversed", **axis_style, row=3, col=1)
    
    fig.update_yaxes(title="<b>Deflection (mm)</b>", **axis_style, row=4, col=1)
    fig.update_xaxes(title="Length (m)", **axis_style, row=4, col=1)
    
    # Zero Lines
    for r in [2, 3, 4]:
        fig.add_hline(y=0, line_width=1, line_color="black", opacity=0.5, row=r, col=1)

    st.plotly_chart(fig, use_container_width=True)

def render_result_tables(df, reac, spans, u_force, u_len):
    st.markdown("---")
    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown("### 📍 Reactions")
        if reac is not None:
            r_data = []
            for i in range(len(reac)//2):
                fy = reac[2*i]
                mz = reac[2*i+1]
                if abs(fy)>0.1 or abs(mz)>0.1:
                    r_data.append({"Node": i, "Ry": f"{fy:,.2f}", "M": f"{mz:,.2f}"})
            st.table(pd.DataFrame(r_data))
            
    with c2:
        st.markdown("### 📐 Max Forces / Span")
        s_data = []
        nodes = [0] + list(np.cumsum(spans))
        for i in range(len(spans)):
            sub = df[(df['x'] >= nodes[i]) & (df['x'] <= nodes[i+1])]
            s_data.append({
                "Span": i+1,
                "V max": f"{sub['shear'].abs().max():,.0f}",
                "+M (Bot)": f"{sub['moment'].max():,.0f}",
                "-M (Top)": f"{sub['moment'].min():,.0f}",
                "δ max": f"{sub['deflection'].abs().max()*1000:.2f}"
            })
        st.dataframe(pd.DataFrame(s_data), hide_index=True, use_container_width=True)
