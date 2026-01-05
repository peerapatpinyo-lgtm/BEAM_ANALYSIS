import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

def draw_interactive_diagrams(df, reac, spans, sup_df, raw_loads):
    """
    Ultimate Structural Visualization:
    - Adaptive Load Scaling
    - Moment on Tension Side (Inverted Y-Axis)
    - Professional Engineering Styling
    """
    nodes = [0] + list(np.cumsum(spans))
    total_len = nodes[-1]
    
    # --- 1. CONFIGURATION & SCALING ---
    # คำนวณ Scale คาน
    beam_h = total_len * 0.08  # ความลึกคาน 8% ของความยาวรวม (ดูสมส่วน)
    beam_h = max(0.4, min(beam_h, 0.8)) # Clamp ไม่ให้เล็ก/ใหญ่เกิน
    beam_top = beam_h / 2
    beam_bot = -beam_h / 2
    
    # คำนวณ Scale ของ Load (ไม่ให้สูงทะลุจอ)
    max_load_val = 1.0
    if raw_loads:
        max_load_val = max([l['mag'] for l in raw_loads])
    
    # กำหนดความสูง Max ของ Load Graphic ให้ไม่เกิน 2 เท่าของความลึกคาน
    max_load_h = beam_h * 2.0 

    # --- 2. PLOT LAYOUT ---
    fig = make_subplots(
        rows=4, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.06,
        row_heights=[0.30, 0.22, 0.26, 0.22], # ให้พื้นที่ Load เยอะหน่อย
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
    
    # 1.1 วาดคาน (Concrete Style)
    fig.add_shape(type="rect",
        x0=0, x1=total_len, y0=beam_bot, y1=beam_top,
        fillcolor="#E0E0E0", line=dict(color="#424242", width=2.5),
        layer="below", row=1, col=1
    )
    # Centerline
    fig.add_trace(go.Scatter(
        x=[0, total_len], y=[0, 0], mode="lines",
        line=dict(color="#9E9E9E", width=1, dash="dashdot"),
        hoverinfo="skip", showlegend=False
    ), row=1, col=1)

    # 1.2 วาด Supports
    if not sup_df.empty:
        for _, s in sup_df.iterrows():
            x = nodes[int(s['id'])]
            stype = s['type']
            sup_sz = beam_h * 0.6
            
            if stype == "Pin":
                fig.add_trace(go.Scatter(
                    x=[x], y=[beam_bot], mode="markers",
                    marker=dict(symbol="triangle-up", size=14, color="#212121"),
                    hoverinfo="text", text="Pin", showlegend=False
                ), row=1, col=1)
                # Base Line
                fig.add_shape(type="line", x0=x-0.2, x1=x+0.2, y0=beam_bot-0.1, y1=beam_bot-0.1,
                              line=dict(color="#212121", width=2), row=1, col=1)
                
            elif stype == "Roller":
                fig.add_trace(go.Scatter(
                    x=[x], y=[beam_bot - sup_sz/3], mode="markers",
                    marker=dict(symbol="circle", size=12, color="#212121", line=dict(width=1, color="white")),
                    showlegend=False
                ), row=1, col=1)
                fig.add_shape(type="line", x0=x-0.2, x1=x+0.2, y0=beam_bot-sup_sz/1.5, y1=beam_bot-sup_sz/1.5,
                              line=dict(color="#212121", width=2), row=1, col=1)

            elif stype == "Fixed":
                fig.add_shape(type="line", x0=x, x1=x, y0=beam_bot-0.3, y1=beam_top+0.3,
                    line=dict(color="#212121", width=4), row=1, col=1)
                # Hatching
                hatch_dir = 1 if x == 0 else -1
                for h in np.linspace(beam_bot-0.3, beam_top+0.3, 7):
                    fig.add_shape(type="line", 
                        x0=x, y0=h, x1=x - (0.25 * hatch_dir), y1=h - 0.1,
                        line=dict(color="#212121", width=1), row=1, col=1)

    # 1.3 วาด Loads (Smart Scaling)
    for l in raw_loads:
        # Load Color: Live=Red, Dead=Blue
        color = "#D32F2F" if l['case'] == 'LL' else "#1976D2"
        fill_color = "rgba(211, 47, 47, 0.15)" if l['case'] == 'LL' else "rgba(25, 118, 210, 0.15)"
        
        # Calculate Visual Height (Logarithmic-like dampening to prevent huge blocks)
        # สูตร: เอา ratio มาถอด root จะช่วยลดความต่างระหว่าง load น้อยกับเยอะ
        ratio = (l['mag'] / max_load_val) ** 0.8
        this_h = max_load_h * ratio
        this_h = max(0.5, this_h) # ขั้นต่ำต้องสูง 0.5 หน่วย
        
        if l['type'] == 'P':
            # === Point Load ===
            x_loc = nodes[int(l['span_idx'])] + l['x']
            
            fig.add_annotation(
                x=x_loc, y=beam_top,
                ax=0, ay=-60, # Fixed pixel length for clean look
                xref="x1", yref="y1",
                text=f"<b>P={l['mag']:,.0f}</b>",
                showarrow=True, arrowhead=2, arrowwidth=2.5, arrowcolor=color, arrowsize=1.2,
                font=dict(color=color, size=11, family="Arial"),
                bgcolor="white", bordercolor=color, borderwidth=1, borderpad=2,
                row=1, col=1
            )
            
        elif l['type'] == 'U':
            # === Uniform Load (Comb Style) ===
            start_x = nodes[int(l['span_idx'])] + l['x']
            end_x = nodes[int(l['span_idx'])] + l.get('end', spans[int(l['span_idx'])])
            
            # 1. Shaded Block
            fig.add_trace(go.Scatter(
                x=[start_x, end_x, end_x, start_x],
                y=[beam_top, beam_top, beam_top + this_h, beam_top + this_h],
                fill='toself', fillcolor=fill_color,
                mode='none', hoverinfo='skip', showlegend=False
            ), row=1, col=1)
            
            # 2. Top Bar
            fig.add_trace(go.Scatter(
                x=[start_x, end_x], y=[beam_top + this_h, beam_top + this_h],
                mode='lines', line=dict(color=color, width=2),
                hoverinfo='skip', showlegend=False
            ), row=1, col=1)
            
            # 3. Arrows (Comb)
            n_arrows = max(3, int((end_x - start_x) * 2.5))
            for ax in np.linspace(start_x, end_x, n_arrows):
                # Line shaft
                fig.add_shape(type="line",
                    x0=ax, x1=ax, y0=beam_top, y1=beam_top + this_h,
                    line=dict(color=color, width=1.5), layer="below", row=1, col=1
                )
                # Arrow head
                fig.add_trace(go.Scatter(
                    x=[ax], y=[beam_top],
                    mode='markers', marker=dict(symbol="triangle-down", size=7, color=color),
                    hoverinfo='skip', showlegend=False
                ), row=1, col=1)
                
            # Label (Floating above)
            fig.add_annotation(
                x=(start_x + end_x)/2, y=beam_top + this_h,
                text=f"<b>w={l['mag']:,.0f}</b>",
                yshift=12, showarrow=False,
                font=dict(color=color, size=11), bgcolor="white", opacity=0.9,
                row=1, col=1
            )

    # ==========================================
    # ROW 2: SHEAR (SFD)
    # ==========================================
    fig.add_trace(go.Scatter(
        x=df['x'], y=df['shear'], mode='lines', line_shape='hv',
        fill='tozeroy', line=dict(color='#FB8C00', width=2),
        fillcolor='rgba(251, 140, 0, 0.2)', name="Shear"
    ), row=2, col=1)
    
    # Annotate Max/Min V
    v_max, v_min = df['shear'].max(), df['shear'].min()
    fig.add_annotation(x=df.loc[df['shear'].idxmax(), 'x'], y=v_max, text=f"{v_max:.0f}", showarrow=False, yshift=10, font=dict(color="#E65100"), row=2, col=1)
    fig.add_annotation(x=df.loc[df['shear'].idxmin(), 'x'], y=v_min, text=f"{v_min:.0f}", showarrow=False, yshift=-10, font=dict(color="#E65100"), row=2, col=1)

    # ==========================================
    # ROW 3: MOMENT (BMD) - TENSION SIDE
    # ==========================================
    # Concept: Tension Side Plotting
    # Positive Moment (Sagging) -> Tension Bottom -> Plot DOWN
    # Negative Moment (Hogging) -> Tension Top -> Plot UP
    
    # แยกสีเพื่อความชัดเจน (Reinforcement Zones)
    # Tension Bottom (+M) = Blue
    # Tension Top (-M) = Red
    
    df['m_pos'] = df['moment'].apply(lambda x: x if x >= 0 else 0)
    df['m_neg'] = df['moment'].apply(lambda x: x if x < 0 else 0)

    # Plot Positive (Sagging) -> Will appear BELOW axis because we reverse Y
    fig.add_trace(go.Scatter(
        x=df['x'], y=df['m_pos'], 
        mode='lines', line=dict(width=0),
        fill='tozeroy', fillcolor='rgba(33, 150, 243, 0.4)', # Blue
        name="Bot. Rebar (+M)"
    ), row=3, col=1)
    
    # Plot Negative (Hogging) -> Will appear ABOVE axis because we reverse Y
    fig.add_trace(go.Scatter(
        x=df['x'], y=df['m_neg'], 
        mode='lines', line=dict(width=0),
        fill='tozeroy', fillcolor='rgba(233, 30, 99, 0.4)', # Red
        name="Top Rebar (-M)"
    ), row=3, col=1)

    # Main Line
    fig.add_trace(go.Scatter(
        x=df['x'], y=df['moment'], 
        mode='lines', line=dict(color='#455A64', width=2.5),
        showlegend=False
    ), row=3, col=1)

    # Annotate Extremes (Label placement handled automatically but text is key)
    m_max = df['moment'].max() # Sagging
    m_min = df['moment'].min() # Hogging
    
    # Sagging Label (Bottom)
    if m_max > 1:
        fig.add_annotation(
            x=df.loc[df['moment'].idxmax(), 'x'], y=m_max, 
            text=f"<b>M(+): {m_max:,.0f}</b><br><span style='font-size:9px'>Bot. Steel</span>", 
            showarrow=True, arrowcolor="#1976D2", yshift=10, 
            font=dict(color="#1976D2"), row=3, col=1
        )
        
    # Hogging Label (Top)
    if abs(m_min) > 1:
        fig.add_annotation(
            x=df.loc[df['moment'].idxmin(), 'x'], y=m_min, 
            text=f"<b>M(-): {m_min:,.0f}</b><br><span style='font-size:9px'>Top Steel</span>", 
            showarrow=True, arrowcolor="#C2185B", yshift=-10,
            font=dict(color="#C2185B"), row=3, col=1
        )

    # ==========================================
    # ROW 4: DEFLECTION
    # ==========================================
    defl_mm = df['deflection'] * 1000
    fig.add_trace(go.Scatter(
        x=df['x'], y=defl_mm, mode='lines',
        line=dict(color='#43A047', width=2.5, dash='dot'), name="Deflection"
    ), row=4, col=1)
    
    # Max Deflection Label
    d_abs_max = defl_mm.abs().max()
    d_idx = defl_mm.abs().idxmax()
    d_val = defl_mm[d_idx]
    
    fig.add_annotation(
        x=df.loc[d_idx, 'x'], y=d_val,
        text=f"δ max: {d_val:.2f} mm",
        showarrow=True, arrowhead=1, row=4, col=1
    )

    # ==========================================
    # GLOBAL LAYOUT STYLING
    # ==========================================
    fig.update_layout(
        height=1300, # สูงขึ้นเพื่อให้กราฟไม่อัดแน่น
        template="plotly_white",
        hovermode="x unified",
        margin=dict(l=50, r=50, t=60, b=50),
        font=dict(family="Roboto, sans-serif", size=12),
        showlegend=False
    )

    # Grid & Axis Settings
    grid_style = dict(showgrid=True, gridcolor='#F5F5F5', showline=True, linewidth=1, linecolor='black', mirror=True)
    
    # Row 1: Load (Hidden Y)
    fig.update_yaxes(visible=False, showticklabels=False, row=1, col=1)
    
    # Row 2: Shear
    fig.update_yaxes(title="<b>V</b> (kg)", **grid_style, row=2, col=1)
    fig.add_hline(y=0, line_width=1, line_color="black", row=2, col=1)
    
    # Row 3: Moment (INVERTED AXIS HERE!)
    # autorange="reversed" คือหัวใจสำคัญของ "Tension Side" Plot
    fig.update_yaxes(title="<b>M</b> (kg-m)", autorange="reversed", **grid_style, row=3, col=1)
    fig.add_hline(y=0, line_width=1, line_color="black", row=3, col=1)
    
    # Row 4: Deflection
    fig.update_yaxes(title="<b>δ</b> (mm)", **grid_style, row=4, col=1)
    fig.update_xaxes(title="<b>Distance</b> (m)", showgrid=True, row=4, col=1)

    st.plotly_chart(fig, use_container_width=True)

def render_result_tables(df, reac, spans, u_force, u_len):
    st.markdown("---")
    st.subheader("📋 Structural Analysis Summary")
    
    c1, c2 = st.columns([1, 2])
    
    with c1:
        st.markdown("**📍 Support Reactions**")
        if reac is not None:
            r_data = []
            for i in range(len(reac)//2):
                fy = reac[2*i]
                mz = reac[2*i+1]
                if abs(fy)>0.1 or abs(mz)>0.1:
                    r_data.append({"Node": f"#{i}", "Ry (kg)": f"{fy:,.2f}", "Mz (kg-m)": f"{mz:,.2f}"})
            st.dataframe(pd.DataFrame(r_data), hide_index=True, use_container_width=True)
            
    with c2:
        st.markdown("**📐 Design Forces (Envelope)**")
        s_data = []
        nodes = [0] + list(np.cumsum(spans))
        for i in range(len(spans)):
            sub = df[(df['x'] >= nodes[i]) & (df['x'] <= nodes[i+1])]
            s_data.append({
                "Span": f"Span {i+1}",
                "V max": f"{sub['shear'].abs().max():,.0f}",
                "+M (Bot)": f"{sub['moment'].max():,.0f}",
                "-M (Top)": f"{sub['moment'].min():,.0f}",
                "δ max": f"{sub['deflection'].abs().max()*1000:.2f}"
            })
        st.dataframe(pd.DataFrame(s_data), hide_index=True, use_container_width=True)
