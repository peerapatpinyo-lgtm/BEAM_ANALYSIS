import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
import streamlit as st

def plot_analysis_results(res_df, spans, supports, loads, reactions):
    """
    Creates a Textbook-style structural analysis plot.
    Includes internal unit conversion (/1000) for display.
    """
    
    # --- Create Subplots ---
    fig = make_subplots(
        rows=4, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.08,
        subplot_titles=(
            "<b>1. Free Body Diagram (FBD)</b>", 
            "<b>2. Shear Force Diagram (SFD)</b>", 
            "<b>3. Bending Moment Diagram (BMD)</b>",
            "<b>4. Elastic Curve (Deflection)</b>"
        ),
        row_heights=[0.20, 0.25, 0.25, 0.30]
    )

    # ==========================================
    # ROW 1: LOAD MODEL (FREE BODY DIAGRAM)
    # ==========================================
    total_L = sum(spans)
    cum_dist = [0] + list(np.cumsum(spans))
    
    # Beam Line
    fig.add_trace(go.Scatter(
        x=[0, total_L], y=[0, 0], 
        mode='lines', line=dict(color='black', width=4), hoverinfo='skip'
    ), row=1, col=1)
    
    # Supports
    for idx, row in supports.iterrows():
        sym = "triangle-up"
        if row['type'] == 'Fixed': sym = "square"
        elif row['type'] == 'Roller': sym = "circle"
        
        fig.add_trace(go.Scatter(
            x=[row['x']], y=[-0.08], 
            mode='markers+text',
            marker=dict(symbol=sym, size=14, color='white', line=dict(width=2, color='black')),
            text=[row['type'][0]], textposition="bottom center",
            hoverinfo='name', name=f"Support"
        ), row=1, col=1)

    # Loads
    if isinstance(loads, pd.DataFrame):
        load_iter = loads.to_dict('records')
    else:
        load_iter = loads

    for l in load_iter:
        span_idx = int(l['span_index'])
        start_x = cum_dist[span_idx]
        mag_kN = l['mag'] / 1000.0 # แสดงผลเป็น kN
        
        if l['type'] == 'P':
            x_loc = start_x + float(l['d_start']) 
            fig.add_annotation(
                x=x_loc, y=0, ax=0, ay=-50,
                xref="x1", yref="y1",
                showarrow=True, arrowhead=2, arrowsize=1.2, arrowwidth=2, arrowcolor="#c0392b",
                text=f"<b>P={mag_kN:.2f} kN</b>", yshift=5, row=1, col=1
            )

        elif l['type'] == 'U':
            x_s = start_x + float(l.get('d_start', 0))
            x_e = x_s + float(l['dist'])
            h_vis = 0.25
            
            fig.add_trace(go.Scatter(
                x=[x_s, x_e], y=[h_vis, h_vis],
                mode='lines', line=dict(color='#2980b9', width=2), hoverinfo='skip'
            ), row=1, col=1)
            
            n_arrows = max(3, int(float(l['dist']) * 3)) 
            arrow_x = np.linspace(x_s, x_e, n_arrows)
            for ax_x in arrow_x:
                fig.add_annotation(
                    x=ax_x, y=0, ax=0, ay=-30,
                    xref="x1", yref="y1",
                    showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1, arrowcolor="#2980b9",
                    row=1, col=1
                )
            
            fig.add_annotation(
                x=(x_s+x_e)/2, y=h_vis,
                text=f"<b>w={mag_kN:.2f} kN/m</b>",
                showarrow=False, yshift=10, font=dict(color="#2980b9"),
                row=1, col=1
            )

    # ==========================================
    # ROW 2: SHEAR FORCE (SFD)
    # ==========================================
    fig.add_hline(y=0, line_color="black", line_width=1, row=2, col=1)
    fig.add_trace(go.Scatter(
        x=res_df['x'], y=res_df['shear']/1000, 
        mode='lines', name='Shear (kN)', line=dict(color='#e74c3c', width=2),
        fill='tozeroy', fillcolor='rgba(231, 76, 60, 0.1)'
    ), row=2, col=1)
    
    # Label Max/Min Shear
    v_max = res_df['shear'].max() / 1000
    v_min = res_df['shear'].min() / 1000
    for val in [v_max, v_min]:
        if abs(val) > 0.01:
            idx = (res_df['shear']/1000 - val).abs().idxmin()
            fig.add_annotation(
                x=res_df['x'].iloc[idx], y=val,
                text=f"<b>{val:.2f} kN</b>", showarrow=False, yshift=15 if val>0 else -15,
                font=dict(color='#e74c3c', size=11), row=2, col=1
            )

    # ==========================================
    # ROW 3: BENDING MOMENT (BMD)
    # ==========================================
    fig.add_hline(y=0, line_color="black", line_width=1, row=3, col=1)
    fig.add_trace(go.Scatter(
        x=res_df['x'], y=res_df['moment']/1000, 
        mode='lines', name='Moment (kNm)', line=dict(color='#27ae60', width=2),
        fill='tozeroy', fillcolor='rgba(39, 174, 96, 0.1)'
    ), row=3, col=1)

    # Label Max/Min Moment
    m_max = res_df['moment'].max() / 1000
    m_min = res_df['moment'].min() / 1000
    for val in [m_max, m_min]:
        if abs(val) > 0.01:
            idx = (res_df['moment']/1000 - val).abs().idxmin()
            fig.add_annotation(
                x=res_df['x'].iloc[idx], y=val,
                text=f"<b>{val:.2f} kN-m</b>", 
                showarrow=True, arrowhead=1, ay=30 if val>0 else -30,
                font=dict(color='#27ae60', size=11), row=3, col=1
            )

    # ==========================================
    # ROW 4: DEFLECTION (Elastic Curve)
    # ==========================================
    fig.add_hline(y=0, line_color="black", line_width=1, row=4, col=1)
    fig.add_trace(go.Scatter(
        x=res_df['x'], y=res_df['deflection'], 
        mode='lines', name='Deflection (mm)', line=dict(color='#8e44ad', width=2)
    ), row=4, col=1)
    
    idx_max_def = res_df['deflection'].abs().idxmax()
    max_def_val = res_df['deflection'].iloc[idx_max_def]
    
    fig.add_annotation(
        x=res_df['x'].iloc[idx_max_def], y=max_def_val,
        text=f"<b>Max δ: {max_def_val:.3f} mm</b>",
        showarrow=True, arrowhead=1, 
        ay=40 if max_def_val < 0 else -40,
        font=dict(color='#8e44ad', size=11), row=4, col=1
    )

    # ==========================================
    # LAYOUT & STYLING
    # ==========================================
    for x_pos in cum_dist:
        fig.add_vline(x=x_pos, line_width=1, line_dash="dash", line_color="gray", opacity=0.3)

    fig.update_layout(
        title="<b>Structural Analysis & Design Forces</b>",
        height=950, 
        showlegend=False, 
        template="plotly_white", 
        hovermode="x unified",
        margin=dict(t=80, b=60, l=60, r=20)
    )
    
    fig.update_yaxes(visible=False, range=[-0.5, 0.8], row=1, col=1)
    fig.update_yaxes(title_text="V (kN)", showgrid=True, row=2, col=1)
    # BMD: นิยมวาดโมเมนต์บวกด้านล่าง (autorange='reversed')
    fig.update_yaxes(title_text="M (kN-m)", autorange="reversed", showgrid=True, row=3, col=1)
    fig.update_yaxes(title_text="δ (mm)", showgrid=True, zeroline=True, row=4, col=1)
    fig.update_xaxes(title_text="Beam Length, x (m)", row=4, col=1)

    return fig

def display_design_comparison(mu_pos, mu_neg, vu, design_res):
    """
    NEW: Dashboard สำหรับเปรียบเทียบค่าดีไซน์ Mu vs PhiMn และ Vu vs PhiVn
    แบบ Interactive พร้อมหน่วยครบถ้วน
    """
    st.markdown("---")
    st.subheader("🛠 Interactive RC Design Check")
    
    # ดึงค่าจาก Span แรกเป็นตัวอย่าง (หรือวน Loop ตามจำนวน Span)
    # ในที่นี้สมมติค่า Mu_pos, Mu_neg, Vu เป็นค่าสูงสุดที่เกิดในคาน
    
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.markdown("**Flexure Check (Positive)**")
        phi_mn_pos = design_res.get('phi_Mn_pos', 0.0)
        st.write(f"Demand $M_u^+$: `{mu_pos:.2f}` kN-m")
        st.write(f"Capacity $\phi M_n^+$: `{phi_mn_pos:.2f}` kN-m")
        if phi_mn_pos >= mu_pos:
            st.success("✅ PASS")
        else:
            st.error("❌ FAIL")

    with c2:
        st.markdown("**Flexure Check (Negative)**")
        phi_mn_neg = design_res.get('phi_Mn_neg', 0.0)
        st.write(f"Demand $M_u^-$: `{abs(mu_neg):.2f}` kN-m")
        st.write(f"Capacity $\phi M_n^-$: `{phi_mn_neg:.2f}` kN-m")
        if phi_mn_neg >= abs(mu_neg):
            st.success("✅ PASS")
        else:
            st.error("❌ FAIL")

    with c3:
        st.markdown("**Shear Check**")
        phi_vn = design_res.get('phi_Vn', 0.0)
        st.write(f"Demand $V_u$: `{vu:.2f}` kN")
        st.write(f"Capacity $\phi V_n$: `{phi_vn:.2f}` kN")
        if phi_vn >= vu:
            st.success("✅ PASS")
        else:
            st.error("❌ FAIL")
            
    # รายละเอียดเหล็กเสริม
    st.info(f"💡 **Rebar Info:** Top {design_res['top_n']}DB{design_res['top_db']} | "
            f"Bot {design_res['bot_n']}DB{design_res['bot_db']} | "
            f"Stirrup RB{design_res['stir_db']}@{design_res['stir_spacing']} mm")
