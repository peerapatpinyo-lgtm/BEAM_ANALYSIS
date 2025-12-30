import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- THEME CONSTANTS ---
C_BEAM = 'black'
C_PIN_ROLLER = 'white' 
C_OUTLINE = 'black'
C_SHEAR_LINE = '#D97706'    # Amber Dark
C_SHEAR_FILL = 'rgba(217, 119, 6, 0.1)' 
C_MOMENT_LINE = '#2563EB'   # Blue Dark
C_MOMENT_FILL = 'rgba(37, 99, 235, 0.1)'

def draw_interactive_diagrams(df, spans, sup_df, loads, unit_force, unit_len):
    st.markdown("### 📊 Structural Analysis")

    # --- 0. STABILITY CHECK (แจ้งเตือน Unstable) ---
    # ถ้า df เป็น None หรือมีค่า NaN/Infinity แสดงว่าคำนวณไม่ได้ (Unstable)
    if df is None or df.isnull().values.any() or not np.isfinite(df['shear']).all():
        st.error("⚠️ **Structure is Unstable!** Please check your supports (e.g., missing constraints or parallel rollers).")
        return # หยุดการวาดกราฟทันที

    total_len = sum(spans)
    cum_spans = [0] + list(np.cumsum(spans))
    n_nodes = len(cum_spans)
    
    # --- 1. LOAD SCALING LOGIC (คำนวณสัดส่วนแรง) ---
    # หาค่า Max เพื่อเทียบอัตราส่วน (ไม่ให้แรง 1 เท่ากับแรง 2)
    max_P = 1.0
    max_w = 1.0
    
    if loads:
        p_vals = [abs(l['P']) for l in loads if l['type'] == 'P']
        w_vals = [abs(l['w']) for l in loads if l['type'] == 'U']
        if p_vals: max_P = max(p_vals)
        if w_vals: max_w = max(w_vals)

    # --- 2. SETUP LAYOUT ---
    fig = make_subplots(
        rows=3, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.15, 
        row_heights=[0.30, 0.35, 0.35], # Model ให้พื้นที่เยอะหน่อยสำหรับลูกศร
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
                    x=[x], y=[-0.12], 
                    mode='markers',
                    marker=dict(symbol='triangle-up', size=18, color=C_PIN_ROLLER, line=dict(width=2, color=C_OUTLINE)),
                    hoverinfo='name', name=f"Pin"
                ), row=1, col=1)
                # Ground
                fig.add_shape(type="line", x0=x-0.2, y0=-0.22, x1=x+0.2, y1=-0.22, line=dict(color='black', width=2), row=1, col=1)
                for hx in np.linspace(x-0.2, x+0.2, 5):
                    fig.add_shape(type="line", x0=hx, y0=-0.22, x1=hx-0.05, y1=-0.28, line=dict(color='black', width=1), row=1, col=1)

            elif sType == 'Roller':
                fig.add_trace(go.Scatter(
                    x=[x], y=[-0.12],
                    mode='markers',
                    marker=dict(symbol='circle', size=16, color=C_PIN_ROLLER, line=dict(width=2, color=C_OUTLINE)),
                    hoverinfo='name', name=f"Roller"
                ), row=1, col=1)
                # Ground
                fig.add_shape(type="line", x0=x-0.2, y0=-0.22, x1=x+0.2, y1=-0.22, line=dict(color='black', width=2), row=1, col=1)
                for hx in np.linspace(x-0.2, x+0.2, 5):
                    fig.add_shape(type="line", x0=hx, y0=-0.22, x1=hx-0.05, y1=-0.28, line=dict(color='black', width=1), row=1, col=1)

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
                marker=dict(symbol='circle', size=9, color='white', line=dict(width=2, color='black')),
                hoverinfo='name', name=f"Node {i+1}"
            ), row=1, col=1)

    # 1.3 Loads (Dynamic Scaling)
    for l in loads:
        x_s = cum_spans[int(l['span_idx'])]
        
        if l['type'] == 'P':
            # --- Dynamic Arrow Length ---
            # Min length = 40, Max adds up to 60 more pixels based on ratio
            ratio = abs(l['P']) / max_P if max_P > 0 else 0
            arrow_len = 40 + (ratio * 50) 
            
            text_label = f"<b>P={l['P']} {unit_force}</b>"
            fig.add_annotation(
                x=x_s + l['x'], y=0, ax=0, ay=-arrow_len, 
                arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor='#DC2626',
                text=text_label, font=dict(color='#DC2626', size=12),
                bgcolor="rgba(255,255,255,0.85)",
                row=1, col=1
            )
            
        elif l['type'] == 'U':
            # --- Dynamic Block Height ---
            ratio = abs(l['w']) / max_w if max_w > 0 else 0
            # Min height 0.15, Max adds 0.25
            block_h = 0.15 + (ratio * 0.25)
            
            x_e = cum_spans[int(l['span_idx'])+1]
            text_label = f"<b>w={l['w']} {unit_force}/{unit_len}</b>"
            
            # Draw Rect
            fig.add_shape(type="rect", x0=x_s, y0=0.15, x1=x_e, y1=0.15+block_h, line_width=0, fillcolor='#DC2626', opacity=0.15, row=1, col=1)
            # Label at top of rect
            fig.add_annotation(
                x=(x_s+x_e)/2, y=0.15+block_h+0.05, showarrow=False, 
                text=text_label, font=dict(color='#DC2626', size=12),
                bgcolor="rgba(255,255,255,0.85)",
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
        name='Shear'
    ), row=2, col=1)

    # Labels
    v_max = df['shear'].abs().max()
    if v_max > 0:
        idx = df['shear'].abs().idxmax()
        row_v = df.loc[idx]
        val = row_v['shear']
        x_loc = row_v['x']
        
        yshift = 25 if val > 0 else -25
        xanchor = 'center'
        if x_loc < total_len * 0.1: xanchor = 'left'
        elif x_loc > total_len * 0.9: xanchor = 'right'

        fig.add_annotation(
            x=x_loc, y=val, 
            text=f"<b>{val:.2f}</b>",
            showarrow=False, yshift=yshift, xanchor=xanchor,
            font=dict(color=C_SHEAR_LINE, size=12),
            bgcolor="rgba(255,255,255,0.8)",
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
        name='Moment'
    ), row=3, col=1)

    # Labels
    m_max = df['moment'].max()
    m_min = df['moment'].min()
    
    # Max (+)
    if m_max > 0.01:
        xm = df.loc[df['moment'] == m_max, 'x'].iloc[0]
        xanchor = 'center'
        if xm < total_len * 0.1: xanchor = 'left'
        elif xm > total_len * 0.9: xanchor = 'right'

        fig.add_annotation(
            x=xm, y=m_max, 
            text=f"<b>{m_max:.2f}</b>",
            showarrow=False, yshift=25, xanchor=xanchor,
            font=dict(color=C_MOMENT_LINE, size=12),
            bgcolor="rgba(255,255,255,0.8)",
            row=3, col=1
        )

    # Min (-)
    if m_min < -0.01:
        xm = df.loc[df['moment'] == m_min, 'x'].iloc[0]
        xanchor = 'center'
        if xm < total_len * 0.1: xanchor = 'left'
        elif xm > total_len * 0.9: xanchor = 'right'

        fig.add_annotation(
            x=xm, y=m_min, 
            text=f"<b>{m_min:.2f}</b>",
            showarrow=False, yshift=-25, xanchor=xanchor,
            font=dict(color=C_MOMENT_LINE, size=12),
            bgcolor="rgba(255,255,255,0.8)",
            row=3, col=1
        )

    # ==========================================
    # GLOBAL LAYOUT
    # ==========================================
    fig.update_layout(
        height=1000,
        margin=dict(l=60, r=40, t=60, b=50),
        plot_bgcolor='white',
        paper_bgcolor='white',
        showlegend=False,
        hovermode="x unified",
        font=dict(family="Arial, sans-serif", size=14, color='black')
    )

    # Clean Axis (No Box)
    ax_style = dict(
        showline=True, linewidth=1, linecolor='black',
        mirror=False, # No Border
        showgrid=True, gridcolor='#F0F0F0',
        ticks="outside", tickwidth=1, ticklen=5,
    )

    fig.update_xaxes(**ax_style)
    fig.update_yaxes(**ax_style)

    # Setup Axes Titles
    fig.update_yaxes(visible=False, row=1, col=1, range=[-0.6, 1.0]) # เพิ่ม Range บน Model เผื่อกราฟ Load สูง
    fig.update_xaxes(visible=True, showticklabels=True, title_text="", row=1, col=1)

    fig.update_yaxes(title_text=f"<b>Shear ({unit_force})</b>", row=2, col=1)
    fig.update_yaxes(title_text=f"<b>Moment ({unit_force}-{unit_len})</b>", row=3, col=1)
    fig.update_xaxes(title_text=f"<b>Distance ({unit_len})</b>", row=3, col=1)

    st.plotly_chart(fig, use_container_width=True)

def render_result_tables(df, reac, spans):
    # Check Stability again for table
    if df is None or df.isnull().values.any():
        return

    st.markdown("---")
    
    # Summary Cards
    c1, c2, c3 = st.columns(3)
    c1.metric("Max Shear", f"{df['shear'].abs().max():.2f}")
    c2.metric("Max Moment (+)", f"{df['moment'].max():.2f}")
    c3.metric("Max Moment (-)", f"{df['moment'].min():.2f}")

    # Reactions Table
    st.subheader("📍 Support Reactions")
    data = []
    n_nodes = len(spans) + 1
    
    # Handle reaction array mismatch (safeguard)
    if len(reac) >= 2*n_nodes:
        for i in range(n_nodes):
            ry = reac[2*i]
            mz = reac[2*i+1]
            data.append({"Node": i+1, "Ry": f"{ry:.2f}", "Mz": f"{mz:.2f}" if abs(mz)>0.001 else "-"})
        st.dataframe(pd.DataFrame(data), hide_index=True, use_container_width=True)
    else:
        st.error("Error in Reaction Data")
