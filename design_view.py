import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- THEME CONSTANTS ---
C_BEAM = 'black'
C_PIN_ROLLER = 'white' 
C_OUTLINE = 'black'
C_SHEAR_LINE = '#D97706'    # Amber
C_SHEAR_FILL = 'rgba(217, 119, 6, 0.1)' 
C_MOMENT_LINE = '#2563EB'   # Blue
C_MOMENT_FILL = 'rgba(37, 99, 235, 0.1)'
C_DEFLECT_LINE = '#10B981'  # Green
C_DEFLECT_FILL = 'rgba(16, 185, 129, 0.1)'
C_TEXT_BG = "rgba(255,255,255,0.85)"

def draw_interactive_diagrams(df, spans, sup_df, loads, unit_force, unit_len):
    st.markdown("### 📊 Structural Analysis Results")

    # --- 0. STABILITY CHECK ---
    if df is None or df.empty or df.isnull().values.any():
        st.error("⚠️ **Structure is Unstable!** Please check supports.")
        return

    total_len = sum(spans)
    cum_spans = [0] + list(np.cumsum(spans))
    n_nodes = len(cum_spans)
    
    # --- 1. SETUP LAYOUT (4 ROWS) ---
    fig = make_subplots(
        rows=4, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.08, 
        row_heights=[0.20, 0.25, 0.25, 0.30], 
        subplot_titles=(
            "<b>1. Structural Model</b>", 
            "<b>2. Shear Force Diagram (SFD)</b>", 
            "<b>3. Bending Moment Diagram (BMD)</b>",
            "<b>4. Deflection Diagram</b>"
        )
    )

    # ==========================================
    # ROW 1: STRUCTURAL MODEL
    # ==========================================
    fig.add_trace(go.Scatter(
        x=[0, total_len], y=[0, 0],
        mode='lines', line=dict(color=C_BEAM, width=5), hoverinfo='skip'
    ), row=1, col=1)

    # Supports
    sup_map = {int(r['id']): r['type'] for _, r in sup_df.iterrows()}
    for i in range(n_nodes):
        x = cum_spans[i]
        if i in sup_map:
            sType = sup_map[i]
            if sType == 'Pin':
                fig.add_trace(go.Scatter(x=[x], y=[-0.12], mode='markers', marker=dict(symbol='triangle-up', size=18, color=C_PIN_ROLLER, line=dict(width=2, color=C_OUTLINE)), hoverinfo='name', name="Pin"), row=1, col=1)
                fig.add_shape(type="line", x0=x-0.2, y0=-0.22, x1=x+0.2, y1=-0.22, line=dict(color='black', width=2), row=1, col=1)
            elif sType == 'Roller':
                fig.add_trace(go.Scatter(x=[x], y=[-0.12], mode='markers', marker=dict(symbol='circle', size=16, color=C_PIN_ROLLER, line=dict(width=2, color=C_OUTLINE)), hoverinfo='name', name="Roller"), row=1, col=1)
                fig.add_shape(type="line", x0=x-0.2, y0=-0.22, x1=x+0.2, y1=-0.22, line=dict(color='black', width=2), row=1, col=1)
            elif sType == 'Fixed':
                fig.add_shape(type="line", x0=x, y0=-0.3, x1=x, y1=0.3, line=dict(color='black', width=4), row=1, col=1)
                for hy in np.linspace(-0.3, 0.3, 7):
                     fig.add_shape(type="line", x0=x, y0=hy, x1=x+(0.15 if x==0 else -0.15), y1=hy-0.05, line=dict(color='black', width=1), row=1, col=1)
        else:
            fig.add_trace(go.Scatter(x=[x], y=[0], mode='markers', marker=dict(symbol='circle', size=9, color='white', line=dict(width=2, color='black')), hoverinfo='name', name=f"Node {i+1}"), row=1, col=1)

    # Loads (Refined Scaling)
    # Goal: Point load arrow approx same visual weight as UDL block
    for l in loads:
        x_s = cum_spans[int(l['span_idx'])]
        if l['type'] == 'P':
            # Point Load: Fixed visual range 40-70px to avoid looking too huge or tiny
            arrow_len = 50 
            text_label = f"<b>P={l['P']}</b>"
            fig.add_annotation(
                x=x_s + l['x'], y=0, ax=0, ay=-arrow_len, 
                arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor='#DC2626',
                text=text_label, font=dict(color='#DC2626', size=11),
                bgcolor=C_TEXT_BG, row=1, col=1
            )
        elif l['type'] == 'U':
            # UDL: Reasonable block height
            x_e = cum_spans[int(l['span_idx'])+1]
            text_label = f"<b>w={l['w']}</b>"
            fig.add_shape(type="rect", x0=x_s, y0=0.12, x1=x_e, y1=0.30, line_width=0, fillcolor='#DC2626', opacity=0.15, row=1, col=1)
            fig.add_annotation(x=(x_s+x_e)/2, y=0.35, showarrow=False, text=text_label, font=dict(color='#DC2626', size=11), bgcolor=C_TEXT_BG, row=1, col=1)

    # ==========================================
    # ROW 2: SHEAR (SFD)
    # ==========================================
    fig.add_shape(type="line", x0=0, x1=total_len, y0=0, y1=0, line=dict(color='black', width=1), row=2, col=1)
    fig.add_trace(go.Scatter(x=df['x'], y=df['shear'], mode='lines', line=dict(color=C_SHEAR_LINE, width=2), fill='tozeroy', fillcolor=C_SHEAR_FILL, name='Shear'), row=2, col=1)
    
    # Label
    v_max = df['shear'].abs().max()
    if v_max > 0:
        idx = df['shear'].abs().idxmax()
        row_v = df.loc[idx]
        val = row_v['shear']
        fig.add_annotation(
            x=row_v['x'], y=val, 
            text=f"<b>{val:.2f} @ {row_v['x']:.2f}m</b>",
            showarrow=False, yshift=20 if val>0 else -20,
            font=dict(color=C_SHEAR_LINE, size=11), bgcolor=C_TEXT_BG,
            row=2, col=1
        )

    # ==========================================
    # ROW 3: MOMENT (BMD)
    # ==========================================
    fig.add_shape(type="line", x0=0, x1=total_len, y0=0, y1=0, line=dict(color='black', width=1), row=3, col=1)
    fig.add_trace(go.Scatter(x=df['x'], y=df['moment'], mode='lines', line=dict(color=C_MOMENT_LINE, width=2), fill='tozeroy', fillcolor=C_MOMENT_FILL, name='Moment'), row=3, col=1)
    
    # Labels (Max & Min)
    m_max = df['moment'].max()
    m_min = df['moment'].min()
    if m_max > 0.01:
        xm = df.loc[df['moment'] == m_max, 'x'].iloc[0]
        fig.add_annotation(x=xm, y=m_max, text=f"<b>{m_max:.2f} @ {xm:.2f}m</b>", showarrow=False, yshift=20, font=dict(color=C_MOMENT_LINE, size=11), bgcolor=C_TEXT_BG, row=3, col=1)
    if m_min < -0.01:
        xm = df.loc[df['moment'] == m_min, 'x'].iloc[0]
        fig.add_annotation(x=xm, y=m_min, text=f"<b>{m_min:.2f} @ {xm:.2f}m</b>", showarrow=False, yshift=-20, font=dict(color=C_MOMENT_LINE, size=11), bgcolor=C_TEXT_BG, row=3, col=1)

    # ==========================================
    # ROW 4: DEFLECTION
    # ==========================================
    fig.add_shape(type="line", x0=0, x1=total_len, y0=0, y1=0, line=dict(color='black', width=1), row=4, col=1)
    fig.add_trace(go.Scatter(x=df['x'], y=df['deflection'], mode='lines', line=dict(color=C_DEFLECT_LINE, width=2), fill='tozeroy', fillcolor=C_DEFLECT_FILL, name='Deflection'), row=4, col=1)
    
    # Label Max Deflection
    d_max_abs = df['deflection'].abs().max()
    if d_max_abs > 1e-9:
        idx_d = df['deflection'].abs().idxmax()
        row_d = df.loc[idx_d]
        val_d = row_d['deflection']
        # Convert scientific notation if very small
        txt_d = f"{val_d:.4f}" if abs(val_d) > 0.0001 else f"{val_d:.2e}"
        fig.add_annotation(
            x=row_d['x'], y=val_d, 
            text=f"<b>Max: {txt_d} @ {row_d['x']:.2f}m</b>",
            showarrow=False, yshift=20 if val_d>0 else -20,
            font=dict(color=C_DEFLECT_LINE, size=11), bgcolor=C_TEXT_BG,
            row=4, col=1
        )

    # ==========================================
    # LAYOUT STYLING
    # ==========================================
    fig.update_layout(
        height=1200, 
        margin=dict(l=60, r=40, t=50, b=50),
        plot_bgcolor='white', paper_bgcolor='white',
        showlegend=False, hovermode="x unified",
        font=dict(family="Arial, sans-serif", size=13, color='black')
    )
    
    ax_style = dict(showline=True, linewidth=1, linecolor='black', mirror=False, showgrid=True, gridcolor='#F0F0F0', ticks="outside")
    fig.update_xaxes(**ax_style)
    fig.update_yaxes(**ax_style)

    # Titles
    fig.update_yaxes(visible=False, row=1, col=1, range=[-0.6, 0.6])
    fig.update_xaxes(visible=True, showticklabels=True, title_text="", row=1, col=1)
    
    fig.update_yaxes(title_text=f"<b>V ({unit_force})</b>", row=2, col=1)
    fig.update_yaxes(title_text=f"<b>M ({unit_force}-{unit_len})</b>", row=3, col=1)
    fig.update_yaxes(title_text=f"<b>δ (m)</b>", row=4, col=1)
    fig.update_xaxes(title_text=f"<b>Distance ({unit_len})</b>", row=4, col=1)

    st.plotly_chart(fig, use_container_width=True)

def render_result_tables(df, reac, spans):
    if df is None or df.empty: return
    
    st.markdown("---")
    # Summary Cards
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Max Shear", f"{df['shear'].abs().max():.2f}")
    c2.metric("Max Moment (+)", f"{df['moment'].max():.2f}")
    c3.metric("Max Moment (-)", f"{df['moment'].min():.2f}")
    c4.metric("Max Deflection", f"{df['deflection'].abs().max():.4f}")

    # Reactions
    st.subheader("📍 Support Reactions")
    data = []
    n_nodes = len(spans) + 1
    if len(reac) >= 2*n_nodes:
        for i in range(n_nodes):
            ry = reac[2*i]
            mz = reac[2*i+1]
            data.append({"Node": i+1, "Ry": f"{ry:.2f}", "Mz": f"{mz:.2f}" if abs(mz)>0.001 else "-"})
        st.dataframe(pd.DataFrame(data), hide_index=True, use_container_width=True)
