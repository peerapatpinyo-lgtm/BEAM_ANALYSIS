import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- PROFESSIONAL COLORS ---
C_SHEAR_LINE = '#D97706'  # Engineering Orange
C_SHEAR_FILL = 'rgba(217, 119, 6, 0.1)' 
C_MOMENT_LINE = '#2563EB' # Engineering Blue
C_MOMENT_FILL = 'rgba(37, 99, 235, 0.1)'
C_ZERO_LINE = 'black'
C_TEXT = '#374151'

def draw_interactive_diagrams(df, spans, sup_df, loads, unit_force, unit_len):
    total_len = sum(spans)
    cum_spans = [0] + list(np.cumsum(spans))
    
    # Setup Layout 3 Rows
    fig = make_subplots(
        rows=3, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.1,
        row_heights=[0.2, 0.4, 0.4],
        subplot_titles=("<b>1. Structural Model</b>", "<b>2. Shear Force Diagram (SFD)</b>", "<b>3. Bending Moment Diagram (BMD)</b>")
    )

    # === 1. MODEL ===
    # Beam
    fig.add_trace(go.Scatter(
        x=[0, total_len], y=[0, 0],
        mode='lines', line=dict(color='black', width=5),
        hoverinfo='skip'
    ), row=1, col=1)

    # Supports
    sup_x = [cum_spans[int(row['id'])] for _, row in sup_df.iterrows()]
    sup_txt = [row['type'] for _, row in sup_df.iterrows()]
    fig.add_trace(go.Scatter(
        x=sup_x, y=[0]*len(sup_x),
        mode='markers+text',
        marker=dict(symbol='triangle-up', size=20, color='#6B7280', line=dict(width=2, color='black')),
        text=sup_txt, textposition="bottom center",
        hoverinfo='name', name='Support'
    ), row=1, col=1)

    # Loads Arrows
    for l in loads:
        x_s = cum_spans[int(l['span_idx'])]
        if l['type'] == 'P':
            fig.add_annotation(
                x=x_s + l['x'], y=0, ax=0, ay=-60,
                arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor='#EF4444',
                text=f"P={l['P']}", font=dict(color='#EF4444', size=12, weight='bold'),
                bgcolor="white", borderpad=2,
                row=1, col=1
            )
        elif l['type'] == 'U':
            x_e = cum_spans[int(l['span_idx'])+1]
            fig.add_shape(type="rect", x0=x_s, y0=0.1, x1=x_e, y1=0.3, line_width=0, fillcolor='#EF4444', opacity=0.15, row=1, col=1)
            fig.add_annotation(x=(x_s+x_e)/2, y=0.35, showarrow=False, text=f"w={l['w']}", font=dict(color='#EF4444'), row=1, col=1)

    # === 2. SHEAR FORCE ===
    # Zero Line
    fig.add_shape(type="line", x0=0, x1=total_len, y0=0, y1=0, line=dict(color='black', width=1), row=2, col=1)
    
    # Plot SFD
    fig.add_trace(go.Scatter(
        x=df['x'], y=df['shear'],
        mode='lines', 
        line=dict(color=C_SHEAR_LINE, width=2.5, shape='linear'), # shape linear + dense points = vertical line
        fill='tozeroy', fillcolor=C_SHEAR_FILL,
        name='Shear (V)',
        hovertemplate='x: %{x:.2f}<br>V: %{y:.2f}'
    ), row=2, col=1)

    # Annotate Max Shear
    v_max = df['shear'].abs().max()
    if v_max > 0.01:
        # Find index of max
        idx_max = df['shear'].abs().idxmax()
        row_max = df.loc[idx_max]
        fig.add_annotation(
            x=row_max['x'], y=row_max['shear'],
            text=f"V_max: {row_max['shear']:.2f}",
            showarrow=True, arrowhead=1, ax=40, ay=-30 if row_max['shear']>0 else 30,
            bgcolor="rgba(255,255,255,0.8)", bordercolor=C_SHEAR_LINE,
            row=2, col=1
        )

    # === 3. BENDING MOMENT ===
    # Zero Line
    fig.add_shape(type="line", x0=0, x1=total_len, y0=0, y1=0, line=dict(color='black', width=1), row=3, col=1)

    # Plot BMD
    fig.add_trace(go.Scatter(
        x=df['x'], y=df['moment'],
        mode='lines', 
        line=dict(color=C_MOMENT_LINE, width=2.5),
        fill='tozeroy', fillcolor=C_MOMENT_FILL,
        name='Moment (M)',
        hovertemplate='x: %{x:.2f}<br>M: %{y:.2f}'
    ), row=3, col=1)

    # Annotate Max/Min Moment
    m_max = df['moment'].max()
    m_min = df['moment'].min()
    
    # Max (+)
    if abs(m_max) > 0.01:
        xm = df.loc[df['moment'] == m_max, 'x'].iloc[0]
        fig.add_annotation(
            x=xm, y=m_max, text=f"M(+): {m_max:.2f}", 
            showarrow=True, arrowhead=1, ax=0, ay=-30,
            bgcolor="rgba(255,255,255,0.8)", bordercolor=C_MOMENT_LINE,
            row=3, col=1
        )
    # Min (-)
    if abs(m_min) > 0.01:
        xm = df.loc[df['moment'] == m_min, 'x'].iloc[0]
        fig.add_annotation(
            x=xm, y=m_min, text=f"M(-): {m_min:.2f}", 
            showarrow=True, arrowhead=1, ax=0, ay=30,
            bgcolor="rgba(255,255,255,0.8)", bordercolor=C_MOMENT_LINE,
            row=3, col=1
        )

    # === SETTINGS ===
    fig.update_layout(
        height=800,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family="Sarabun, sans-serif", size=14, color=C_TEXT),
        showlegend=False,
        hovermode="x unified"
    )
    
    # Axis Grid
    grid_style = dict(showline=True, linewidth=1, linecolor='#E5E7EB', showgrid=True, gridcolor='#F3F4F6')
    fig.update_xaxes(**grid_style)
    fig.update_yaxes(**grid_style)
    fig.update_yaxes(visible=False, range=[-0.5, 0.5], row=1, col=1) # Lock model axis

    st.plotly_chart(fig, use_container_width=True)

def render_result_tables(df, reac, spans):
    st.markdown("#### 4. สรุปผลการคำนวณ (Analysis Summary)")
    
    # 1. Metric Cards
    c1, c2, c3 = st.columns(3)
    c1.metric("Max Shear (แรงเฉือนสูงสุด)", f"{df['shear'].abs().max():.2f}")
    c2.metric("Max Moment (+) (บวกสูงสุด)", f"{df['moment'].max():.2f}")
    c3.metric("Max Moment (-) (ลบสูงสุด)", f"{df['moment'].min():.2f}")

    # 2. Tabs
    t1, t2 = st.tabs(["📍 Support Reactions", "📄 Detailed Table"])
    
    with t1:
        # Display Reactions nicely
        # Note: reactions from solver are in global DOF order
        data = []
        n_nodes = len(spans) + 1
        for i in range(n_nodes):
            ry = reac[2*i]
            mz = reac[2*i+1]
            # Only show if significant
            val_ry = f"{ry:.2f}"
            val_mz = f"{mz:.2f}" if abs(mz) > 0.001 else "-"
            data.append({"Node": i+1, "Vertical Reaction (Ry)": val_ry, "Moment Reaction (Mz)": val_mz})
        st.dataframe(pd.DataFrame(data), hide_index=True, use_container_width=True)

    with t2:
        # Clean Table for export
        df_export = df.copy()
        df_export.columns = ["Distance (x)", "Shear (V)", "Moment (M)"]
        st.dataframe(df_export.style.format("{:.2f}"), use_container_width=True, height=400)
