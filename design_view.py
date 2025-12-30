import streamlit as st
import pandas as pd
import numpy as np
import math
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- COLORS & STYLES ---
C_SHEAR_FILL = 'rgba(245, 158, 11, 0.2)'
C_SHEAR_LINE = '#D97706'
C_MOMENT_FILL = 'rgba(37, 99, 235, 0.2)'
C_MOMENT_LINE = '#2563EB'
C_LOAD = '#DC2626'
C_GRID = '#E5E7EB'

# --- 1. INTERACTIVE PLOTLY DIAGRAMS ---
def draw_interactive_diagrams(df, spans, sup_df, loads, unit_force, unit_len):
    total_len = sum(spans)
    cum_spans = [0] + list(np.cumsum(spans))
    
    # Create Subplots: Model, Shear, Moment
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        row_heights=[0.2, 0.4, 0.4],
        subplot_titles=("<b>STRUCTURAL MODEL</b>", f"<b>SHEAR FORCE ({unit_force})</b>", f"<b>BENDING MOMENT ({unit_force}-{unit_len})</b>")
    )

    # -- ROW 1: Model --
    fig.add_trace(go.Scatter(x=[0, total_len], y=[0, 0], mode='lines', line=dict(color='#1F2937', width=5), hoverinfo='skip'), row=1, col=1)
    
    # Supports
    sup_x = [cum_spans[int(row['id'])] for _, row in sup_df.iterrows()]
    sup_txt = [row['type'] for _, row in sup_df.iterrows()]
    fig.add_trace(go.Scatter(x=sup_x, y=[0]*len(sup_x), mode='markers', marker=dict(symbol='triangle-up', size=15, color='#374151'), text=sup_txt, hoverinfo='text'), row=1, col=1)

    # Loads
    for l in loads:
        x_s = cum_spans[int(l['span_idx'])]
        if l['type'] == 'P':
            fig.add_annotation(x=x_s + l['x'], y=0, ax=0, ay=-40, arrowhead=2, text=f"<b>P={l['P']}</b>", font=dict(color=C_LOAD), row=1, col=1)
        elif l['type'] == 'U':
            x_e = cum_spans[int(l['span_idx'])+1]
            fig.add_shape(type="rect", x0=x_s, y0=0.05, x1=x_e, y1=0.25, line=dict(width=0), fillcolor=C_LOAD, opacity=0.2, row=1, col=1)
            fig.add_annotation(x=(x_s+x_e)/2, y=0.35, showarrow=False, text=f"<b>w={l['w']}</b>", font=dict(color=C_LOAD), row=1, col=1)

    # -- ROW 2: Shear --
    fig.add_trace(go.Scatter(x=df['x'], y=df['shear'], mode='lines', line=dict(color=C_SHEAR_LINE, width=2), fill='tozeroy', fillcolor=C_SHEAR_FILL, name='Shear'), row=2, col=1)
    # Max Shear Label
    if not df['shear'].empty:
        max_v = df['shear'].abs().max()
        if max_v > 0.01:
            row_v = df.loc[df['shear'].abs() == max_v].iloc[0]
            fig.add_annotation(x=row_v['x'], y=row_v['shear'], text=f"<b>{row_v['shear']:.2f}</b>", showarrow=True, arrowhead=1, row=2, col=1, font=dict(color=C_SHEAR_LINE))

    # -- ROW 3: Moment --
    fig.add_trace(go.Scatter(x=df['x'], y=df['moment'], mode='lines', line=dict(color=C_MOMENT_LINE, width=2), fill='tozeroy', fillcolor=C_MOMENT_FILL, name='Moment'), row=3, col=1)
    # Max/Min Moment Labels
    if not df['moment'].empty:
        max_m = df['moment'].max()
        min_m = df['moment'].min()
        if max_m > 0.01:
            xm = df.loc[df['moment'] == max_m, 'x'].iloc[0]
            fig.add_annotation(x=xm, y=max_m, text=f"<b>{max_m:.2f}</b>", showarrow=True, arrowhead=1, row=3, col=1, font=dict(color=C_MOMENT_LINE))
        if min_m < -0.01:
            xm = df.loc[df['moment'] == min_m, 'x'].iloc[0]
            fig.add_annotation(x=xm, y=min_m, text=f"<b>{min_m:.2f}</b>", showarrow=True, arrowhead=1, row=3, col=1, ay=30, font=dict(color=C_MOMENT_LINE))

    # Layout
    fig.update_layout(
        height=600, margin=dict(l=10, r=10, t=30, b=10), 
        paper_bgcolor='white', plot_bgcolor='white', showlegend=False,
        hovermode="x unified", hoverlabel=dict(bgcolor="white")
    )
    fig.update_xaxes(showgrid=True, gridcolor=C_GRID)
    fig.update_yaxes(showgrid=True, gridcolor=C_GRID, zeroline=True, zerolinecolor='black')
    fig.update_yaxes(visible=False, row=1, col=1)

    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

# --- 2. RC DESIGN LOGIC (Restored & Fixed) ---
def get_bar_area(dia_mm):
    return (math.pi * (dia_mm/10)**2) / 4

def render_design_sheet(df, spans, params):
    st.subheader("🏗️ Reinforced Concrete Design (ACI 318)")
    
    # 2.1 Design Inputs
    st.info("Define section properties for each span:")
    n_spans = len(spans)
    span_props = []
    
    cols = st.columns(n_spans)
    for i in range(n_spans):
        with cols[i]:
            st.markdown(f"**Span {i+1}**")
            b = st.number_input(f"b (cm)", value=30.0, key=f"b_{i}")
            h = st.number_input(f"h (cm)", value=60.0, key=f"h_{i}")
            # Fix: Renamed variable 'st' to 'stirrup_val' to avoid conflict
            main_bar = st.selectbox("Main Reinforcement", [12,16,20,25,28], index=1, key=f"mb_{i}", format_func=lambda x: f"DB{x}")
            stirrup_val = st.selectbox("Stirrup", [6,9], index=0, key=f"sb_{i}", format_func=lambda x: f"RB{x}")
            span_props.append({"b":b, "h":h, "cv":3.5, "main":main_bar, "stirrup":stirrup_val})

    # 2.2 Calculation Logic
    results = []
    cum_dist = 0.0
    fc, fy = params['fc'], params['fy']
    phi_b, phi_v = 0.90, 0.75
    
    for i, L in enumerate(spans):
        prop = span_props[i]
        b, h = prop['b'], prop['h']
        db, ds = prop['main'], prop['stirrup']
        d = h - prop['cv'] - (ds/10) - (db/20) # Effective depth
        
        # Get Forces for this span
        mask = (df['x'] >= cum_dist) & (df['x'] <= cum_dist + L)
        span_data = df[mask]
        Mu_pos = max(0, span_data['moment'].max()) * 100 # kg-cm
        Vu_max = span_data['shear'].abs().max() # kg
        
        # Flexure Design
        Ab = get_bar_area(db)
        As_req_approx = Mu_pos / (phi_b * fy * 0.9 * d) if Mu_pos > 0 else 0
        n_bars = max(2, math.ceil(As_req_approx / Ab)) if As_req_approx > 0 else 2
        As_prov = n_bars * Ab
        
        a_depth = (As_prov * fy) / (0.85 * fc * b)
        Mn = As_prov * fy * (d - a_depth/2)
        phi_Mn = phi_b * Mn
        dc_flex = Mu_pos / phi_Mn if phi_Mn > 0 else 0
        status_m = "✅ OK" if dc_flex <= 1.0 else "❌ Fail"

        # Shear Design
        Vc = 0.53 * math.sqrt(fc) * b * d
        phi_Vc = phi_v * Vc
        
        if Vu_max > phi_Vc / 2:
            Av = 2 * get_bar_area(ds)
            s_req = (Av * fy * d) / ((Vu_max/phi_v) - Vc) if Vu_max > phi_Vc else 999
            s_max = min(d/2, 60)
            s_prov = max(5, int(min(s_req, s_max)))
            stir_txt = f"RB{ds}@{s_prov}"
            
            Vs = (Av * fy * d) / s_prov
            phi_Vn = phi_v * (Vc + Vs)
            dc_v = Vu_max / phi_Vn
        else:
            stir_txt = "Min"
            dc_v = Vu_max / phi_Vc
        
        results.append({
            "Span": f"{i+1}",
            "Section": f"{b:.0f}x{h:.0f}",
            "Rebar": f"{n_bars}-DB{db}",
            "Mu+ (kg-m)": f"{Mu_pos/100:.0f}",
            "φMn (kg-m)": f"{phi_Mn/100:.0f}",
            "Status (M)": status_m,
            "Vu (kg)": f"{Vu_max:.0f}",
            "Stirrups": stir_txt,
            "D/C (V)": f"{dc_v:.2f}"
        })
        cum_dist += L
        
    # 2.3 Render Table
    st.markdown("---")
    st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)
