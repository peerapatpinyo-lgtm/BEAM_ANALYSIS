import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import rc_design 

def draw_diagrams(df, spans, supports, loads, u_force, u_len):
    cum_len = [0] + list(np.cumsum(spans))
    L_total = cum_len[-1]
    
    # 1. Aggregate Point Loads
    display_loads = []
    for l in loads:
        if l['type'] == 'U': display_loads.append(l)
    p_map = {}
    for l in loads:
        if l['type'] == 'P':
            key = (l['span_idx'], l['x'])
            p_map[key] = p_map.get(key, 0) + l['P']
    for (s_idx, x_val), total_p in p_map.items():
        if total_p != 0:
            display_loads.append({'span_idx': s_idx, 'type': 'P', 'P': total_p, 'x': x_val})

    # Scale calculation
    val_list = [abs(l['w']) for l in display_loads if l['type']=='U'] + [abs(l['P']) for l in display_loads if l['type']=='P']
    max_load_val = max(val_list) if val_list else 100
    
    viz_h = max_load_val * 1.5 
    sup_h = viz_h * 0.20 # เพิ่มขนาด Support
    sup_w = L_total * 0.025

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=(f"<b>Loading Diagram</b>", 
                                        f"<b>Shear Force Diagram (SFD)</b>", 
                                        f"<b>Bending Moment Diagram (BMD)</b>"),
                        row_heights=[0.3, 0.35, 0.35])
    
    # --- LOAD DIAGRAM ---
    fig.add_trace(go.Scatter(x=[0, L_total], y=[0, 0], mode='lines', 
                             line=dict(color='black', width=4), hoverinfo='none'), row=1, col=1)

    # --- PROFESSIONAL SUPPORTS (SVG PATHS / SHAPES) ---
    for i, x in enumerate(cum_len):
        stype = supports.iloc[i]['type']
        if stype == "Pin":
            # Triangle with Hatch Base
            fig.add_trace(go.Scatter(x=[x, x-sup_w, x+sup_w, x], y=[0, -sup_h, -sup_h, 0], 
                                     fill="toself", fillcolor="#CFD8DC", line_color="black", mode='lines', hoverinfo='text', text="Pin"), row=1, col=1)
            # Hatch lines
            for hx in np.linspace(x-sup_w, x+sup_w, 5):
                fig.add_trace(go.Scatter(x=[hx, hx-sup_w*0.5], y=[-sup_h, -sup_h*1.3], mode="lines", line=dict(color="black", width=1), hoverinfo='skip'), row=1, col=1)
                
        elif stype == "Roller":
            # Circle
            fig.add_trace(go.Scatter(x=[x], y=[-sup_h/2], mode="markers", 
                                     marker=dict(size=12, color="white", line=dict(color="black", width=2)), 
                                     hoverinfo='text', text="Roller"), row=1, col=1)
            # Base Line
            fig.add_trace(go.Scatter(x=[x-sup_w, x+sup_w], y=[-sup_h, -sup_h], mode="lines", line=dict(color="black", width=2), hoverinfo='skip'), row=1, col=1)
            # Hatch lines
            for hx in np.linspace(x-sup_w, x+sup_w, 5):
                fig.add_trace(go.Scatter(x=[hx, hx-sup_w*0.5], y=[-sup_h, -sup_h*1.3], mode="lines", line=dict(color="black", width=1), hoverinfo='skip'), row=1, col=1)

        elif stype == "Fixed":
            # Vertical Line
            fig.add_shape(type="line", x0=x, y0=-sup_h, x1=x, y1=sup_h, line=dict(color="black", width=4), row=1, col=1)
            # Hatch Lines (Diagonal)
            dir_sign = -1 if x==0 else 1
            for hy in np.linspace(-sup_h, sup_h, 8):
                fig.add_shape(type="line", x0=x, y0=hy, x1=x + (sup_w*0.6*dir_sign), y1=hy - (sup_h*0.15), line=dict(color="black", width=1), row=1, col=1)

    # Loads
    for l in display_loads:
        if l['type'] == 'U':
            x1, x2 = cum_len[l['span_idx']], cum_len[l['span_idx']+1]
            h = (abs(l['w'])/max_load_val) * (viz_h * 0.6)
            fig.add_trace(go.Scatter(x=[x1, x2, x2, x1], y=[0, 0, h, h], fill="toself", fillcolor="rgba(33, 150, 243, 0.3)", line_width=0, hoverinfo='skip'), row=1, col=1)
            fig.add_trace(go.Scatter(x=[x1, x2], y=[h, h], mode='lines', line=dict(color="#1976D2", width=2), hoverinfo='skip'), row=1, col=1)
            fig.add_annotation(x=(x1+x2)/2, y=h, text=f"<b>Wu={l['w']:.0f}</b>", showarrow=False, yshift=15, font=dict(color="#0D47A1", size=11), row=1, col=1)
            
        elif l['type'] == 'P':
            px = cum_len[l['span_idx']] + l['x']
            h = (abs(l['P'])/max_load_val) * (viz_h * 0.6)
            if h == 0: h = viz_h * 0.2
            fig.add_annotation(x=px, y=0, ax=0, ay=-40, text=f"<b>Pu={l['P']:.0f}</b>", arrowhead=2, arrowsize=1.5, arrowwidth=2, arrowcolor="#D32F2F", font=dict(color="#D32F2F", size=11), row=1, col=1)

    fig.update_yaxes(visible=False, range=[-sup_h*1.5, viz_h*1.4], row=1, col=1)
    
    # --- SFD & BMD (Interactive Mode Restored) ---
    # Shear
    fig.add_trace(go.Scatter(x=df['x'], y=df['shear'], fill='tozeroy', line=dict(color='#D32F2F', width=2), 
                             fillcolor='rgba(211, 47, 47, 0.1)', name="Shear", 
                             hovertemplate='x: %{x:.2f}<br>V: %{y:.2f}'), row=2, col=1)
    # Moment
    fig.add_trace(go.Scatter(x=df['x'], y=df['moment'], fill='tozeroy', line=dict(color='#1565C0', width=2), 
                             fillcolor='rgba(21, 101, 192, 0.1)', name="Moment",
                             hovertemplate='x: %{x:.2f}<br>M: %{y:.2f}'), row=3, col=1)

    # --- Min/Max Labels ---
    for col, row, color, unit in [('shear', 2, '#D32F2F', u_force), ('moment', 3, '#1565C0', f"{u_force}-{u_len}")]:
        arr = df[col].to_numpy()
        mx, mn = np.max(arr), np.min(arr)
        imx, imn = np.argmax(arr), np.argmin(arr)
        
        for val, idx, pos in [(mx, imx, "top"), (mn, imn, "bottom")]:
            if abs(val) > 1e-3:
                ys = 25 if pos=="top" else -25
                x_pos = df['x'].iloc[idx]
                label_txt = f"<b>{val:,.2f}</b><br><span style='font-size:10px'>@ {x_pos:.2f}m</span>"
                fig.add_annotation(x=x_pos, y=val, text=label_txt, showarrow=False, bgcolor="rgba(255,255,255,0.85)", bordercolor=color, borderwidth=1, font=dict(color=color, size=11), yshift=ys, row=row, col=1)

        rng = mx - mn
        pad = (rng if rng!=0 else 10) * 0.3
        fig.update_yaxes(title_text=f"{col.capitalize()} ({unit})", range=[mn-pad, mx+pad], row=row, col=1)

    # --- GLOBAL INTERACTIVITY (Spikes & Hover) ---
    fig.update_xaxes(showgrid=True, gridcolor='#ECEFF1', showspikes=True, spikemode='across', spikethickness=1, spikedash='dash', spikecolor='#90A4AE', row=3, col=1)
    fig.update_xaxes(showspikes=True, spikemode='across', spikethickness=1, spikedash='dash', spikecolor='#90A4AE', row=2, col=1)
    fig.update_layout(height=900, template="plotly_white", margin=dict(t=30, b=30, l=60, r=30), showlegend=False, hovermode="x unified")
    
    st.plotly_chart(fig, use_container_width=True)

# --- รูป Cross Section ---
def render_cross_section(b, h, cover, top_bars, bot_bars):
    fig = go.Figure()
    fig.add_shape(type="rect", x0=0, y0=0, x1=b, y1=h, line=dict(color="black", width=2), fillcolor="#E0E0E0")
    fig.add_shape(type="rect", x0=cover, y0=cover, x1=b-cover, y1=h-cover, line=dict(color="#4CAF50", width=2, dash="dash"))
    
    def draw_bars(bar_str, y_pos):
        parsed = rc_design.parse_bars(bar_str)
        if parsed:
            num, db = parsed
            eff_width = b - 2*cover
            if num > 1:
                spacing = eff_width / (num - 1)
                xs = [cover + i*spacing for i in range(num)]
            else:
                xs = [b/2]
            for x in xs:
                r = (db/10)/2 
                fig.add_shape(type="circle", x0=x-r, y0=y_pos-r, x1=x+r, y1=y_pos+r, fillcolor="#D32F2F", line_color="black")
    
    draw_bars(bot_bars, cover + 1.5)
    draw_bars(top_bars, h - cover - 1.5)
    fig.update_xaxes(visible=False, range=[-5, b+5])
    fig.update_yaxes(visible=False, range=[-5, h+5], scaleanchor="x", scaleratio=1)
    fig.update_layout(width=150, height=200, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', showlegend=False)
    return fig

# --- REAL ENGINEERING LONGITUDINAL VIEW ---
def render_longitudinal_view(spans, supports, res_pos, res_neg, stirrup_txt):
    total_len = sum(spans)
    cum_len = [0] + list(np.cumsum(spans))
    
    fig = go.Figure()
    
    # 1. Concrete Beam Outline
    fig.add_shape(type="rect", x0=0, y0=0, x1=total_len, y1=10, 
                  line=dict(color="black", width=2), fillcolor="#F5F5F5", layer="below")
    
    # 2. Supports (Simplified Triangles)
    for x in cum_len:
        fig.add_trace(go.Scatter(x=[x, x-0.15, x+0.15, x], y=[0, -1.5, -1.5, 0], 
                                 fill="toself", fillcolor="#90A4AE", line_color="black", showlegend=False, hoverinfo='skip'))

    # 3. Main Reinforcement (Engineering Detailing)
    
    # Bottom Bars (Continuous / Splice not shown, assume full run for visual)
    if res_pos['Bars']:
        fig.add_trace(go.Scatter(x=[0, total_len], y=[1.5, 1.5], mode="lines",
                                 line=dict(color="#1565C0", width=4), name="Bottom Bar", hovertemplate=f"Bottom: {res_pos['Bars']}"))
        # Label Midspan
        for i, span_L in enumerate(spans):
            mid_x = cum_len[i] + span_L/2
            fig.add_annotation(x=mid_x, y=1.5, text=f"<b>{res_pos['Bars']}</b>", yshift=-15, showarrow=False, font=dict(color="#1565C0", size=10))

    # Top Bars (Cutoff @ L/3 or L/4)
    if res_neg['Bars']:
        for i, span_L in enumerate(spans):
            # Left Support of span
            x_start = cum_len[i]
            cutoff_len = span_L / 3.0 # Engineering rule of thumb
            
            # Draw from left support inward
            fig.add_trace(go.Scatter(x=[x_start, x_start + cutoff_len], y=[8.5, 8.5], mode="lines",
                                     line=dict(color="#D32F2F", width=4), showlegend=False, hovertemplate=f"Top: {res_neg['Bars']}"))
            
            # Draw from right support inward
            x_end = cum_len[i+1]
            fig.add_trace(go.Scatter(x=[x_end - cutoff_len, x_end], y=[8.5, 8.5], mode="lines",
                                     line=dict(color="#D32F2F", width=4), showlegend=False, hovertemplate=f"Top: {res_neg['Bars']}"))
            
            # Label
            if i==0: # Label once at first support
                fig.add_annotation(x=0, y=8.5, text=f"<b>{res_neg['Bars']}</b>", yshift=15, showarrow=False, font=dict(color="#D32F2F", size=10))

    # 4. Stirrups (Real Spacing Visual)
    # Parse spacing from text e.g., "RB6@20cm"
    s_val = 0.20 # default 20cm
    if "@" in stirrup_txt:
        try:
            val_str = stirrup_txt.split("@")[1].replace("cm","").replace("mm","")
            if "Min" in val_str: s_val = 0.25
            else: s_val = float(val_str)/100 if "cm" in stirrup_txt else float(val_str)/1000
        except: s_val = 0.25
    
    if s_val > 0 and "None" not in stirrup_txt:
        stirrup_xs = []
        curr_x = 0.05
        while curr_x < total_len:
            # Check if near support (skip support width approx)
            stirrup_xs.append(curr_x)
            curr_x += s_val
            
        # Draw all stirrups as a single trace for performance
        x_pts = []
        y_pts = []
        for sx in stirrup_xs:
            x_pts.extend([sx, sx, None])
            y_pts.extend([1.5, 8.5, None])
            
        fig.add_trace(go.Scatter(x=x_pts, y=y_pts, mode="lines", 
                                 line=dict(color="green", width=1), name="Stirrup", hoverinfo='skip'))
        
        # Label Stirrup
        fig.add_annotation(x=total_len/2, y=5, text=f"<b>Stirrup: {stirrup_txt}</b>", bgcolor="white", bordercolor="green", showarrow=False)

    fig.update_xaxes(showgrid=False, visible=False)
    fig.update_yaxes(visible=False, range=[-2, 12])
    fig.update_layout(height=250, title="<b>Longitudinal Reinforcement Detail</b>",
                      margin=dict(l=20, r=20, t=40, b=10), template="plotly_white", showlegend=False)
    
    return fig

def render_design_results(df, params, spans, supports):
    st.markdown('<div class="section-header">4️⃣ Reinforced Concrete Design</div>', unsafe_allow_html=True)
    
    m_max = df['moment'].max()
    m_min = df['moment'].min()
    v_max = df['shear'].abs().max()
    
    # --- SHEAR CALCULATION ---
    vu_val, phi_vc, stirrup_txt = rc_design.calculate_shear_capacity(v_max, params)
    
    if "SDM" in params['method']:
        res_pos = rc_design.calculate_flexure_sdm(m_max, "Mid-Span (+M)", params)
        res_neg = rc_design.calculate_flexure_sdm(m_min, "Support (-M)", params)
        
        # --- Cards ---
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"**Top Bars (-M)**")
            st.info(f"{res_neg['Bars']}")
            st.caption(f"Mu: {res_neg['Mu']:,.2f} | As: {res_neg['As_req']:.2f}")
            if "Over" not in res_neg['Status']:
                st.plotly_chart(render_cross_section(params['b'], params['h'], params['cv'], res_neg['Bars'], ""), use_container_width=True)

        with c2:
            st.markdown(f"**Bottom Bars (+M)**")
            st.info(f"{res_pos['Bars']}")
            st.caption(f"Mu: {res_pos['Mu']:,.2f} | As: {res_pos['As_req']:.2f}")
            if "Over" not in res_pos['Status']:
                st.plotly_chart(render_cross_section(params['b'], params['h'], params['cv'], "", res_pos['Bars']), use_container_width=True)
                
        with c3:
            st.markdown(f"**Shear (Stirrups)**")
            st.success(f"{stirrup_txt}")
            st.caption(f"Vu: {vu_val:,.2f} | ϕVc: {phi_vc:,.2f}")
            # Graphic for stirrup info (text mostly)

        # --- Longitudinal View (Updated) ---
        st.markdown("---")
        st.plotly_chart(render_longitudinal_view(spans, supports, res_pos, res_neg, stirrup_txt), use_container_width=True)

    else:
        st.info("WSD Method not implemented in this version.")
