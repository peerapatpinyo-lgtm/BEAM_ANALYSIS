import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import rc_design 

# ==========================================
# 1. PLOT ANALYSIS DIAGRAMS
# ==========================================
def draw_diagrams(df, spans, supports, loads, u_force, u_len):
    cum_len = [0] + list(np.cumsum(spans))
    L_total = cum_len[-1]
    
    # --- Prepare Data ---
    display_loads = [l for l in loads if l['type'] == 'U']
    p_map = {}
    for l in loads:
        if l['type'] == 'P':
            key = (l['span_idx'], l['x'])
            p_map[key] = p_map.get(key, 0) + l['P']
    for (s_idx, x_val), total_p in p_map.items():
        if total_p != 0:
            display_loads.append({'span_idx': s_idx, 'type': 'P', 'P': total_p, 'x': x_val})

    # Scale Helpers
    val_list = [abs(l['w']) for l in display_loads if l['type']=='U'] + [abs(l['P']) for l in display_loads if l['type']=='P']
    max_load = max(val_list) if val_list else 100
    viz_h = max_load * 1.5 
    
    # Support Visual Size (Scaled relative to Length but clamped)
    sup_sz = max(L_total * 0.02, 0.3) 

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=(f"<b>Load Model</b>", f"<b>Shear Force (SFD)</b>", f"<b>Bending Moment (BMD)</b>"),
                        row_heights=[0.25, 0.375, 0.375])
    
    # --- ROW 1: LOADS & SUPPORTS ---
    fig.add_trace(go.Scatter(x=[0, L_total], y=[0, 0], mode='lines', line=dict(color='black', width=3), hoverinfo='none'), row=1, col=1)

    # *** NEW: Engineering Supports (SVG Paths) ***
    for i, x in enumerate(cum_len):
        stype = supports.iloc[i]['type']
        
        if stype == "Pin":
            # Triangle
            path = f"M {x} 0 L {x-sup_sz} {-sup_sz} L {x+sup_sz} {-sup_sz} Z"
            fig.add_shape(type="path", path=path, fillcolor="#B0BEC5", line=dict(color="black"), row=1, col=1)
            # Base Hatch
            fig.add_shape(type="line", x0=x-sup_sz*1.2, y0=-sup_sz*1.2, x1=x+sup_sz*1.2, y1=-sup_sz*1.2, line=dict(color="black", width=2), row=1, col=1)
            
        elif stype == "Roller":
            # Circle
            fig.add_shape(type="circle", x0=x-sup_sz, y0=-sup_sz, x1=x+sup_sz, y1=0, fillcolor="white", line=dict(color="black"), row=1, col=1)
            # Base
            fig.add_shape(type="line", x0=x-sup_sz*1.2, y0=-sup_sz, x1=x+sup_sz*1.2, y1=-sup_sz, line=dict(color="black", width=2), row=1, col=1)

        elif stype == "Fixed":
            # Vertical Wall
            fig.add_shape(type="line", x0=x, y0=-sup_sz*1.5, x1=x, y1=sup_sz*1.5, line=dict(color="black", width=4), row=1, col=1)
            # Hatch lines
            sign = -1 if x == 0 else 1
            for hy in np.linspace(-sup_sz*1.5, sup_sz*1.5, 6):
                fig.add_shape(type="line", x0=x, y0=hy, x1=x + (sup_sz*0.5*sign), y1=hy-(sup_sz*0.2), line=dict(color="black", width=1), row=1, col=1)

    # Loads Drawing
    for l in display_loads:
        if l['type'] == 'U':
            x1, x2 = cum_len[l['span_idx']], cum_len[l['span_idx']+1]
            h = (abs(l['w'])/max_load) * (viz_h * 0.6)
            fig.add_trace(go.Scatter(x=[x1, x2, x2, x1], y=[0, 0, h, h], fill="toself", fillcolor="rgba(33, 150, 243, 0.3)", line_width=0, hoverinfo='skip'), row=1, col=1)
            fig.add_trace(go.Scatter(x=[x1, x2], y=[h, h], mode='lines', line=dict(color="#1976D2", width=2), hoverinfo='skip'), row=1, col=1)
            fig.add_annotation(x=(x1+x2)/2, y=h, text=f"<b>Wu={l['w']:.0f}</b>", showarrow=False, yshift=10, font=dict(color="#0D47A1", size=10), row=1, col=1)
        elif l['type'] == 'P':
            px = cum_len[l['span_idx']] + l['x']
            h = (abs(l['P'])/max_load) * (viz_h * 0.6)
            if h==0: h = viz_h*0.2
            fig.add_annotation(x=px, y=0, ax=0, ay=-40, text=f"<b>Pu={l['P']:.0f}</b>", arrowhead=2, arrowcolor="#C62828", font=dict(color="#C62828", size=10), row=1, col=1)
            
    fig.update_yaxes(visible=False, range=[-sup_sz*2.5, viz_h*1.2], row=1, col=1)

    # --- ROW 2 & 3: SFD & BMD ---
    # Shear
    fig.add_trace(go.Scatter(x=df['x'], y=df['shear'], fill='tozeroy', line=dict(color='#D32F2F', width=2), 
                             name="Shear", hovertemplate='x: %{x:.2f} m<br>V: %{y:,.2f}'), row=2, col=1)
    # Moment
    fig.add_trace(go.Scatter(x=df['x'], y=df['moment'], fill='tozeroy', line=dict(color='#1565C0', width=2), 
                             name="Moment", hovertemplate='x: %{x:.2f} m<br>M: %{y:,.2f}'), row=3, col=1)

    # Min/Max Labels
    for col, row, color, unit in [('shear', 2, '#D32F2F', u_force), ('moment', 3, '#1565C0', f"{u_force}-{u_len}")]:
        arr = df[col].to_numpy()
        mx, mn = np.max(arr), np.min(arr)
        # Add labels
        for val, idx in [(mx, np.argmax(arr)), (mn, np.argmin(arr))]:
            if abs(val) > 0.01:
                fig.add_annotation(x=df['x'].iloc[idx], y=val, text=f"<b>{val:,.2f}</b>", showarrow=False, 
                                   bgcolor="rgba(255,255,255,0.9)", bordercolor=color, borderwidth=1, yshift=15 if val>0 else -15, row=row, col=1)
        
        pad = (mx - mn) * 0.2 if (mx-mn) > 0 else 1.0
        fig.update_yaxes(title_text=f"{col.capitalize()} ({unit})", range=[mn-pad, mx+pad], row=row, col=1)

    # *** FIX: SPIKE LINES (Vertical Dashed Lines) ***
    # ต้องเปิด showspikes=True และตั้ง hovermode='x unified'
    for r in [2, 3]:
        fig.update_xaxes(showspikes=True, spikemode='across', spikethickness=1, spikedash='dash', spikecolor='#546E7A', row=r, col=1)
        fig.update_yaxes(showspikes=True, spikemode='across', spikethickness=1, spikedash='dash', spikecolor='#546E7A', row=r, col=1)

    fig.update_layout(height=800, template="plotly_white", margin=dict(t=40, b=40, l=60, r=40), 
                      hovermode="x unified", showlegend=False) # Important for vertical line correlation
    
    st.plotly_chart(fig, use_container_width=True)

# ==========================================
# 2. CROSS SECTION RENDERER
# ==========================================
def render_cross_section(b, h, cover, bars, label=""):
    fig = go.Figure()
    # Concrete
    fig.add_shape(type="rect", x0=0, y0=0, x1=b, y1=h, line=dict(color="black", width=2), fillcolor="#ECEFF1")
    # Stirrup
    fig.add_shape(type="rect", x0=cover, y0=cover, x1=b-cover, y1=h-cover, line=dict(color="#43A047", width=2, dash="dash"))
    
    parsed = rc_design.parse_bars(bars)
    if parsed:
        num, db = parsed
        eff_w = b - 2*cover
        xs = np.linspace(cover, b-cover, num) if num > 1 else [b/2]
        
        y_pos = cover + 1.5 if "Bot" in label else h - cover - 1.5
        color = "#1565C0" if "Bot" in label else "#D32F2F"
        
        for x in xs:
            r = (db/10)/2 if db>0 else 0.5
            fig.add_shape(type="circle", x0=x-r, y0=y_pos-r, x1=x+r, y1=y_pos+r, fillcolor=color, line_color="black")

    fig.update_xaxes(visible=False, range=[-2, b+2])
    fig.update_yaxes(visible=False, range=[-2, h+2], scaleanchor="x", scaleratio=1)
    fig.update_layout(width=160, height=220, margin=dict(l=5, r=5, t=30, b=5), 
                      title=dict(text=label, y=0.95, x=0.5, font=dict(size=12)),
                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', showlegend=False)
    return fig

# ==========================================
# 3. LONGITUDINAL PROFILE (ENGINEERED)
# ==========================================
def render_longitudinal_view(spans, supports, design_data):
    total_len = sum(spans)
    cum_len = [0] + list(np.cumsum(spans))
    
    fig = go.Figure()
    
    # 1. Beam Body
    fig.add_shape(type="rect", x0=0, y0=0, x1=total_len, y1=10, line=dict(color="black", width=2), fillcolor="#FAFAFA", layer="below")
    
    # 2. Supports (Use same style as main graph roughly)
    for i, x in enumerate(cum_len):
        stype = supports.iloc[i]['type']
        if stype != "None":
            # Simplified Schematic Support
            fig.add_shape(type="path", path=f"M {x} 0 L {x-0.2} -1.5 L {x+0.2} -1.5 Z", fillcolor="#B0BEC5", line_color="black")
    
    # 3. Rebar Detailing
    for i, span_len in enumerate(spans):
        x_start = cum_len[i]
        x_end = cum_len[i+1]
        data = design_data[i]
        
        # Bottom Bar (Blue): Runs almost full span, avoids support slightly
        if data['bot_bars']:
            fig.add_trace(go.Scatter(x=[x_start+0.1, x_end-0.1], y=[1.5, 1.5], mode="lines", 
                                     line=dict(color="#1565C0", width=4), name="Bot", hoverinfo='text', text=f"Bot: {data['bot_bars']}"))
            fig.add_annotation(x=x_start+span_len/2, y=1.5, text=data['bot_bars'], yshift=-15, showarrow=False, font=dict(color="#1565C0", size=10))

        # Top Bar (Red): At Supports (L/3 approx)
        if data['top_bars']:
            cut = span_len/3
            # Left Hogging
            fig.add_trace(go.Scatter(x=[x_start, x_start+cut], y=[8.5, 8.5], mode="lines", 
                                     line=dict(color="#C62828", width=4), name="Top", hoverinfo='text', text=f"Top: {data['top_bars']}"))
            # Right Hogging
            fig.add_trace(go.Scatter(x=[x_end-cut, x_end], y=[8.5, 8.5], mode="lines", 
                                     line=dict(color="#C62828", width=4), showlegend=False, hoverinfo='text', text=f"Top: {data['top_bars']}"))
            
        # Stirrups (Green)
        s_val = 0.25
        if "@" in data['stirrups']:
            try:
                s_txt = data['stirrups'].split("@")[1].replace("cm","")
                s_val = float(s_txt)/100
            except: pass
            
        if "None" not in data['stirrups']:
            # Draw vertical lines
            sx = np.arange(x_start+0.2, x_end-0.2, s_val)
            x_pts, y_pts = [], []
            for x in sx:
                x_pts.extend([x, x, None])
                y_pts.extend([1.5, 8.5, None])
            fig.add_trace(go.Scatter(x=x_pts, y=y_pts, mode="lines", line=dict(color="#43A047", width=1), hoverinfo='skip'))
            fig.add_annotation(x=x_start+span_len/2, y=5.0, text=f"<b>{data['stirrups']}</b>", bgcolor="white", bordercolor="#43A047", showarrow=False, font=dict(size=10))

    fig.update_xaxes(showgrid=False, visible=False)
    fig.update_yaxes(visible=False, range=[-2, 12])
    fig.update_layout(height=250, title="<b>Longitudinal Reinforcement Profile (Side View)</b>", margin=dict(l=20, r=20, t=40, b=10), showlegend=False)
    return fig

# ==========================================
# 4. MAIN RENDERER
# ==========================================
def render_design_results(df, params, spans, supports):
    st.markdown('<div class="section-header">4️⃣ Structural Design Results</div>', unsafe_allow_html=True)
    
    cum_len = [0] + list(np.cumsum(spans))
    span_data = []

    # Create Tabs for Spans
    tabs = st.tabs([f"Span {i+1}" for i in range(len(spans))])

    for i, tab in enumerate(tabs):
        # Filter forces for this span
        mask = (df['x'] >= cum_len[i]) & (df['x'] <= cum_len[i+1])
        sub_df = df[mask]
        
        if sub_df.empty: continue
        
        # Design Values
        m_pos = max(0, sub_df['moment'].max())
        m_neg = min(0, sub_df['moment'].min())
        v_u = sub_df['shear'].abs().max()
        
        # Perform Design
        des_pos = rc_design.calculate_flexure_sdm(m_pos, "Midspan (+)", params)
        des_neg = rc_design.calculate_flexure_sdm(m_neg, "Support (-)", params)
        v_act, v_cap, stir_txt = rc_design.calculate_shear_capacity(v_u, params)
        
        span_data.append({'bot_bars': des_pos['Bars'], 'top_bars': des_neg['Bars'], 'stirrups': stir_txt})

        with tab:
            # Layout: 3 Columns (Top, Bot, Shear)
            c1, c2, c3 = st.columns(3)
            
            # 1. Top Steel
            with c1:
                st.markdown(f"##### Top Support (-M)")
                st.markdown(f"**{des_neg['Bars']}**")
                st.caption(f"$M_u^- = {abs(des_neg['Mu']):.2f}$ | $A_{{s,req}} = {des_neg['As_req']:.2f}$")
                if des_neg['Bars'] and "Over" not in des_neg['Status']:
                    st.plotly_chart(render_cross_section(params['b'], params['h'], params['cv'], des_neg['Bars'], "Top Section"), use_container_width=True)
                else: st.warning(des_neg['Status'])
            
            # 2. Bottom Steel
            with c2:
                st.markdown(f"##### Bottom Midspan (+M)")
                st.markdown(f"**{des_pos['Bars']}**")
                st.caption(f"$M_u^+ = {des_pos['Mu']:.2f}$ | $A_{{s,req}} = {des_pos['As_req']:.2f}$")
                if des_pos['Bars'] and "Over" not in des_pos['Status']:
                    st.plotly_chart(render_cross_section(params['b'], params['h'], params['cv'], des_pos['Bars'], "Bot Section (Bot)"), use_container_width=True)
                else: st.warning(des_pos['Status'])

            # 3. Shear
            with c3:
                st.markdown(f"##### Shear (Stirrups)")
                st.markdown(f"**{stir_txt}**")
                st.write(f"$V_u = {v_act:,.2f}$")
                st.write(f"$\phi V_c = {v_cap:,.2f}$")
                check = "OK" if v_act <= v_cap else "NOT OK"
                color = "green" if check=="OK" else "red"
                st.markdown(f"Status: <span style='color:{color}; font-weight:bold'>{check}</span>", unsafe_allow_html=True)

    st.markdown("---")
    # Draw Longitudinal Profile using collected data
    st.plotly_chart(render_longitudinal_view(spans, supports, span_data), use_container_width=True)
