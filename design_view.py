import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import rc_design 

# ==========================================
# 1. PLOT ANALYSIS DIAGRAMS (SFD/BMD)
# ==========================================
def draw_diagrams(df, spans, supports, loads, u_force, u_len):
    cum_len = [0] + list(np.cumsum(spans))
    L_total = cum_len[-1]
    
    # --- PREPARE LOADS VISUALIZATION ---
    display_loads = []
    # Uniform Loads
    for l in loads:
        if l['type'] == 'U': display_loads.append(l)
    # Aggregated Point Loads
    p_map = {}
    for l in loads:
        if l['type'] == 'P':
            key = (l['span_idx'], l['x'])
            p_map[key] = p_map.get(key, 0) + l['P']
    for (s_idx, x_val), total_p in p_map.items():
        if total_p != 0:
            display_loads.append({'span_idx': s_idx, 'type': 'P', 'P': total_p, 'x': x_val})

    # Auto Scale Calculation
    val_list = [abs(l['w']) for l in display_loads if l['type']=='U'] + [abs(l['P']) for l in display_loads if l['type']=='P']
    max_load_val = max(val_list) if val_list else 100
    
    viz_h = max_load_val * 1.5 
    sup_h = viz_h * 0.20
    sup_w = L_total * 0.025

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=(f"<b>Loading Diagram</b>", 
                                        f"<b>Shear Force Diagram (SFD)</b>", 
                                        f"<b>Bending Moment Diagram (BMD)</b>"),
                        row_heights=[0.3, 0.35, 0.35])
    
    # --- ROW 1: LOADING & SUPPORTS ---
    fig.add_trace(go.Scatter(x=[0, L_total], y=[0, 0], mode='lines', 
                             line=dict(color='black', width=3), hoverinfo='none'), row=1, col=1)

    # Engineering Standard Supports
    for i, x in enumerate(cum_len):
        stype = supports.iloc[i]['type']
        if stype == "Pin":
            # Triangle Body
            fig.add_trace(go.Scatter(x=[x, x-sup_w, x+sup_w, x], y=[0, -sup_h, -sup_h, 0], 
                                     fill="toself", fillcolor="#CFD8DC", line_color="black", mode='lines', hoverinfo='text', text="Pin"), row=1, col=1)
            # Hatching Base
            for hx in np.linspace(x-sup_w, x+sup_w, 5):
                fig.add_trace(go.Scatter(x=[hx, hx-sup_w*0.5], y=[-sup_h, -sup_h*1.3], mode="lines", line=dict(color="black", width=1), hoverinfo='skip'), row=1, col=1)
                
        elif stype == "Roller":
            # Circle
            fig.add_trace(go.Scatter(x=[x], y=[-sup_h/2], mode="markers", 
                                     marker=dict(size=12, color="white", line=dict(color="black", width=2)), 
                                     hoverinfo='text', text="Roller"), row=1, col=1)
            # Base Line
            fig.add_trace(go.Scatter(x=[x-sup_w, x+sup_w], y=[-sup_h, -sup_h], mode="lines", line=dict(color="black", width=2), hoverinfo='skip'), row=1, col=1)
            # Hatching Base
            for hx in np.linspace(x-sup_w, x+sup_w, 5):
                fig.add_trace(go.Scatter(x=[hx, hx-sup_w*0.5], y=[-sup_h, -sup_h*1.3], mode="lines", line=dict(color="black", width=1), hoverinfo='skip'), row=1, col=1)

        elif stype == "Fixed":
            # Vertical Thick Line
            fig.add_shape(type="line", x0=x, y0=-sup_h, x1=x, y1=sup_h, line=dict(color="black", width=4), row=1, col=1)
            # Diagonal Hatches
            dir_sign = -1 if x==0 else 1
            for hy in np.linspace(-sup_h, sup_h, 8):
                fig.add_shape(type="line", x0=x, y0=hy, x1=x + (sup_w*0.6*dir_sign), y1=hy - (sup_h*0.15), line=dict(color="black", width=1), row=1, col=1)

    # Draw Loads
    for l in display_loads:
        if l['type'] == 'U':
            x1, x2 = cum_len[l['span_idx']], cum_len[l['span_idx']+1]
            h = (abs(l['w'])/max_load_val) * (viz_h * 0.6)
            fig.add_trace(go.Scatter(x=[x1, x2, x2, x1], y=[0, 0, h, h], fill="toself", fillcolor="rgba(33, 150, 243, 0.3)", line_width=0, hoverinfo='skip'), row=1, col=1)
            fig.add_trace(go.Scatter(x=[x1, x2], y=[h, h], mode='lines', line=dict(color="#1976D2", width=2), hoverinfo='skip'), row=1, col=1)
            fig.add_annotation(x=(x1+x2)/2, y=h, text=f"<b>Wu={l['w']:.0f}</b>", showarrow=False, yshift=15, font=dict(color="#0D47A1", size=10), row=1, col=1)
            
        elif l['type'] == 'P':
            px = cum_len[l['span_idx']] + l['x']
            h = (abs(l['P'])/max_load_val) * (viz_h * 0.6)
            if h == 0: h = viz_h * 0.2
            fig.add_annotation(x=px, y=0, ax=0, ay=-40, text=f"<b>Pu={l['P']:.0f}</b>", arrowhead=2, arrowsize=1.5, arrowwidth=2, arrowcolor="#D32F2F", font=dict(color="#D32F2F", size=10), row=1, col=1)

    fig.update_yaxes(visible=False, range=[-sup_h*1.5, viz_h*1.4], row=1, col=1)
    
    # --- ROW 2 & 3: SHEAR & MOMENT ---
    # SFD
    fig.add_trace(go.Scatter(x=df['x'], y=df['shear'], fill='tozeroy', line=dict(color='#D32F2F', width=2), 
                             fillcolor='rgba(211, 47, 47, 0.1)', name="Shear", 
                             hovertemplate='x: %{x:.2f}<br>V: %{y:.2f}'), row=2, col=1)
    # BMD
    fig.add_trace(go.Scatter(x=df['x'], y=df['moment'], fill='tozeroy', line=dict(color='#1565C0', width=2), 
                             fillcolor='rgba(21, 101, 192, 0.1)', name="Moment",
                             hovertemplate='x: %{x:.2f}<br>M: %{y:.2f}'), row=3, col=1)

    # Labels for Min/Max
    for col, row, color, unit in [('shear', 2, '#D32F2F', u_force), ('moment', 3, '#1565C0', f"{u_force}-{u_len}")]:
        arr = df[col].to_numpy()
        mx, mn = np.max(arr), np.min(arr)
        imx, imn = np.argmax(arr), np.argmin(arr)
        
        # Determine range for padding
        rng = mx - mn
        pad = (rng if rng!=0 else 10) * 0.3
        
        # Label Max/Min points
        for val, idx, pos in [(mx, imx, "top"), (mn, imn, "bottom")]:
            if abs(val) > 1e-3: # Show label if not zero
                ys = 25 if pos=="top" else -25
                x_pos = df['x'].iloc[idx]
                label_txt = f"<b>{val:,.2f}</b><br><span style='font-size:10px'>@ {x_pos:.2f}m</span>"
                fig.add_annotation(x=x_pos, y=val, text=label_txt, showarrow=False, 
                                   bgcolor="rgba(255,255,255,0.85)", bordercolor=color, borderwidth=1, 
                                   font=dict(color=color, size=11), yshift=ys, row=row, col=1)

        fig.update_yaxes(title_text=f"{col.capitalize()} ({unit})", range=[mn-pad, mx+pad], row=row, col=1)

    # Interactive Spike Lines (The "Vertical Dashed Lines" requested)
    fig.update_xaxes(showgrid=True, gridcolor='#ECEFF1', showspikes=True, spikemode='across', spikethickness=1, spikedash='dash', spikecolor='#78909C', row=3, col=1)
    fig.update_xaxes(showspikes=True, spikemode='across', spikethickness=1, spikedash='dash', spikecolor='#78909C', row=2, col=1)
    
    fig.update_layout(height=900, template="plotly_white", margin=dict(t=30, b=30, l=60, r=30), showlegend=False, hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)


# ==========================================
# 2. CROSS SECTION HELPER
# ==========================================
def render_cross_section(b, h, cover, top_bars, bot_bars):
    fig = go.Figure()
    # Concrete Face
    fig.add_shape(type="rect", x0=0, y0=0, x1=b, y1=h, line=dict(color="black", width=2), fillcolor="#E0E0E0")
    # Stirrup Line (Schematic)
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
                r = (db/10)/2 if db > 0 else 0.5
                fig.add_shape(type="circle", x0=x-r, y0=y_pos-r, x1=x+r, y1=y_pos+r, fillcolor="#D32F2F", line_color="black")
    
    draw_bars(bot_bars, cover + 1.5)
    draw_bars(top_bars, h - cover - 1.5)
    
    fig.update_xaxes(visible=False, range=[-5, b+5])
    fig.update_yaxes(visible=False, range=[-5, h+5], scaleanchor="x", scaleratio=1)
    fig.update_layout(width=150, height=200, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', showlegend=False)
    return fig


# ==========================================
# 3. LONGITUDINAL VIEW (Detailed per Span)
# ==========================================
def render_longitudinal_view(spans, design_data):
    """
    design_data: list of dicts (one per span)
    [{'span': 1, 'bot_bars': '...', 'top_bars': '...', 'stirrups': '...'}]
    """
    total_len = sum(spans)
    cum_len = [0] + list(np.cumsum(spans))
    
    fig = go.Figure()
    
    # 1. Concrete Beam Body
    fig.add_shape(type="rect", x0=0, y0=0, x1=total_len, y1=10, 
                  line=dict(color="black", width=2), fillcolor="#F5F5F5", layer="below")
    
    # 2. Supports
    for x in cum_len:
        fig.add_trace(go.Scatter(x=[x, x-0.15, x+0.15, x], y=[0, -1.5, -1.5, 0], 
                                 fill="toself", fillcolor="#90A4AE", line_color="black", showlegend=False, hoverinfo='skip'))

    # 3. Reinforcement (Per Span)
    for i, span_len in enumerate(spans):
        x_start = cum_len[i]
        x_end = cum_len[i+1]
        data = design_data[i]
        
        # -- Bottom Bars (Run full span length minus cover) --
        if data['bot_bars']:
            fig.add_trace(go.Scatter(
                x=[x_start + 0.1, x_end - 0.1], 
                y=[1.5, 1.5], 
                mode="lines+text",
                line=dict(color="#1565C0", width=4), 
                text=[f"{data['bot_bars']}", ""],
                textposition="top right",
                name=f"Span {i+1} Bot",
                hovertemplate=f"Span {i+1} Bottom: {data['bot_bars']}"
            ))

        # -- Top Bars (Schematic: Over supports) --
        # Note: In real continuous beams, top bars are at supports. 
        # Here we visualize the requirement "Top" from the design result near the supports of this span.
        if data['top_bars']:
            # L/3 length from each side
            cut_len = span_len / 3.0
            # Left side of span
            fig.add_trace(go.Scatter(x=[x_start, x_start + cut_len], y=[8.5, 8.5], mode="lines",
                                     line=dict(color="#D32F2F", width=4), showlegend=False, hovertemplate=f"Top: {data['top_bars']}"))
            # Right side of span
            fig.add_trace(go.Scatter(x=[x_end - cut_len, x_end], y=[8.5, 8.5], mode="lines",
                                     line=dict(color="#D32F2F", width=4), showlegend=False, hovertemplate=f"Top: {data['top_bars']}"))
            
            # Label just once per span center-top (schematic)
            fig.add_annotation(x=x_start + span_len/2, y=9.5, text=f"Top: {data['top_bars']}", showarrow=False, font=dict(color="#D32F2F", size=9))

        # -- Stirrups (Visualizing Spacing) --
        stirrup_txt = data['stirrups']
        s_val = 0.25 # default
        if "@" in stirrup_txt:
            try:
                # Extract number after @
                val_str = stirrup_txt.split("@")[1]
                # Remove units if present
                val_str = ''.join(filter(str.isdigit, val_str)) 
                # Convert
                if val_str:
                    s_val = float(val_str) / 100.0 if "cm" in stirrup_txt else float(val_str) / 1000.0
            except:
                s_val = 0.25
        
        if "None" not in stirrup_txt and s_val > 0:
            # Generate stirrup positions for this span
            sx_list = np.arange(x_start + 0.1, x_end - 0.1, s_val)
            # Create a single trace for all stirrups in this span (optimization)
            x_pts, y_pts = [], []
            for sx in sx_list:
                x_pts.extend([sx, sx, None])
                y_pts.extend([1.5, 8.5, None])
            
            fig.add_trace(go.Scatter(x=x_pts, y=y_pts, mode="lines", line=dict(color="green", width=1), hoverinfo='skip'))
            # Label
            fig.add_annotation(x=x_start + span_len/2, y=5, text=f"<b>{stirrup_txt}</b>", bgcolor="rgba(255,255,255,0.7)", bordercolor="green", showarrow=False, font=dict(size=9))

    fig.update_xaxes(showgrid=False, visible=False)
    fig.update_yaxes(visible=False, range=[-2, 12])
    fig.update_layout(height=280, title="<b>Longitudinal Reinforcement Detail (Profile)</b>",
                      margin=dict(l=20, r=20, t=40, b=10), template="plotly_white", showlegend=False)
    
    return fig


# ==========================================
# 4. MAIN DESIGN RENDERER (Split by Span)
# ==========================================
def render_design_results(df, params, spans, supports):
    st.markdown('<div class="section-header">4️⃣ Reinforced Concrete Design</div>', unsafe_allow_html=True)
    
    if "WSD" in params['method']:
        st.warning("WSD Method is not implemented yet. Please switch to SDM.")
        return

    cum_len = [0] + list(np.cumsum(spans))
    
    # Storage for Visualization
    span_design_data = [] 

    # --- TABS FOR EACH SPAN ---
    tabs = st.tabs([f"Span {i+1}" for i in range(len(spans))])
    
    for i, tab in enumerate(tabs):
        x_start = cum_len[i]
        x_end = cum_len[i+1]
        
        # FILTER DATA FOR THIS SPAN
        # Add small buffer to avoid boundary issues, but include ends
        mask = (df['x'] >= x_start) & (df['x'] <= x_end)
        df_span = df[mask]
        
        if df_span.empty:
            continue
            
        # 1. FIND FORCES IN THIS SPAN
        # Max Positive Moment (usually Midspan)
        m_pos_max = df_span['moment'].max()
        # Max Negative Moment (usually Supports - taking min since negative)
        m_neg_max = df_span['moment'].min() 
        # Max Shear (Absolute)
        v_max = df_span['shear'].abs().max()
        
        # 2. DESIGN CALCULATIONS
        # Positive M (Bottom Steel)
        res_pos = rc_design.calculate_flexure_sdm(max(0, m_pos_max), f"Span {i+1} (+M)", params)
        # Negative M (Top Steel) - Design for the worst negative moment in this span region
        res_neg = rc_design.calculate_flexure_sdm(m_neg_max if m_neg_max < 0 else 0, f"Span {i+1} (-M)", params)
        # Shear
        vu, phi_vc, stirrup_txt = rc_design.calculate_shear_capacity(v_max, params)
        
        # Save for Longitudinal View
        span_design_data.append({
            'bot_bars': res_pos['Bars'],
            'top_bars': res_neg['Bars'],
            'stirrups': stirrup_txt
        })

        # 3. DISPLAY CARDS
        with tab:
            col1, col2, col3 = st.columns(3)
            
            # -- Top Steel Card --
            with col1:
                st.markdown("**Top Reinforcement (-M)**")
                st.error(f"{res_neg['Bars']}")
                st.caption(f"Mu: {abs(res_neg['Mu']):,.2f} | As: {res_neg['As_req']:.2f}")
                if "Over" not in res_neg['Status'] and res_neg['Bars']:
                    st.plotly_chart(render_cross_section(params['b'], params['h'], params['cv'], res_neg['Bars'], ""), use_container_width=True)
                else:
                    st.write(res_neg['Status'])

            # -- Bottom Steel Card --
            with col2:
                st.markdown("**Bottom Reinforcement (+M)**")
                st.info(f"{res_pos['Bars']}")
                st.caption(f"Mu: {res_pos['Mu']:,.2f} | As: {res_pos['As_req']:.2f}")
                if "Over" not in res_pos['Status'] and res_pos['Bars']:
                    st.plotly_chart(render_cross_section(params['b'], params['h'], params['cv'], "", res_pos['Bars']), use_container_width=True)
                else:
                    st.write(res_pos['Status'])

            # -- Shear Card --
            with col3:
                st.markdown("**Shear Reinforcement**")
                st.success(f"**{stirrup_txt}**")
                st.write(f"Vu max: {vu:,.2f}")
                st.write(f"ϕVc: {phi_vc:,.2f}")
                status_color = "green" if vu <= phi_vc else "red"
                st.markdown(f"Status: <span style='color:{status_color}'>{'OK' if vu <= phi_vc else 'Check Section'}</span>", unsafe_allow_html=True)

    # --- LONGITUDINAL DIAGRAM (GLOBAL) ---
    st.markdown("---")
    st.plotly_chart(render_longitudinal_view(spans, span_design_data), use_container_width=True)
