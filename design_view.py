import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd # เพิ่ม import pandas
import rc_design 

def add_peak_annotations(fig, x_data, y_data, row, col, unit):
    """Annotate Max/Min values with smart positioning."""
    try:
        y_vals = [float(y) for y in y_data]
        idx_max = np.argmax(y_vals)
        idx_min = np.argmin(y_vals)
        
        # annotate max
        fig.add_annotation(
            x=x_data[idx_max], y=y_vals[idx_max],
            text=f"<b>Max: {y_vals[idx_max]:,.0f} {unit}</b>",
            showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2,
            ax=0, ay=-40 if y_vals[idx_max] >= 0 else 40,
            bgcolor="rgba(255,255,255,0.7)", bordercolor="#333",
            row=row, col=col
        )
        # annotate min (only if different from max)
        if idx_min != idx_max:
            fig.add_annotation(
                x=x_data[idx_min], y=y_vals[idx_min],
                text=f"<b>Min: {y_vals[idx_min]:,.0f} {unit}</b>",
                showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2,
                ax=0, ay=40 if y_vals[idx_min] <= 0 else -40,
                bgcolor="rgba(255,255,255,0.7)", bordercolor="#333",
                row=row, col=col
            )
    except:
        pass

def draw_diagrams(df, spans, supports, loads, u_force, u_len):
    cum_len = [0] + list(np.cumsum(spans))
    L_total = cum_len[-1]
    
    # --- FIX SCALING ISSUE ---
    # ใช้สัดส่วนความยาวคานกำหนดขนาด Support แทนการใช้ค่า Load
    viz_scale = L_total / 20.0  # Support สูงประมาณ 1/20 ของความยาวรวม
    
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.1,
        subplot_titles=("<b>Structure & Loads</b>", f"<b>Shear Force ({u_force})</b>", f"<b>Bending Moment ({u_force}-{u_len})</b>"),
        row_heights=[0.3, 0.35, 0.35]
    )
    
    # Grid Lines
    for x_s in cum_len:
        fig.add_vline(x=x_s, line_width=1, line_dash="dash", line_color="gray", opacity=0.5)

    # --- ROW 1: Structure ---
    # Draw Beam Line
    fig.add_trace(go.Scatter(x=[0, L_total], y=[0, 0], mode='lines', line=dict(color='black', width=4), hoverinfo='none'), row=1, col=1)

    # Draw Supports (Scaled geometrically)
    for i, x in enumerate(cum_len):
        stype = supports.iloc[i]['type']
        # Label Node
        fig.add_annotation(x=x, y=-viz_scale*1.2, text=f"<b>N{i+1}</b>", showarrow=False, font=dict(size=12), row=1, col=1)
        
        if stype == "Pin":
            fig.add_trace(go.Scatter(x=[x], y=[-viz_scale/2], mode='markers', marker=dict(symbol='triangle-up', size=15, color='#90A4AE', line=dict(width=2, color='black')), showlegend=False, hoverinfo='text', hovertext=f"Pin @ {x:.2f}"), row=1, col=1)
        elif stype == "Roller":
            fig.add_trace(go.Scatter(x=[x], y=[-viz_scale/2], mode='markers', marker=dict(symbol='circle', size=15, color='white', line=dict(width=2, color='black')), showlegend=False, hoverinfo='text', hovertext=f"Roller @ {x:.2f}"), row=1, col=1)
            fig.add_shape(type="line", x0=x-viz_scale/2, y0=-viz_scale, x1=x+viz_scale/2, y1=-viz_scale, line=dict(color="black", width=2), row=1, col=1)
        elif stype == "Fixed":
            fig.add_shape(type="rect", x0=x-viz_scale/4, y0=-viz_scale, x1=x+viz_scale/4, y1=viz_scale, line=dict(color="black", width=2), fillcolor="gray", row=1, col=1)

    # Draw Loads (Normalized Height)
    # หาค่า Load สูงสุดเพื่อ Normalize ความสูงลูกศร ไม่ให้บังคาน
    max_load_val = 1.0
    val_list = [abs(l['w']) for l in loads if l['type']=='U'] + [abs(l['P']) for l in loads if l['type']=='P']
    if val_list: max_load_val = max(val_list)

    arrow_h = viz_scale * 1.5 # ความสูงลูกศรให้สัมพันธ์กับ Support

    for l in loads:
        if l['type'] == 'U':
            x1, x2 = cum_len[l['span_idx']], cum_len[l['span_idx']+1]
            # Normalize height: ถ้า load เยอะให้เต็ม max_h, ถ้าน้อยให้ลดหลั่นลงมา
            h_ratio = abs(float(l['w'])) / max_load_val
            h_draw = arrow_h * max(0.3, h_ratio) # ขั้นต่ำ 30% ของความสูง
            
            fig.add_trace(go.Scatter(x=[x1, x2, x2, x1], y=[0, 0, h_draw, h_draw], fill="toself", fillcolor="rgba(30, 136, 229, 0.3)", line_width=0, hoverinfo='skip'), row=1, col=1)
            fig.add_annotation(x=(x1+x2)/2, y=h_draw, text=f"w={float(l['w']):.0f}", showarrow=False, yshift=10, font=dict(color="#1565C0", size=10), row=1, col=1)
            
        elif l['type'] == 'P':
            px = cum_len[l['span_idx']] + float(l['x'])
            h_ratio = abs(float(l['P'])) / max_load_val
            len_arrow = arrow_h * max(0.5, h_ratio)
            
            fig.add_annotation(x=px, y=0, ax=0, ay=-len_arrow*40, # Plotly arrow unit pixels
                               text=f"P={float(l['P']):.0f}", arrowhead=2, arrowwidth=2, arrowcolor="#D32F2F", font=dict(color="#D32F2F", size=10), row=1, col=1)

    # Fix Row 1 Y-Axis (Geometry Scale Only)
    fig.update_yaxes(range=[-viz_scale*2, arrow_h*2], visible=False, row=1, col=1)

    # --- ROW 2 & 3: Shear & Moment ---
    # Shear
    fig.add_trace(go.Scatter(x=df['x'], y=df['shear'], fill='tozeroy', line=dict(color='#D32F2F', width=2), name="Shear", hovertemplate='V: %{y:,.0f}'), row=2, col=1)
    add_peak_annotations(fig, df['x'].values, df['shear'].values, 2, 1, u_force)
    
    # Moment
    fig.add_trace(go.Scatter(x=df['x'], y=df['moment'], fill='tozeroy', line=dict(color='#1976D2', width=2), name="Moment", hovertemplate='M: %{y:,.0f}'), row=3, col=1)
    add_peak_annotations(fig, df['x'].values, df['moment'].values, 3, 1, f"{u_force}-{u_len}")

    fig.update_layout(height=800, template="plotly_white", showlegend=False, hovermode="x unified", margin=dict(t=30, b=30, l=20, r=20))
    st.plotly_chart(fig, use_container_width=True, key="main_diagram")

def render_combined_section(b, h, cover, top_bars, bot_bars):
    """Draws section with safe casting."""
    try:
        b, h, cover = float(b), float(h), float(cover)
    except: return go.Figure()

    fig = go.Figure()
    # Concrete Face
    fig.add_shape(type="rect", x0=0, y0=0, x1=b, y1=h, line=dict(color="black", width=3), fillcolor="#ECEFF1")
    # Stirrup Line
    fig.add_shape(type="rect", x0=cover, y0=cover, x1=b-cover, y1=h-cover, line=dict(color="#388E3C", width=2, dash="longdash"))
    
    def draw_rebars(bars, y_center, color, label_pos_offset):
        parsed = rc_design.parse_bars(bars)
        if parsed:
            num, db = parsed
            r = db/20 # Scale for display
            # Distribute bars
            if num == 1:
                xs = [b/2]
            else:
                start = cover + r
                end = b - cover - r
                xs = np.linspace(start, end, num)
                
            for x in xs:
                fig.add_shape(type="circle", x0=x-r, y0=y_center-r, x1=x+r, y1=y_center+r, fillcolor=color, line_color="black")
            
            fig.add_annotation(x=b/2, y=y_center+label_pos_offset, text=f"<b>{bars}</b>", showarrow=False, font=dict(color=color, size=14, weight="bold"))

    draw_rebars(bot_bars, cover + 2.0, "#1565C0", 6) 
    draw_rebars(top_bars, h - cover - 2.0, "#C62828", -6)

    fig.update_xaxes(visible=False, range=[-10, b+10])
    fig.update_yaxes(visible=False, range=[-10, h+10], scaleanchor="x", scaleratio=1)
    fig.update_layout(width=300, height=300, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    return fig

def render_design_results(df, params, spans, span_props_list, supports):
    st.markdown("### 4️⃣ Structural Design Results & Detailing")
    cum_len = [0] + list(np.cumsum(spans))
    
    # 1. PRE-CALCULATE ALL DATA FOR SUMMARY TABLE
    summary_data = []
    
    for i in range(len(spans)):
        sp = span_props_list[i]
        b_s, h_s, cv_s = sp['b'], sp['h'], sp['cv']

        # Get forces in this span
        mask = (df['x'] >= cum_len[i]) & (df['x'] <= cum_len[i+1])
        sub_df = df[mask]
        
        if sub_df.empty: continue
        
        m_pos = max(0, sub_df['moment'].max())
        m_neg = min(0, sub_df['moment'].min()) # Simplified: take max neg of span
        v_u = sub_df['shear'].abs().max()
        
        # Design
        des_pos = rc_design.calculate_flexure_sdm(m_pos, "Midspan", b_s, h_s, cv_s, params)
        des_neg = rc_design.calculate_flexure_sdm(m_neg, "Support", b_s, h_s, cv_s, params)
        v_act, v_cap, stir_txt, v_log = rc_design.calculate_shear_capacity(v_u, b_s, h_s, cv_s, params)
        
        status_icon = "✅" if ("OK" in des_pos['Status'] and "OK" in des_neg['Status'] and v_act <= v_cap) else "⚠️"
        
        summary_data.append({
            "Span": f"Span {i+1}",
            "Section (cm)": f"{b_s:.0f}x{h_s:.0f}",
            "Top Bars (-M)": des_neg['Bars'],
            "Bot Bars (+M)": des_pos['Bars'],
            "Stirrups": stir_txt,
            "Status": status_icon,
            # Hidden objects for detailed view
            "_des_pos": des_pos, "_des_neg": des_neg, "_stir_txt": stir_txt, "_v_log": v_log
        })

    # 2. SHOW SUMMARY TABLE (The missing table!)
    st.markdown("#### 📋 Design Summary")
    df_sum = pd.DataFrame(summary_data).drop(columns=["_des_pos", "_des_neg", "_stir_txt", "_v_log"])
    st.dataframe(df_sum, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("#### 🔍 Detailed Calculation per Span")
    
    # 3. DETAILED TABS
    tabs = st.tabs([d["Span"] for d in summary_data])
    for i, tab in enumerate(tabs):
        data = summary_data[i]
        sp = span_props_list[i]
        
        with tab:
            c1, c2 = st.columns([1, 1.5])
            with c1:
                st.markdown(f"**Section:** {data['Section (cm)']} (Cov: {sp['cv']}cm)")
                # Draw Section
                fig_sec = render_combined_section(sp['b'], sp['h'], sp['cv'], data['_des_neg']['Bars'], data['_des_pos']['Bars'])
                st.plotly_chart(fig_sec, use_container_width=True)
                
            with c2:
                with st.expander("📝 Show Calculation Steps", expanded=False):
                    st.markdown("**Flexure (Top/Support)**")
                    for l in data['_des_neg']['Log']: st.markdown(l)
                    st.divider()
                    st.markdown("**Flexure (Bottom/Midspan)**")
                    for l in data['_des_pos']['Log']: st.markdown(l)
                    st.divider()
                    st.markdown("**Shear Check**")
                    for l in data['_v_log']: st.markdown(l)

    # 4. LONGITUDINAL VIEW
    st.markdown("---")
    # Prepare simple data for longitudinal view
    long_data = []
    for d in summary_data:
        long_data.append({'bot_bars': d['Bot Bars (+M)'], 'top_bars': d['Top Bars (-M)'], 'stirrups': d['Stirrups']})
        
    render_longitudinal_view(spans, supports, long_data)

def render_longitudinal_view(spans, supports, design_data):
    # (ฟังก์ชันนี้เหมือนเดิม แต่ใส่ไว้เพื่อให้ไฟล์ครบสมบูรณ์)
    total_len = sum(spans)
    cum_len = [0] + list(np.cumsum(spans))
    fig = go.Figure()
    
    # Beam body
    fig.add_shape(type="rect", x0=0, y0=0, x1=total_len, y1=10, line=dict(color="black", width=2), fillcolor="#FAFAFA", layer="below")
    
    for i, x in enumerate(cum_len):
        fig.add_vline(x=x, line_width=1, line_dash="dot", color="gray")
        # Draw supports simplified
        fig.add_trace(go.Scatter(x=[x], y=[-1], mode='markers', marker=dict(symbol='triangle-up', size=12, color='gray'), showlegend=False))

    for i, L in enumerate(spans):
        xs, xe = cum_len[i], cum_len[i+1]
        data = design_data[i]
        
        # Bot Bars
        if data['bot_bars']:
            fig.add_trace(go.Scatter(x=[xs+0.2, xe-0.2], y=[2, 2], mode="lines+text", line=dict(color="#1565C0", width=3), 
                                     text=[f"Bot: {data['bot_bars']}", ""], textposition="top right", showlegend=False))
        # Top Bars
        if data['top_bars']:
            cut = L/3.5
            # Left Support
            fig.add_trace(go.Scatter(x=[xs, xs+cut], y=[8, 8], mode="lines+text", line=dict(color="#C62828", width=3),
                                     text=[f"Top: {data['top_bars']}", ""], textposition="bottom right", showlegend=False))
            # Right Support
            fig.add_trace(go.Scatter(x=[xe-cut, xe], y=[8, 8], mode="lines", line=dict(color="#C62828", width=3), showlegend=False))
        
        # Stirrups
        if data['stirrups']:
            fig.add_annotation(x=xs+L/2, y=5, text=f"{data['stirrups']}", showarrow=False, font=dict(size=9, color="green"))

    fig.update_xaxes(visible=False, range=[-0.5, total_len+0.5])
    fig.update_yaxes(visible=False, range=[-2, 12])
    fig.update_layout(height=250, title="<b>Longitudinal Reinforcement Profile</b>", margin=dict(t=30,b=10), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
