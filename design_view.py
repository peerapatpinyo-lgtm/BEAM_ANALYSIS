import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import rc_design 

def draw_diagrams(df, spans, supports, loads, u_force, u_len):
    cum_len = [0] + list(np.cumsum(spans))
    L_total = cum_len[-1]
    
    # --- 1. PRE-PROCESS LOADS (AGGREGATION) ---
    # รวม Point Load ที่ลงตำแหน่งเดียวกันให้เป็นก้อนเดียวเพื่อการแสดงผล
    display_loads = []
    
    # เก็บ Uniform Load เหมือนเดิม
    for l in loads:
        if l['type'] == 'U':
            display_loads.append(l)
            
    # รวม Point Load
    p_map = {} # Key: (span_idx, x), Value: sum_P
    for l in loads:
        if l['type'] == 'P':
            key = (l['span_idx'], l['x'])
            p_map[key] = p_map.get(key, 0) + l['P']
            
    for (s_idx, x_val), total_p in p_map.items():
        if total_p != 0:
            display_loads.append({'span_idx': s_idx, 'type': 'P', 'P': total_p, 'x': x_val})

    # คำนวณ Scale
    val_list = [abs(l['w']) for l in display_loads if l['type']=='U'] + [abs(l['P']) for l in display_loads if l['type']=='P']
    max_load_val = max(val_list) if val_list else 100
    
    viz_h = max_load_val * 1.5 
    sup_h = viz_h * 0.15
    sup_w = L_total * 0.02 

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                        subplot_titles=(f"<b>Loading Diagram</b>", 
                                        f"<b>Shear Force Diagram (SFD)</b>", 
                                        f"<b>Bending Moment Diagram (BMD)</b>"),
                        row_heights=[0.3, 0.35, 0.35])
    
    # --- LOAD DIAGRAM ---
    fig.add_trace(go.Scatter(x=[0, L_total], y=[0, 0], mode='lines', 
                             line=dict(color='black', width=4), hoverinfo='none'), row=1, col=1)

    # Supports
    for i, x in enumerate(cum_len):
        stype = supports.iloc[i]['type']
        if stype == "Pin":
            fig.add_trace(go.Scatter(x=[x, x-sup_w, x+sup_w, x], 
                                     y=[0, -sup_h, -sup_h, 0], 
                                     fill="toself", fillcolor="#90A4AE", line_color="black", mode='lines', showlegend=False, hoverinfo='text', text="Pin"), row=1, col=1)
        elif stype == "Roller":
            fig.add_trace(go.Scatter(x=[x], y=[-sup_h/2], mode="markers", 
                                     marker=dict(size=10, color="white", line=dict(color="black", width=2)), 
                                     showlegend=False, hoverinfo='text', text="Roller"), row=1, col=1)
            fig.add_trace(go.Scatter(x=[x-sup_w, x+sup_w], y=[-sup_h, -sup_h], mode="lines", 
                                     line=dict(color="black", width=2), showlegend=False), row=1, col=1)
        elif stype == "Fixed":
            fig.add_shape(type="line", x0=x, y0=-sup_h, x1=x, y1=sup_h, line=dict(color="black", width=4), row=1, col=1)
            for h_line in np.linspace(-sup_h, sup_h, 5):
                dir_sign = -1 if x==0 else 1
                fig.add_shape(type="line", x0=x, y0=h_line, x1=x + (sup_w*0.5*dir_sign), y1=h_line - (sup_h*0.2), line=dict(color="black", width=1), row=1, col=1)
            
    # Draw Aggregated Loads
    for l in display_loads:
        if l['type'] == 'U':
            x1, x2 = cum_len[l['span_idx']], cum_len[l['span_idx']+1]
            h = (abs(l['w'])/max_load_val) * (viz_h * 0.6)
            fig.add_trace(go.Scatter(x=[x1, x2, x2, x1], y=[0, 0, h, h], fill="toself", fillcolor="rgba(33, 150, 243, 0.3)", line_width=0, showlegend=False), row=1, col=1)
            fig.add_trace(go.Scatter(x=[x1, x2], y=[h, h], mode='lines', line=dict(color="#1976D2", width=2), showlegend=False), row=1, col=1)
            fig.add_annotation(x=(x1+x2)/2, y=h, text=f"<b>Wu={l['w']:.0f}</b>", showarrow=False, yshift=15, font=dict(color="#0D47A1", size=11), row=1, col=1)
            
        elif l['type'] == 'P':
            px = cum_len[l['span_idx']] + l['x']
            h = (abs(l['P'])/max_load_val) * (viz_h * 0.6)
            if h == 0: h = viz_h * 0.2
            
            fig.add_annotation(
                x=px, y=0,
                ax=0, ay=-40,
                text=f"<b>Pu={l['P']:.0f}</b>",
                arrowhead=2, arrowsize=1.5, arrowwidth=2, arrowcolor="#D32F2F",
                font=dict(color="#D32F2F", size=11),
                row=1, col=1
            )
            
    fig.update_yaxes(visible=False, range=[-sup_h*1.5, viz_h*1.2], row=1, col=1)
    
    # --- SFD & BMD (ใส่ชื่อแกน Y) ---
    fig.add_trace(go.Scatter(x=df['x'], y=df['shear'], fill='tozeroy', line=dict(color='#D32F2F', width=2), fillcolor='rgba(211, 47, 47, 0.1)', name="Shear"), row=2, col=1)
    fig.add_trace(go.Scatter(x=df['x'], y=df['moment'], fill='tozeroy', line=dict(color='#1565C0', width=2), fillcolor='rgba(21, 101, 192, 0.1)', name="Moment"), row=3, col=1)
    
    # Update Axes Titles & Labels
    fig.update_yaxes(title_text=f"Shear V ({u_force})", title_font=dict(size=12, color='#D32F2F'), row=2, col=1)
    fig.update_yaxes(title_text=f"Moment M ({u_force}-{u_len})", title_font=dict(size=12, color='#1565C0'), row=3, col=1)

    for col, row, color in [('shear', 2, '#D32F2F'), ('moment', 3, '#1565C0')]:
        arr = df[col].to_numpy()
        mx, mn = np.max(arr), np.min(arr)
        imx, imn = np.argmax(arr), np.argmin(arr)
        
        for val, idx, pos in [(mx, imx, "top"), (mn, imn, "bottom")]:
            if abs(val) > 1e-3:
                ys = 25 if pos=="top" else -25
                x_pos = df['x'].iloc[idx]
                label_txt = f"<b>{val:,.2f}</b><br><span style='font-size:10px'>@ {x_pos:.2f} m</span>"
                
                fig.add_annotation(
                    x=x_pos, y=val, 
                    text=label_txt, 
                    showarrow=False, 
                    bgcolor="rgba(255,255,255,0.85)", bordercolor=color, borderwidth=1, borderpad=3,
                    font=dict(color=color, size=11), yshift=ys, row=row, col=1
                )
        
        rng = mx - mn
        pad = (rng if rng!=0 else 10) * 0.3
        fig.update_yaxes(range=[mn-pad, mx+pad], row=row, col=1)

    fig.update_layout(height=900, template="plotly_white", margin=dict(t=30, b=30, l=60, r=30), showlegend=False)
    fig.update_xaxes(showgrid=True, gridcolor='#ECEFF1', title=f"Distance ({u_len})", row=3, col=1)
    st.plotly_chart(fig, use_container_width=True)

# --- รูป Cross Section (เดิม) ---
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
                radius = (db/10)/2 
                fig.add_shape(type="circle", x0=x-radius, y0=y_pos-radius, x1=x+radius, y1=y_pos+radius, 
                              fillcolor="#D32F2F", line_color="black")
    
    draw_bars(bot_bars, cover + 1.5)
    draw_bars(top_bars, h - cover - 1.5)
    
    fig.update_xaxes(visible=False, range=[-5, b+5])
    fig.update_yaxes(visible=False, range=[-5, h+5], scaleanchor="x", scaleratio=1)
    fig.update_layout(width=150, height=200, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', showlegend=False)
    return fig

# --- เพิ่ม: รูปเหล็กตามยาว (Longitudinal View) ---
def render_longitudinal_view(spans, supports, res_pos, res_neg):
    total_len = sum(spans)
    fig = go.Figure()
    
    # 1. ตัวคาน
    fig.add_shape(type="rect", x0=0, y0=0, x1=total_len, y1=10, 
                  line=dict(color="black", width=2), fillcolor="#F5F5F5")
    
    # 2. Supports
    cum_x = 0
    sup_x_list = [0] + list(np.cumsum(spans))
    
    for i, x in enumerate(sup_x_list):
        # Draw Support Triangle
        fig.add_trace(go.Scatter(x=[x, x-0.15, x+0.15, x], y=[0, -1.5, -1.5, 0], 
                                 fill="toself", fillcolor="#90A4AE", line_color="black", showlegend=False))
    
    # 3. Rebar (Simplified Representation)
    # Top Bar (Mainly at supports, but drawn full length for schematic or split)
    # แสดงเหล็กบน (จากผลคำนวณ -M)
    if res_neg['Bars']:
        fig.add_trace(go.Scatter(x=[0, total_len], y=[8.5, 8.5], mode="lines+text",
                                 line=dict(color="#D32F2F", width=3),
                                 text=[f"Top: {res_neg['Bars']}", ""], textposition="top right",
                                 name="Top Bar"))

    # Bottom Bar (Mainly at midspan, drawn full length)
    # แสดงเหล็กล่าง (จากผลคำนวณ +M)
    if res_pos['Bars']:
        fig.add_trace(go.Scatter(x=[0, total_len], y=[1.5, 1.5], mode="lines+text",
                                 line=dict(color="#1565C0", width=3),
                                 text=[f"Bot: {res_pos['Bars']}", ""], textposition="bottom right",
                                 name="Bottom Bar"))
        
    # Stirrups (Indicative lines)
    stirrup_x = np.linspace(0.2, total_len-0.2, 20)
    for sx in stirrup_x:
        fig.add_shape(type="line", x0=sx, y0=1.5, x1=sx, y1=8.5, line=dict(color="green", width=1, dash="dot"))

    fig.update_xaxes(showgrid=False, visible=False)
    fig.update_yaxes(visible=False, range=[-3, 15])
    fig.update_layout(height=180, title="<b>Longitudinal Reinforcement Profile (Schematic)</b>",
                      margin=dict(l=20, r=20, t=40, b=10), template="plotly_white", showlegend=False)
    
    return fig

def render_design_results(df, params, spans, supports):
    st.markdown('<div class="section-header">4️⃣ Reinforced Concrete Design</div>', unsafe_allow_html=True)
    
    m_max = df['moment'].max()
    m_min = df['moment'].min()
    u_moment = f"{params['u_force']}-{params['u_len']}"
    
    if "SDM" in params['method']:
        res_pos = rc_design.calculate_flexure_sdm(m_max, "Mid-Span (+M)", params)
        res_neg = rc_design.calculate_flexure_sdm(m_min, "Support (-M)", params)
        
        # --- Cards ---
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"""
            <div class="card">
                <h4>{res_pos['Type']}</h4>
                <div style="font-size:1.5em; font-weight:bold; color:#1565C0;">{res_pos['Mu']:,.2f} <span style="font-size:0.6em">{u_moment}</span></div>
                <hr>
                <div style="background:#E3F2FD; padding:8px; border-radius:5px;"><b>Use:</b> {res_pos['Bars']}</div>
                <small style="color:gray">As req: {res_pos['As_req']:.2f} cm² | {res_pos['Status']}</small>
            </div>
            """, unsafe_allow_html=True)
            if "Over" not in res_pos['Status']:
                st.plotly_chart(render_cross_section(params['b'], params['h'], params['cv'], "", res_pos['Bars']), use_container_width=True)

        with c2:
            st.markdown(f"""
            <div class="card">
                <h4>{res_neg['Type']}</h4>
                <div style="font-size:1.5em; font-weight:bold; color:#C62828;">{res_neg['Mu']:,.2f} <span style="font-size:0.6em">{u_moment}</span></div>
                <hr>
                <div style="background:#FFEBEE; padding:8px; border-radius:5px;"><b>Use:</b> {res_neg['Bars']}</div>
                <small style="color:gray">As req: {res_neg['As_req']:.2f} cm² | {res_neg['Status']}</small>
            </div>
            """, unsafe_allow_html=True)
            if "Over" not in res_neg['Status']:
                st.plotly_chart(render_cross_section(params['b'], params['h'], params['cv'], res_neg['Bars'], ""), use_container_width=True)

        # --- Longitudinal View (New) ---
        st.markdown("---")
        st.plotly_chart(render_longitudinal_view(spans, supports, res_pos, res_neg), use_container_width=True)

    else:
        st.info("WSD Method not implemented in this version.")

    st.markdown("---")
    # Shear Part remains same
    v_max = df['shear'].abs().max()
    vu_val, phi_vc = rc_design.calculate_shear_capacity(v_max, params)
    color = "green" if vu_val <= phi_vc/2 else ("orange" if vu_val <= phi_vc else "red")
    req_stirrup = "None" if vu_val <= phi_vc/2 else ("Min Stirrups" if vu_val <= phi_vc else "Design Stirrups Req.")
        
    st.markdown(f"""
    <div style="padding:15px; background-color:white; border-left:6px solid {color}; box-shadow: 0 2px 5px rgba(0,0,0,0.05);">
        <h4>Shear Design ({params['u_force']})</h4>
        <b>Max Shear Vu:</b> {vu_val:,.2f} <br>
        <b>Capacity ϕVc:</b> {phi_vc:,.2f} <br>
        <b>Status:</b> <span style="color:{color}; font-weight:bold">{req_stirrup}</span>
    </div>
    """, unsafe_allow_html=True)
