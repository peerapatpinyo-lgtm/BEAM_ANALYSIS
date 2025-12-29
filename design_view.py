import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import rc_design 

def draw_diagrams(df, spans, supports, loads, u_force, u_len):
    cum_len = [0] + list(np.cumsum(spans))
    L_total = cum_len[-1]
    
    # คำนวณ Scale สำหรับวาดกราฟ
    val_list = [abs(l['w']) for l in loads if l['type']=='U'] + [abs(l['P']) for l in loads if l['type']=='P']
    max_load_val = max(val_list) if val_list else 100
    
    # Scale ความสูงของกราฟ Load Diagram
    # ให้ Support สูงประมาณ 15% ของพื้นที่กราฟ, Load สูงสุด 50%
    viz_h = max_load_val * 1.5 
    sup_h = viz_h * 0.15  # ปรับขนาด Support ตามแกน Y (Load) เพื่อความสวยงาม
    sup_w = L_total * 0.02 # ปรับความกว้าง Support ตามแกน X (Length)

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=(f"<b>Loading Diagram</b>", 
                                        f"<b>Shear Force ({u_force})</b>", 
                                        f"<b>Bending Moment ({u_force}-{u_len})</b>"),
                        row_heights=[0.3, 0.35, 0.35])
    
    # --- 1. Loading Diagram ---
    # เส้นคาน
    fig.add_trace(go.Scatter(x=[0, L_total], y=[0, 0], mode='lines', 
                             line=dict(color='black', width=4), hoverinfo='none'), row=1, col=1)

    # วาด Supports (ปรับสเกลใหม่)
    for i, x in enumerate(cum_len):
        stype = supports.iloc[i]['type']
        if stype == "Pin":
            # สามเหลี่ยม
            fig.add_trace(go.Scatter(x=[x, x-sup_w, x+sup_w, x], 
                                     y=[0, -sup_h, -sup_h, 0], 
                                     fill="toself", fillcolor="#90A4AE", line_color="black", mode='lines', showlegend=False, hoverinfo='text', text="Pin"), row=1, col=1)
        elif stype == "Roller":
            # วงกลม + ฐาน
            fig.add_trace(go.Scatter(x=[x], y=[-sup_h/2], mode="markers", 
                                     marker=dict(size=10, color="white", line=dict(color="black", width=2)), 
                                     showlegend=False, hoverinfo='text', text="Roller"), row=1, col=1)
            fig.add_trace(go.Scatter(x=[x-sup_w, x+sup_w], y=[-sup_h, -sup_h], mode="lines", 
                                     line=dict(color="black", width=2), showlegend=False), row=1, col=1)
        elif stype == "Fixed":
            # ขีดแนวตั้ง + แรเงา
            fig.add_shape(type="line", x0=x, y0=-sup_h, x1=x, y1=sup_h, line=dict(color="black", width=4), row=1, col=1)
            for h_line in np.linspace(-sup_h, sup_h, 5):
                dir_sign = -1 if x==0 else 1 # หันลายแรเงาออกด้านนอก
                fig.add_shape(type="line", x0=x, y0=h_line, x1=x + (sup_w*0.5*dir_sign), y1=h_line - (sup_h*0.2), line=dict(color="black", width=1), row=1, col=1)
            
    # วาด Loads (ใส่ลูกศร + ป้ายชื่อ Wu/Pu)
    for l in loads:
        if l['type'] == 'U':
            x1, x2 = cum_len[l['span_idx']], cum_len[l['span_idx']+1]
            h = (abs(l['w'])/max_load_val) * (viz_h * 0.6)
            # Area Load
            fig.add_trace(go.Scatter(x=[x1, x2, x2, x1], y=[0, 0, h, h], fill="toself", fillcolor="rgba(33, 150, 243, 0.3)", line_width=0, showlegend=False), row=1, col=1)
            # Top Line
            fig.add_trace(go.Scatter(x=[x1, x2], y=[h, h], mode='lines', line=dict(color="#1976D2", width=2), showlegend=False), row=1, col=1)
            # Label Wu
            fig.add_annotation(x=(x1+x2)/2, y=h, text=f"<b>Wu={l['w']:.0f}</b>", showarrow=False, yshift=15, font=dict(color="#0D47A1", size=12), row=1, col=1)
            
        elif l['type'] == 'P':
            px = cum_len[l['span_idx']] + l['x']
            h = (abs(l['P'])/max_load_val) * (viz_h * 0.6)
            if h == 0: h = viz_h * 0.2
            
            # Arrow & Label Pu
            fig.add_annotation(
                x=px, y=0,  # หัวลูกศรชี้ที่คาน
                ax=0, ay=-40, # หางลูกศรอยู่ข้างบน (pixels)
                text=f"<b>Pu={l['P']:.0f}</b>",
                arrowhead=2, arrowsize=1.5, arrowwidth=2, arrowcolor="#D32F2F",
                font=dict(color="#D32F2F", size=12),
                row=1, col=1
            )
            
    fig.update_yaxes(visible=False, range=[-sup_h*1.5, viz_h*1.2], row=1, col=1)
    
    # --- 2 & 3. SFD & BMD (ใส่ Label ระยะ x กลับมา) ---
    # Shear
    fig.add_trace(go.Scatter(x=df['x'], y=df['shear'], fill='tozeroy', line=dict(color='#D32F2F', width=2), fillcolor='rgba(211, 47, 47, 0.1)', name="Shear"), row=2, col=1)
    # Moment
    fig.add_trace(go.Scatter(x=df['x'], y=df['moment'], fill='tozeroy', line=dict(color='#1565C0', width=2), fillcolor='rgba(21, 101, 192, 0.1)', name="Moment"), row=3, col=1)
    
    # Label Logic (Max/Min with Distance)
    for col, row, color, unit in [('shear', 2, '#D32F2F', u_force), ('moment', 3, '#1565C0', f"{u_force}-{u_len}")]:
        arr = df[col].to_numpy()
        mx, mn = np.max(arr), np.min(arr)
        imx, imn = np.argmax(arr), np.argmin(arr)
        
        # Loop annotate ทั้งค่า Max และ Min
        for val, idx, pos in [(mx, imx, "top"), (mn, imn, "bottom")]:
            if abs(val) > 1e-3: # ไม่โชว์ถ้าเป็น 0
                ys = 25 if pos=="top" else -25
                x_pos = df['x'].iloc[idx]
                
                # HTML Label: Value บน, Location ล่าง
                label_txt = f"<b>{val:,.2f}</b><br><span style='font-size:10px'>@ {x_pos:.2f} m</span>"
                
                fig.add_annotation(
                    x=x_pos, y=val, 
                    text=label_txt, 
                    showarrow=False, 
                    bgcolor="rgba(255,255,255,0.85)", bordercolor=color, borderwidth=1, borderpad=3,
                    font=dict(color=color, size=11), yshift=ys, row=row, col=1
                )
        
        # ปรับ Scale แกน Y ให้ label ไม่ล้น
        rng = mx - mn
        if rng == 0: rng = 10
        pad = rng * 0.3
        fig.update_yaxes(range=[mn-pad, mx+pad], row=row, col=1)

    fig.update_layout(height=900, template="plotly_white", margin=dict(t=30, b=30, l=50, r=30), showlegend=False)
    fig.update_xaxes(showgrid=True, gridcolor='#ECEFF1', title=f"Distance ({u_len})", row=3, col=1)
    st.plotly_chart(fig, use_container_width=True)

def render_cross_section(b, h, cover, top_bars, bot_bars):
    fig = go.Figure()
    # Concrete
    fig.add_shape(type="rect", x0=0, y0=0, x1=b, y1=h, line=dict(color="black", width=2), fillcolor="#E0E0E0")
    # Stirrup
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
    fig.update_layout(width=180, height=220, margin=dict(l=5, r=5, t=5, b=5), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', showlegend=False)
    return fig

def render_design_results(df, params):
    st.markdown('<div class="section-header">4️⃣ Reinforced Concrete Design</div>', unsafe_allow_html=True)
    
    m_max = df['moment'].max()
    m_min = df['moment'].min()
    u_moment = f"{params['u_force']}-{params['u_len']}"
    
    if "SDM" in params['method']:
        res_pos = rc_design.calculate_flexure_sdm(m_max, "Mid-Span (+M)", params)
        res_neg = rc_design.calculate_flexure_sdm(m_min, "Support (-M)", params)
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"""
            <div class="card">
                <h4>{res_pos['Type']}</h4>
                <div style="font-size:1.5em; font-weight:bold; color:#1565C0;">{res_pos['Mu']:,.2f} <span style="font-size:0.6em">{u_moment}</span></div>
                <hr>
                <p>Req. As: <b>{res_pos['As_req']:,.2f} cm²</b></p>
                <p style="color:{'red' if 'Over' in res_pos['Status'] else 'green'}"><b>{res_pos['Status']}</b></p>
                <div style="background:#E3F2FD; padding:8px; border-radius:5px;"><b>Use:</b> {res_pos['Bars']}</div>
                <small style="color:gray">Logs: {', '.join(res_pos.get('Logs',[]))}</small>
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
                <p>Req. As: <b>{res_neg['As_req']:,.2f} cm²</b></p>
                <p style="color:{'red' if 'Over' in res_neg['Status'] else 'green'}"><b>{res_neg['Status']}</b></p>
                <div style="background:#FFEBEE; padding:8px; border-radius:5px;"><b>Use:</b> {res_neg['Bars']}</div>
                <small style="color:gray">Logs: {', '.join(res_neg.get('Logs',[]))}</small>
            </div>
            """, unsafe_allow_html=True)
            if "Over" not in res_neg['Status']:
                st.plotly_chart(render_cross_section(params['b'], params['h'], params['cv'], res_neg['Bars'], ""), use_container_width=True)
    else:
        st.info("WSD Method not implemented in this version.")

    st.markdown("---")
    st.markdown(f"#### Shear Design ({params['u_force']})")
    v_max = df['shear'].abs().max()
    vu_val, phi_vc = rc_design.calculate_shear_capacity(v_max, params)
    
    color = "green" if vu_val <= phi_vc/2 else ("orange" if vu_val <= phi_vc else "red")
    req_stirrup = "None" if vu_val <= phi_vc/2 else ("Min Stirrups" if vu_val <= phi_vc else "Design Stirrups Req.")
        
    st.markdown(f"""
    <div style="padding:15px; background-color:white; border-left:6px solid {color}; box-shadow: 0 2px 5px rgba(0,0,0,0.05);">
        <b>Max Shear Vu:</b> {vu_val:,.2f} {params['u_force']} <br>
        <b>Capacity ϕVc:</b> {phi_vc:,.2f} {params['u_force']} <br>
        <b>Status:</b> <span style="color:{color}; font-weight:bold">{req_stirrup}</span>
    </div>
    """, unsafe_allow_html=True)
