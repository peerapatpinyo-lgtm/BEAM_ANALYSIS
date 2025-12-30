import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# เพิ่มตัวแปร reac เข้ามาใน arguments เพื่อแก้ Error
def draw_interactive_diagrams(df, reac, spans, sup_df, loads, unit_force="kg", unit_len="m"):
    
    # --- 1. DETAILED LOAD COMBINATION LIST (กู้คืนรายการคำนวณละเอียด) ---
    st.markdown("### 📋 Load Calculation List")
    
    if loads:
        load_data = []
        for i, l in enumerate(loads):
            span_idx = int(l['span_idx'])
            span_num = span_idx + 1
            
            if l['type'] == 'P':
                type_lbl = "Point Load (P)"
                mag_lbl = f"{l['mag']} {unit_force}"
                # ระบุตำแหน่งละเอียด
                pos_lbl = f"@ x = {l['x']:.2f} {unit_len} (on Span {span_num})"
            elif l['type'] == 'U':
                type_lbl = "Uniform Load (w)"
                mag_lbl = f"{l['mag']} {unit_force}/{unit_len}"
                pos_lbl = f"Full Length (Span {span_num})"
            elif l['type'] == 'M':
                type_lbl = "Moment (M)"
                mag_lbl = f"{l['mag']} {unit_force}-{unit_len}"
                pos_lbl = f"@ x = {l['x']:.2f} {unit_len} (on Span {span_num})"
            
            load_data.append([i+1, type_lbl, mag_lbl, pos_lbl])
            
        df_loads = pd.DataFrame(load_data, columns=["No.", "Load Type", "Magnitude", "Position / Detail"])
        st.table(df_loads)
    else:
        st.info("No loads applied yet.")

    if df is None or df.empty: 
        return

    st.markdown("---")
    st.markdown("### 📊 Structural Analysis Diagrams")

    # --- 2. PREPARE PLOTTING DATA ---
    total_len = sum(spans)
    cum_spans = [0] + list(np.cumsum(spans))
    
    # Key Points for Grid Lines (Supports + Loads)
    key_points = set()
    for x in cum_spans: key_points.add(round(x, 3))
    for l in loads:
        abs_x = cum_spans[int(l['span_idx'])] + l['x']
        key_points.add(round(abs_x, 3))
    sorted_keys = sorted(list(key_points))

    # Create Subplots
    fig = make_subplots(
        rows=4, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.08,
        subplot_titles=(
            "<b>Structure Model (FBD)</b>", 
            f"<b>Shear Force Diagram (SFD)</b>", 
            f"<b>Bending Moment Diagram (BMD)</b>", 
            f"<b>Deflection (δ)</b>"
        ),
        row_heights=[0.15, 0.28, 0.28, 0.29]
    )

    # ==========================================
    # ROW 1: STRUCTURE (FBD) - ปรับสัดส่วน Support
    # ==========================================
    # Beam Line
    fig.add_trace(go.Scatter(x=[0, total_len], y=[0, 0], line=dict(color='black', width=4), hoverinfo='skip'), row=1, col=1)
    
    # Supports
    sup_map = {int(r['id']): r['type'] for _, r in sup_df.iterrows()}
    for i, x in enumerate(cum_spans):
        # Node Labels
        fig.add_annotation(
            x=x, y=0.3, text=f"Node {i+1}", showarrow=False,
            font=dict(size=10, color="gray"), row=1, col=1
        )
        
        if i in sup_map:
            stype = sup_map[i]
            if stype == 'Fixed':
                # ปรับเส้น Fixed ให้สมส่วน (ไม่ใหญ่เวอร์)
                fig.add_shape(type="line", x0=x, y0=-0.2, x1=x, y1=0.2, line=dict(width=5, color='black'), row=1, col=1)
            else:
                sym = 'triangle-up' if stype == 'Pin' else 'circle'
                # ปรับขนาด Marker (size=14) ให้พอดีกับเส้นคาน width=4
                fig.add_trace(go.Scatter(
                    x=[x], y=[0], 
                    mode='markers', 
                    marker=dict(symbol=sym, size=14, color='white', line=dict(color='black', width=2)), 
                    name=stype, hoverinfo='name'
                ), row=1, col=1)
                
                # ถ้าเป็น Roller ให้วาดเส้นพื้นรองรับด้านล่างเล็กน้อย
                if stype == 'Roller':
                    fig.add_shape(type="line", x0=x-0.1, y0=-0.08, x1=x+0.1, y1=-0.08, line=dict(width=2, color='black'), row=1, col=1)

    # Loads Arrows
    for l in loads:
        x_abs = cum_spans[int(l['span_idx'])] + l['x']
        if l['type'] == 'P':
            fig.add_annotation(x=x_abs, y=0, ax=0, ay=-40, arrowhead=2, arrowwidth=2, arrowcolor='red', text=f"P={l['mag']}", font=dict(color='red', size=10), row=1, col=1)
        elif l['type'] == 'U':
             xs, xe = cum_spans[int(l['span_idx'])], cum_spans[int(l['span_idx'])+1]
             fig.add_shape(type="rect", x0=xs, x1=xe, y0=0.05, y1=0.15, fillcolor="red", opacity=0.15, line_width=0, row=1, col=1)
             fig.add_annotation(x=(xs+xe)/2, y=0.2, text=f"w={l['mag']}", showarrow=False, font=dict(color="red", size=10), row=1, col=1)
        elif l['type'] == 'M':
            fig.add_annotation(x=x_abs, y=0, text=f"M={l['mag']}", showarrow=True, arrowhead=1, ax=0, ay=-30, arrowcolor='purple', font=dict(size=10), row=1, col=1)

    # ==========================================
    # HELPER: Labels with UNITS
    # ==========================================
    def add_eng_labels(x_data, y_data, row_idx, color_code, unit_suffix):
        y_arr = np.array(y_data)
        if len(y_arr) == 0: return

        # Global Max/Min
        max_idx = np.argmax(y_arr)
        min_idx = np.argmin(y_arr)
        
        # Style Box
        style = dict(
            showarrow=False,
            font=dict(color=color_code, size=11),
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor=color_code, borderwidth=1, borderpad=3
        )

        # Draw Label Function
        def plot_lbl(val, x_pos, is_top):
            # ใส่หน่วย (Unit) ใน Label
            txt = f"<b>Max:</b> {val:.2f} {unit_suffix}<br>@ {x_pos:.2f}m" if is_top else f"<b>Min:</b> {val:.2f} {unit_suffix}<br>@ {x_pos:.2f}m"
            shift = 20 if is_top else -20
            fig.add_annotation(
                x=x_pos, y=val,
                text=txt,
                yshift=shift,
                row=row_idx, col=1, **style
            )

        # Plot Max
        if abs(y_arr[max_idx]) > 1e-4:
            plot_lbl(y_arr[max_idx], x_data[max_idx], True)

        # Plot Min
        if abs(y_arr[min_idx]) > 1e-4 and abs(x_data[max_idx] - x_data[min_idx]) > 0.05:
            plot_lbl(y_arr[min_idx], x_data[min_idx], False)

    # ==========================================
    # ROW 2: SHEAR (V)
    # ==========================================
    c_shear = '#E67E22'
    fig.add_trace(go.Scatter(x=df['x'], y=df['shear'], fill='tozeroy', line=dict(color=c_shear, width=2), name="Shear"), row=2, col=1)
    # ใส่หน่วยให้ถูกต้องตามตัวแปร
    add_eng_labels(df['x'], df['shear'], 2, c_shear, unit_force)

    # ==========================================
    # ROW 3: MOMENT (M)
    # ==========================================
    c_moment = '#2980B9'
    fig.add_trace(go.Scatter(x=df['x'], y=df['moment'], fill='tozeroy', line=dict(color=c_moment, width=2), name="Moment"), row=3, col=1)
    # หน่วย Moment = Force-Length
    add_eng_labels(df['x'], df['moment'], 3, c_moment, f"{unit_force}-{unit_len}")

    # ==========================================
    # ROW 4: DEFLECTION
    # ==========================================
    c_defl = '#27AE60'
    fig.add_trace(go.Scatter(x=df['x'], y=df['deflection'], fill='tozeroy', line=dict(color=c_defl, width=2), name="Deflection"), row=4, col=1)
    
    # Deflection Label
    abs_d_idx = df['deflection'].abs().idxmax()
    d_val = df.loc[abs_d_idx, 'deflection']
    d_x = df.loc[abs_d_idx, 'x']
    if abs(d_val) > 1e-9:
        fig.add_annotation(
            x=d_x, y=d_val,
            text=f"<b>Max:</b> {d_val:.4f} {unit_len}<br>@ {d_x:.2f}m",
            showarrow=False,
            yshift=25 if d_val > 0 else -25,
            font=dict(color=c_defl, size=11),
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor=c_defl, borderwidth=1, borderpad=3,
            row=4, col=1
        )

    # ==========================================
    # LAYOUT & AXIS TITLES
    # ==========================================
    for kp in sorted_keys:
        fig.add_vline(x=kp, line_width=1, line_dash="dash", line_color="gray", opacity=0.4)

    fig.update_layout(
        height=1000, 
        showlegend=False,
        template="simple_white",
        margin=dict(l=60, r=20, t=40, b=50),
        font=dict(family="Roboto, Arial", size=12)
    )
    
    # Axis Titles (Label แกน Y X กลับมาแล้ว)
    fig.update_yaxes(visible=False, row=1, col=1)
    fig.update_yaxes(title_text=f"V ({unit_force})", row=2, col=1, showgrid=True, gridcolor='#EEE')
    fig.update_yaxes(title_text=f"M ({unit_force}-{unit_len})", row=3, col=1, showgrid=True, gridcolor='#EEE')
    fig.update_yaxes(title_text=f"δ ({unit_len})", row=4, col=1, showgrid=True, gridcolor='#EEE')
    
    # Axis X Title (Bottom only)
    fig.update_xaxes(title_text=f"Distance along beam ({unit_len})", row=4, col=1, showgrid=True, gridcolor='#EEE')

    st.plotly_chart(fig, use_container_width=True)

    # --- 3. REACTION TABLE (แก้ไขให้ใช้ตัวแปร reac ได้ถูกต้อง) ---
    st.markdown("### 📍 Support Reactions")
    reac_data = []
    
    # ตรวจสอบว่า reac มีค่าและขนาดถูกต้องหรือไม่
    expected_size = (len(spans) + 1) * 2
    if reac is not None and len(reac) >= expected_size:
        for i in range(len(spans)+1):
            ry = reac[2*i]
            mz = reac[2*i+1]
            
            # โชว์เฉพาะจุดที่มีแรงปฏิกิริยา (หรือเป็น Support)
            if abs(ry) > 1e-4 or abs(mz) > 1e-4 or i in sup_map:
                reac_data.append({
                    "Node": i+1, 
                    f"Vertical (Ry) [{unit_force}]": f"{ry:.2f}", 
                    f"Moment (Mz) [{unit_force}-{unit_len}]": f"{mz:.2f}"
                })
        
        if reac_data:
            st.table(pd.DataFrame(reac_data))
        else:
            st.write("No significant reactions.")
    else:
        st.error("Error: Reaction data (reac) unavailable or size mismatch.")
