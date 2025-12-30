import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def draw_interactive_diagrams(df, reac, spans, sup_df, loads, unit_force="kg", unit_len="m"):
    # --- 0. ROBUST DATA CONVERSION (กัน Error string indices) ---
    # แปลงข้อมูลให้เป็น DataFrame เสมอ ป้องกันปัญหา Type Mismatch
    if isinstance(df, list): df = pd.DataFrame(df)
    if isinstance(sup_df, list): sup_df = pd.DataFrame(sup_df)
    
    # --- 1. DETAILED LOAD CALCULATION LIST ---
    st.markdown("### 📋 Load Calculation List")
    
    if loads:
        load_data = []
        for i, l in enumerate(loads):
            # ตรวจสอบ Type ของ l เผื่อเป็น String (กัน Error)
            if isinstance(l, dict):
                span_idx = int(l.get('span_idx', 0))
                span_num = span_idx + 1
                mag = l.get('mag', 0)
                x_pos = l.get('x', 0)
                l_type = l.get('type', 'P')

                if l_type == 'P':
                    type_lbl, mag_lbl = "Point Load (P)", f"{mag} {unit_force}"
                    pos_lbl = f"@ x = {x_pos:.2f} {unit_len} (Span {span_num})"
                elif l_type == 'U':
                    type_lbl, mag_lbl = "Uniform Load (w)", f"{mag} {unit_force}/{unit_len}"
                    pos_lbl = f"Full Span {span_num}"
                elif l_type == 'M':
                    type_lbl, mag_lbl = "Moment (M)", f"{mag} {unit_force}-{unit_len}"
                    pos_lbl = f"@ x = {x_pos:.2f} {unit_len} (Span {span_num})"
                
                load_data.append([i+1, type_lbl, mag_lbl, pos_lbl])
            
        df_loads = pd.DataFrame(load_data, columns=["No.", "Load Type", "Magnitude", "Position / Detail"])
        st.table(df_loads)
    else:
        st.info("No loads applied yet.")

    # ถ้าไม่มีข้อมูลกราฟ ให้จบการทำงานแค่นี้
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
        if isinstance(l, dict):
            abs_x = cum_spans[int(l.get('span_idx', 0))] + l.get('x', 0)
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
    # ROW 1: STRUCTURE (FBD)
    # ==========================================
    # Beam Line
    fig.add_trace(go.Scatter(x=[0, total_len], y=[0, 0], line=dict(color='black', width=4), hoverinfo='skip'), row=1, col=1)
    
    # Supports
    # ใช้ .get เพื่อความปลอดภัยกรณี key หาย
    if not sup_df.empty and 'id' in sup_df.columns and 'type' in sup_df.columns:
        sup_map = {int(r['id']): r['type'] for _, r in sup_df.iterrows()}
    else:
        sup_map = {}

    for i, x in enumerate(cum_spans):
        # Node Labels
        fig.add_annotation(
            x=x, y=0.3, text=f"Node {i+1}", showarrow=False,
            font=dict(size=10, color="gray"), row=1, col=1
        )
        
        if i in sup_map:
            stype = sup_map[i]
            if stype == 'Fixed':
                fig.add_shape(type="line", x0=x, y0=-0.2, x1=x, y1=0.2, line=dict(width=5, color='black'), row=1, col=1)
            else:
                sym = 'triangle-up' if stype == 'Pin' else 'circle'
                fig.add_trace(go.Scatter(
                    x=[x], y=[0], 
                    mode='markers', 
                    marker=dict(symbol=sym, size=14, color='white', line=dict(color='black', width=2)), 
                    name=stype, hoverinfo='name'
                ), row=1, col=1)
                
                if stype == 'Roller':
                    fig.add_shape(type="line", x0=x-0.1, y0=-0.08, x1=x+0.1, y1=-0.08, line=dict(width=2, color='black'), row=1, col=1)

    # Loads Arrows
    for l in loads:
        if isinstance(l, dict):
            x_abs = cum_spans[int(l.get('span_idx', 0))] + l.get('x', 0)
            mag = l.get('mag', 0)
            l_type = l.get('type', 'P')
            
            if l_type == 'P':
                fig.add_annotation(x=x_abs, y=0, ax=0, ay=-40, arrowhead=2, arrowwidth=2, arrowcolor='red', text=f"P={mag}", font=dict(color='red', size=10), row=1, col=1)
            elif l_type == 'U':
                xs = cum_spans[int(l.get('span_idx', 0))]
                xe = cum_spans[int(l.get('span_idx', 0))+1]
                fig.add_shape(type="rect", x0=xs, x1=xe, y0=0.05, y1=0.15, fillcolor="red", opacity=0.15, line_width=0, row=1, col=1)
                fig.add_annotation(x=(xs+xe)/2, y=0.2, text=f"w={mag}", showarrow=False, font=dict(color="red", size=10), row=1, col=1)
            elif l_type == 'M':
                fig.add_annotation(x=x_abs, y=0, text=f"M={mag}", showarrow=True, arrowhead=1, ax=0, ay=-30, arrowcolor='purple', font=dict(size=10), row=1, col=1)

    # ==========================================
    # HELPER: Labels with UNITS & Clean Style
    # ==========================================
    def add_eng_labels(x_data, y_data, row_idx, color_code, unit_suffix):
        # แปลงเป็น numpy array เพื่อความชัวร์
        y_arr = np.array(y_data)
        x_arr = np.array(x_data)
        if len(y_arr) == 0: return

        # หาค่า Max/Min
        max_idx = np.argmax(y_arr)
        min_idx = np.argmin(y_arr)
        
        # Style (กล่องข้อความ ไม่มีลูกศร)
        style = dict(
            showarrow=False,
            font=dict(color=color_code, size=11),
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor=color_code, borderwidth=1, borderpad=3
        )

        def plot_lbl(val, x_pos, is_top):
            label_txt = "Max" if is_top else "Min"
            txt = f"<b>{label_txt}:</b> {val:.2f} {unit_suffix}<br>@ {x_pos:.2f}m"
            shift = 25 if is_top else -25
            fig.add_annotation(
                x=x_pos, y=val,
                text=txt,
                yshift=shift,
                row=row_idx, col=1, **style
            )

        # Plot Max
        if abs(y_arr[max_idx]) > 1e-4:
            plot_lbl(y_arr[max_idx], x_arr[max_idx], True)

        # Plot Min (ถ้าค่าต่างจาก Max อย่างมีนัยยะ)
        if abs(y_arr[min_idx]) > 1e-4 and abs(x_arr[max_idx] - x_arr[min_idx]) > 0.05:
            plot_lbl(y_arr[min_idx], x_arr[min_idx], False)

    # ==========================================
    # ROW 2: SHEAR (V)
    # ==========================================
    c_shear = '#E67E22'
    fig.add_trace(go.Scatter(x=df['x'], y=df['shear'], fill='tozeroy', line=dict(color=c_shear, width=2), name="Shear"), row=2, col=1)
    add_eng_labels(df['x'], df['shear'], 2, c_shear, unit_force)

    # ==========================================
    # ROW 3: MOMENT (M)
    # ==========================================
    c_moment = '#2980B9'
    fig.add_trace(go.Scatter(x=df['x'], y=df['moment'], fill='tozeroy', line=dict(color=c_moment, width=2), name="Moment"), row=3, col=1)
    add_eng_labels(df['x'], df['moment'], 3, c_moment, f"{unit_force}-{unit_len}")

    # ==========================================
    # ROW 4: DEFLECTION
    # ==========================================
    c_defl = '#27AE60'
    fig.add_trace(go.Scatter(x=df['x'], y=df['deflection'], fill='tozeroy', line=dict(color=c_defl, width=2), name="Deflection"), row=4, col=1)
    
    # Label เฉพาะค่า Abs Max ของ Deflection
    if not df.empty:
        abs_d_idx = df['deflection'].abs().idxmax()
        d_val = df.iloc[abs_d_idx]['deflection'] # ใช้ iloc ปลอดภัยกว่า
        d_x = df.iloc[abs_d_idx]['x']
        
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
    # LAYOUT CONFIG
    # ==========================================
    # เส้น Drop Line (เส้นประแนวดิ่ง)
    for kp in sorted_keys:
        fig.add_vline(x=kp, line_width=1, line_dash="dash", line_color="gray", opacity=0.4)

    fig.update_layout(
        height=1000, 
        showlegend=False,
        template="simple_white",
        margin=dict(l=60, r=20, t=40, b=50),
        font=dict(family="Roboto, Arial", size=12)
    )
    
    # Axis Titles & Grids
    fig.update_yaxes(visible=False, row=1, col=1)
    fig.update_yaxes(title_text=f"V ({unit_force})", row=2, col=1, showgrid=True, gridcolor='#EEE')
    fig.update_yaxes(title_text=f"M ({unit_force}-{unit_len})", row=3, col=1, showgrid=True, gridcolor='#EEE')
    fig.update_yaxes(title_text=f"δ ({unit_len})", row=4, col=1, showgrid=True, gridcolor='#EEE')
    fig.update_xaxes(title_text=f"Distance ({unit_len})", row=4, col=1, showgrid=True, gridcolor='#EEE')

    st.plotly_chart(fig, use_container_width=True)

    # --- 3. REACTION TABLE ---
    st.markdown("### 📍 Support Reactions")
    reac_data = []
    
    # คำนวณขนาดที่ควรจะเป็นของ reac array
    expected_size = (len(spans) + 1) * 2
    
    # ตรวจสอบว่า reac มีข้อมูลและขนาดถูกต้อง
    if reac is not None and (isinstance(reac, list) or isinstance(reac, np.ndarray)) and len(reac) >= expected_size:
        for i in range(len(spans)+1):
            try:
                ry = reac[2*i]
                mz = reac[2*i+1]
                
                # แสดงผลถ้ามีค่าแรง หรือเป็นจุด Support
                if abs(ry) > 1e-4 or abs(mz) > 1e-4 or i in sup_map:
                    reac_data.append({
                        "Node": i+1, 
                        f"Vertical (Ry) [{unit_force}]": f"{ry:.2f}", 
                        f"Moment (Mz) [{unit_force}-{unit_len}]": f"{mz:.2f}"
                    })
            except IndexError:
                continue
        
        if reac_data:
            st.table(pd.DataFrame(reac_data))
        else:
            st.write("No significant reactions.")
    else:
        # กรณี reac มาผิด format หรือเป็น None
        st.warning("⚠️ Calculation results for reactions are unavailable.")
