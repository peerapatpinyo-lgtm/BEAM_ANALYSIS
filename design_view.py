import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def draw_interactive_diagrams(df, reac, spans, sup_df, loads, unit_force="kg", unit_len="m"):
    # ==========================================
    # 0. DATA SANITIZATION (แก้ปัญหา Int + Str Error)
    # ==========================================
    try:
        # 1. บังคับแปลง Spans เป็นตัวเลขทั้งหมด (Float)
        if spans:
            spans = [float(s) for s in spans]
        
        # 2. บังคับแปลง DataFrame (ถ้ามาเป็น List)
        if isinstance(df, list): df = pd.DataFrame(df)
        if isinstance(sup_df, list): sup_df = pd.DataFrame(sup_df)

    except Exception as e:
        st.error(f"Data Error: {e}")
        return

    # --- 1. LOAD CALCULATION LIST (รายการคำนวณ) ---
    st.markdown("### 📋 Load Calculation List")
    
    clean_loads = [] # เก็บ Load ที่แปลงค่าแล้วเพื่อนำไปใช้ต่อ
    
    if loads:
        load_table_data = []
        for i, l in enumerate(loads):
            if isinstance(l, dict):
                try:
                    # แปลงค่าให้เป็นตัวเลขทันที (กัน Error)
                    span_idx = int(l.get('span_idx', 0))
                    mag = float(l.get('mag', 0))
                    x_pos = float(l.get('x', 0))
                    l_type = str(l.get('type', 'P'))
                    
                    # เก็บค่าที่ Clean แล้วไว้ใช้คำนวณกราฟ
                    clean_loads.append({
                        'span_idx': span_idx,
                        'mag': mag,
                        'x': x_pos,
                        'type': l_type
                    })

                    span_num = span_idx + 1
                    
                    if l_type == 'P':
                        type_lbl, mag_lbl = "Point Load (P)", f"{mag} {unit_force}"
                        pos_lbl = f"@ x = {x_pos:.2f} {unit_len} (Span {span_num})"
                    elif l_type == 'U':
                        type_lbl, mag_lbl = "Uniform Load (w)", f"{mag} {unit_force}/{unit_len}"
                        pos_lbl = f"Full Span {span_num}"
                    elif l_type == 'M':
                        type_lbl, mag_lbl = "Moment (M)", f"{mag} {unit_force}-{unit_len}"
                        pos_lbl = f"@ x = {x_pos:.2f} {unit_len} (Span {span_num})"
                    
                    load_table_data.append([i+1, type_lbl, mag_lbl, pos_lbl])
                except ValueError:
                    continue # ข้ามข้อมูลที่ผิดพลาด
            
        st.table(pd.DataFrame(load_table_data, columns=["No.", "Load Type", "Magnitude", "Position / Detail"]))
    else:
        st.info("No loads applied yet.")

    if df is None or df.empty: 
        return

    st.markdown("---")
    st.markdown("### 📊 Structural Analysis Diagrams")

    # --- 2. PREPARE PLOTTING DATA ---
    total_len = sum(spans)
    cum_spans = [0] + list(np.cumsum(spans))
    
    # Key Points for Grid Lines
    key_points = set()
    for x in cum_spans: key_points.add(round(x, 3))
    
    # ใช้ clean_loads ที่แปลงค่ามาแล้ว รับรองไม่ Error
    for l in clean_loads:
        abs_x = cum_spans[l['span_idx']] + l['x']
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
    if not sup_df.empty and 'id' in sup_df.columns and 'type' in sup_df.columns:
        sup_map = {int(r['id']): r['type'] for _, r in sup_df.iterrows()}
    else:
        sup_map = {}

    for i, x in enumerate(cum_spans):
        # Node Labels
        fig.add_annotation(
            x=x, y=0.35, text=f"Node {i+1}", showarrow=False,
            font=dict(size=10, color="gray"), row=1, col=1
        )
        
        if i in sup_map:
            stype = sup_map[i]
            if stype == 'Fixed':
                fig.add_shape(type="line", x0=x, y0=-0.25, x1=x, y1=0.25, line=dict(width=5, color='black'), row=1, col=1)
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

    # Loads Arrows (ใช้ clean_loads)
    for l in clean_loads:
        x_abs = cum_spans[l['span_idx']] + l['x']
        mag = l['mag']
        l_type = l['type']
        
        if l_type == 'P':
            fig.add_annotation(x=x_abs, y=0, ax=0, ay=-40, arrowhead=2, arrowwidth=2, arrowcolor='red', text=f"P={mag}", font=dict(color='red', size=10), row=1, col=1)
        elif l_type == 'U':
            xs = cum_spans[l['span_idx']]
            xe = cum_spans[l['span_idx']+1]
            fig.add_shape(type="rect", x0=xs, x1=xe, y0=0.05, y1=0.15, fillcolor="red", opacity=0.15, line_width=0, row=1, col=1)
            fig.add_annotation(x=(xs+xe)/2, y=0.2, text=f"w={mag}", showarrow=False, font=dict(color="red", size=10), row=1, col=1)
        elif l_type == 'M':
            fig.add_annotation(x=x_abs, y=0, text=f"M={mag}", showarrow=True, arrowhead=1, ax=0, ay=-30, arrowcolor='purple', font=dict(size=10), row=1, col=1)

    # ==========================================
    # HELPER: LABELS
    # ==========================================
    def add_eng_labels(x_data, y_data, row_idx, color_code, unit_suffix):
        y_arr = np.array(y_data, dtype=float) # Force float numpy array
        x_arr = np.array(x_data, dtype=float)
        
        if len(y_arr) == 0: return

        max_idx = np.argmax(y_arr)
        min_idx = np.argmin(y_arr)
        
        style = dict(
            showarrow=False,
            font=dict(color=color_code, size=11),
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor=color_code, borderwidth=1, borderpad=3
        )

        def plot_lbl(val, x_pos, is_top):
            lbl_type = "Max" if is_top else "Min"
            txt = f"<b>{lbl_type}:</b> {val:.2f} {unit_suffix}<br>@ {x_pos:.2f}m"
            shift = 25 if is_top else -25
            fig.add_annotation(x=x_pos, y=val, text=txt, yshift=shift, row=row_idx, col=1, **style)

        if abs(y_arr[max_idx]) > 1e-4:
            plot_lbl(y_arr[max_idx], x_arr[max_idx], True)

        if abs(y_arr[min_idx]) > 1e-4 and abs(x_arr[max_idx] - x_arr[min_idx]) > 0.05:
            plot_lbl(y_arr[min_idx], x_arr[min_idx], False)

    # ==========================================
    # PLOT GRAPHS
    # ==========================================
    # Shear
    c_shear = '#E67E22'
    fig.add_trace(go.Scatter(x=df['x'], y=df['shear'], fill='tozeroy', line=dict(color=c_shear, width=2), name="Shear"), row=2, col=1)
    add_eng_labels(df['x'], df['shear'], 2, c_shear, unit_force)

    # Moment
    c_moment = '#2980B9'
    fig.add_trace(go.Scatter(x=df['x'], y=df['moment'], fill='tozeroy', line=dict(color=c_moment, width=2), name="Moment"), row=3, col=1)
    add_eng_labels(df['x'], df['moment'], 3, c_moment, f"{unit_force}-{unit_len}")

    # Deflection
    c_defl = '#27AE60'
    fig.add_trace(go.Scatter(x=df['x'], y=df['deflection'], fill='tozeroy', line=dict(color=c_defl, width=2), name="Deflection"), row=4, col=1)
    
    if not df.empty:
        abs_d_idx = df['deflection'].abs().idxmax()
        d_val = float(df.iloc[abs_d_idx]['deflection'])
        d_x = float(df.iloc[abs_d_idx]['x'])
        
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
    # LAYOUT
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
    
    fig.update_yaxes(visible=False, row=1, col=1)
    fig.update_yaxes(title_text=f"V ({unit_force})", row=2, col=1, showgrid=True, gridcolor='#EEE')
    fig.update_yaxes(title_text=f"M ({unit_force}-{unit_len})", row=3, col=1, showgrid=True, gridcolor='#EEE')
    fig.update_yaxes(title_text=f"δ ({unit_len})", row=4, col=1, showgrid=True, gridcolor='#EEE')
    fig.update_xaxes(title_text=f"Distance ({unit_len})", row=4, col=1, showgrid=True, gridcolor='#EEE')

    st.plotly_chart(fig, use_container_width=True)

    # --- 3. REACTION TABLE ---
    st.markdown("### 📍 Support Reactions")
    reac_data = []
    
    expected_size = (len(spans) + 1) * 2
    
    # ตรวจสอบว่า reac เป็น list/array และขนาดถูกต้อง
    is_valid_reac = reac is not None and (isinstance(reac, list) or isinstance(reac, np.ndarray)) and len(reac) >= expected_size

    if is_valid_reac:
        for i in range(len(spans)+1):
            try:
                # แปลงค่า reac เป็น float ให้ชัวร์
                ry = float(reac[2*i])
                mz = float(reac[2*i+1])
                
                if abs(ry) > 1e-4 or abs(mz) > 1e-4 or i in sup_map:
                    reac_data.append({
                        "Node": i+1, 
                        f"Vertical (Ry) [{unit_force}]": f"{ry:.2f}", 
                        f"Moment (Mz) [{unit_force}-{unit_len}]": f"{mz:.2f}"
                    })
            except Exception:
                continue
        
        if reac_data:
            st.table(pd.DataFrame(reac_data))
        else:
            st.write("No significant reactions.")
    else:
        st.warning("⚠️ Calculation results for reactions are unavailable.")
