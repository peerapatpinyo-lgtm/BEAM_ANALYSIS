import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def draw_interactive_diagrams(df, spans, sup_df, loads, unit_force, unit_len):
    # --- 1. LOAD COMBINATION / CALCULATION LIST (นำกลับมาแสดงเป็นอันดับแรก) ---
    st.markdown("### 📋 Load Calculation List")
    
    if loads:
        load_data = []
        for i, l in enumerate(loads):
            if l['type'] == 'P':
                desc = "Point Load"
                val_str = f"{l['mag']} {unit_force}"
                pos_str = f"@ x = {l['x']} {unit_len} (Span {l['span_idx']+1})"
            elif l['type'] == 'U':
                desc = "Uniform Load"
                val_str = f"{l['mag']} {unit_force}/{unit_len}"
                pos_str = f"Full Span {l['span_idx']+1}"
            elif l['type'] == 'M':
                desc = "Moment Load"
                val_str = f"{l['mag']} {unit_force}-{unit_len}"
                pos_str = f"@ x = {l['x']} {unit_len} (Span {l['span_idx']+1})"
            
            load_data.append([i+1, desc, val_str, pos_str])
            
        df_load_list = pd.DataFrame(load_data, columns=["No.", "Type", "Magnitude", "Position"])
        st.table(df_load_list)
    else:
        st.info("No loads applied.")

    # ถ้าไม่มีผลลัพธ์การคำนวณ (df) ให้จบการทำงานตรงนี้ (แสดงแค่ตาราง Load)
    if df is None or df.empty:
        return

    st.markdown("---")
    st.markdown("### 📊 Analysis Diagrams")

    # --- 2. PREPARE DATA FOR PLOTTING ---
    total_len = sum(spans)
    cum_spans = [0] + list(np.cumsum(spans))
    
    # รวบรวมจุดสำคัญ (Key Points) เพื่อวาดเส้นประแนวดิ่ง (Support + Loads)
    # ใช้ set และ round เพื่อป้องกันจุดซ้ำซ้อน (เช่น load ทับ support)
    key_points = set()
    for x in cum_spans: key_points.add(round(x, 3)) # Nodes
    for l in loads:
        abs_x = cum_spans[int(l['span_idx'])] + l['x']
        key_points.add(round(abs_x, 3)) # Load locations
    
    sorted_keys = sorted(list(key_points))

    # สร้าง Subplots
    fig = make_subplots(
        rows=4, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.05,
        subplot_titles=("Structure Model (FBD)", f"Shear Force ({unit_force})", f"Bending Moment ({unit_force}-{unit_len})", f"Deflection ({unit_len})"),
        row_heights=[0.2, 0.25, 0.25, 0.3]
    )

    # ==========================================
    # ROW 1: STRUCTURE (FBD)
    # ==========================================
    # 1.1 Beam Line (ย้ายมาที่ y=0)
    fig.add_trace(go.Scatter(x=[0, total_len], y=[0, 0], line=dict(color='black', width=5), hoverinfo='skip'), row=1, col=1)
    
    # 1.2 Supports & Node Labels
    sup_map = {int(r['id']): r['type'] for _, r in sup_df.iterrows()}
    
    for i, x in enumerate(cum_spans):
        # วาด Node Label (เช่น Node 1, Node 2)
        fig.add_annotation(
            x=x, y=0.3, # อยู่เหนือคานนิดหน่อย
            text=f"<b>Node {i+1}</b>",
            showarrow=False,
            font=dict(size=10, color="gray"),
            row=1, col=1
        )
        
        if i in sup_map:
            stype = sup_map[i]
            # เลือกสัญลักษณ์
            if stype == 'Pin': 
                sym = 'triangle-up'; offset = -0.15 # ปรับตำแหน่งให้แตะคาน
            elif stype == 'Roller': 
                sym = 'circle'; offset = -0.15
            else: 
                sym = 'square'; offset = 0
            
            # วาด Support (ขยับ y ให้แตะเส้น 0)
            if stype == 'Fixed':
                fig.add_shape(type="line", x0=x, y0=-0.2, x1=x, y1=0.2, line=dict(width=6, color='black'), row=1, col=1)
            else:
                # Plot marker at y=0 but visualize slightly below or exactly at 0
                fig.add_trace(go.Scatter(
                    x=[x], y=[0], # Plot ที่ 0 เลยจะได้ไม่ลอย
                    mode='markers', 
                    marker=dict(symbol=sym, size=18, color='white', line=dict(color='black', width=2)), 
                    name=stype, hovertemplate=f"{stype} Support"
                ), row=1, col=1)

    # 1.3 Loads
    for l in loads:
        x_abs = cum_spans[int(l['span_idx'])] + l['x']
        if l['type'] == 'P':
            # ลูกศรชี้ลงแตะที่ y=0
            fig.add_annotation(x=x_abs, y=0, ax=0, ay=-40, arrowhead=2, text=f"P={l['mag']}", row=1, col=1)
        elif l['type'] == 'M':
            fig.add_annotation(x=x_abs, y=0, text=f"M={l['mag']}", showarrow=True, arrowhead=1, ax=0, ay=-30, arrowwidth=2, arrowcolor='purple', row=1, col=1)
            fig.add_trace(go.Scatter(x=[x_abs], y=[0], mode='markers', marker=dict(symbol='star', size=12, color='purple'), showlegend=False, hoverinfo='skip'), row=1, col=1)
        elif l['type'] == 'U':
             xs = cum_spans[int(l['span_idx'])]
             xe = cum_spans[int(l['span_idx'])+1]
             # UDL วาดเป็นกล่องสี่เหลี่ยมเหนือคาน
             fig.add_shape(type="rect", x0=xs, x1=xe, y0=0.05, y1=0.15, fillcolor="red", opacity=0.2, line_width=0, row=1, col=1)
             fig.add_annotation(x=(xs+xe)/2, y=0.18, text=f"w={l['mag']}", showarrow=False, font=dict(color="red"), row=1, col=1)

    # ==========================================
    # HELPER: Function to add Max/Min Annotations
    # ==========================================
    def add_max_min_labels(x_data, y_data, row_idx, unit=""):
        # Max Positive
        max_idx = y_data.idxmax()
        max_val = y_data[max_idx]
        max_x = x_data[max_idx]
        
        # Min Negative (or absolute min)
        min_idx = y_data.idxmin()
        min_val = y_data[min_idx]
        min_x = x_data[min_idx]

        # Annotate Max
        if abs(max_val) > 1e-5:
            fig.add_annotation(
                x=max_x, y=max_val,
                text=f"Max: {max_val:.2f}\n@ {max_x:.2f}m",
                showarrow=True, arrowhead=1, ax=0, ay=-30 if max_val > 0 else 30,
                font=dict(color="red", size=10),
                row=row_idx, col=1
            )
        
        # Annotate Min (Only if it's different enough or signiticant)
        if abs(min_val) > 1e-5 and abs(max_x - min_x) > 0.01:
            fig.add_annotation(
                x=min_x, y=min_val,
                text=f"Min: {min_val:.2f}\n@ {min_x:.2f}m",
                showarrow=True, arrowhead=1, ax=0, ay=30 if min_val < 0 else -30,
                font=dict(color="red", size=10),
                row=row_idx, col=1
            )

    # ==========================================
    # ROW 2: SHEAR (V)
    # ==========================================
    fig.add_trace(go.Scatter(x=df['x'], y=df['shear'], fill='tozeroy', line=dict(color='#D97706'), fillcolor='rgba(217, 119, 6, 0.1)', name="Shear"), row=2, col=1)
    add_max_min_labels(df['x'], df['shear'], 2)

    # ==========================================
    # ROW 3: MOMENT (M)
    # ==========================================
    fig.add_trace(go.Scatter(x=df['x'], y=df['moment'], fill='tozeroy', line=dict(color='#2563EB'), fillcolor='rgba(37, 99, 235, 0.1)', name="Moment"), row=3, col=1)
    add_max_min_labels(df['x'], df['moment'], 3)

    # ==========================================
    # ROW 4: DEFLECTION
    # ==========================================
    fig.add_trace(go.Scatter(x=df['x'], y=df['deflection'], fill='tozeroy', line=dict(color='#10B981'), fillcolor='rgba(16, 185, 129, 0.1)', name="Deflection"), row=4, col=1)
    
    # For deflection, usually we care about the absolute max
    abs_defl = df['deflection'].abs()
    max_d_idx = abs_defl.idxmax()
    max_d_val = df['deflection'][max_d_idx]
    max_d_x = df['x'][max_d_idx]
    
    if abs(max_d_val) > 1e-9:
        fig.add_annotation(
            x=max_d_x, y=max_d_val,
            text=f"Max: {max_d_val:.2e}\n@ {max_d_x:.2f}m",
            showarrow=True, arrowhead=1, ax=0, ay=30 if max_d_val < 0 else -30,
            font=dict(color="red", size=10),
            row=4, col=1
        )

    # ==========================================
    # VERTICAL DROP LINES (เส้นประ)
    # ==========================================
    # ลากเส้นผ่านทุก Subplot (row='all') ที่ตำแหน่ง Key Points
    for kp in sorted_keys:
        fig.add_vline(x=kp, line_width=1, line_dash="dash", line_color="gray", opacity=0.4)

    # ==========================================
    # FINAL LAYOUT
    # ==========================================
    fig.update_layout(height=1000, showlegend=False, title_text="")
    
    # Set y-axes to be visible and clear
    fig.update_yaxes(title_text="Y", showgrid=False, zeroline=False, row=1, col=1, range=[-0.5, 0.5], visible=False) # Hide Y axis for FBD
    fig.update_yaxes(title_text=f"V", row=2, col=1)
    fig.update_yaxes(title_text=f"M", row=3, col=1)
    fig.update_yaxes(title_text="Defl", row=4, col=1)
    
    st.plotly_chart(fig, use_container_width=True)

def render_result_tables(df, reac, spans):
    st.subheader("📍 Support Reactions")
    reac_data = []
    for i in range(len(spans)+1):
        ry = reac[2*i]
        mz = reac[2*i+1]
        reac_data.append({
            "Node": i+1, 
            "Fy (Vertical)": f"{ry:.2f}", 
            "Mz (Moment)": f"{mz:.2f}"
        })
    st.table(pd.DataFrame(reac_data))
