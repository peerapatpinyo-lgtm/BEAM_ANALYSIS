# design_view.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from section_plotter import plot_longitudinal_section_detailed, plot_cross_section
from reporter import render_calculation_report

def calculate_boq_summary(design_res, spans):
    """
    คำนวณปริมาณงาน (BOQ) โดยประมาณจากผลการออกแบบ
    """
    total_concrete_vol = 0.0
    total_formwork_area = 0.0
    total_steel_weight = 0.0
    
    for i, res in enumerate(design_res):
        L = spans[i] # meters
        # ตรวจสอบค่า b, h ว่ามีค่าหรือไม่ ถ้าไม่มีให้ใช้ค่า default
        b_mm = res.get('b', 300)
        h_mm = res.get('h', 500)
        
        b_m = b_mm / 1000.0
        h_m = h_mm / 1000.0
        
        # 1. Concrete (m3)
        vol = b_m * h_m * L
        total_concrete_vol += vol
        
        # 2. Formwork (m2) - Sides + Bottom
        # Assumption: Beam sides + bottom (2h + b) * L
        form_area = (2 * h_m + b_m) * L
        total_formwork_area += form_area
        
        # 3. Steel Weight (kg) - Estimation
        # Main Bars weight -> Weight (kg/m) = d^2 / 162
        w_main = 0.0
        
        # Top Bars
        top_layers = res.get('top', {}).get('all_layers', [])
        for layer in top_layers:
            if layer['n'] > 0:
                unit_w = (layer['db']**2 / 162)
                # Assume bar length = Span length + anchorage (approx 10%)
                w_main += layer['n'] * unit_w * (L * 1.1)
                
        # Bottom Bars
        bot_layers = res.get('bot', {}).get('all_layers', [])
        for layer in bot_layers:
            if layer['n'] > 0:
                unit_w = (layer['db']**2 / 162)
                w_main += layer['n'] * unit_w * (L * 1.1)

        # Stirrups
        # Length per stirrup approx = 2*(b+h) (ignoring cover for quick calc)
        # Number = L / s
        shear_data = res.get('shear', {})
        stir_db = shear_data.get('db', 6)
        stir_s = shear_data.get('s', 200) / 1000.0 # convert mm to m
        
        if stir_s > 0:
            n_stir = int(L / stir_s) + 1
            len_stir = 2 * (b_m + h_m) 
            w_stir_unit = (stir_db**2 / 162)
            w_stir_total = n_stir * len_stir * w_stir_unit
        else:
            w_stir_total = 0
            
        span_steel = w_main + w_stir_total
        total_steel_weight += span_steel

    # Create Summary Data
    data = [
        {"Item": "Concrete Structure (240 ksc)", "Unit": "m3", "Quantity": float(f"{total_concrete_vol:.2f}")},
        {"Item": "Formwork (Beam sides & bottom)", "Unit": "m2", "Quantity": float(f"{total_formwork_area:.2f}")},
        {"Item": "Deformed Bars (DB) + Stirrups (RB)", "Unit": "kg", "Quantity": float(f"{total_steel_weight:.2f}")}
    ]
    
    return pd.DataFrame(data)

def plot_analysis_results(df, spans, sup_df, loads, reactions):
    """
    ฟังก์ชันสำหรับวาดกราฟ SFD, BMD, Deflection โดยใช้ Matplotlib
    (ฟังก์ชันนี้จำเป็นต้องมีเพื่อให้ app.py เดิมทำงานได้)
    """
    # Create Figure
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True, constrained_layout=True)
    
    x = df.iloc[:, 0] # Column 0: x
    M = df.iloc[:, 1] # Column 1: Moment
    V = df.iloc[:, 2] # Column 2: Shear
    D = df.iloc[:, 3] # Column 3: Deflection

    # 1. Bending Moment Diagram (BMD)
    # Note: RC convention usually plots positive moment (tension bottom) downwards, 
    # but here we plot standard sign. Fill between for clarity.
    ax1.plot(x, M/1000.0, color='#e74c3c', linewidth=2, label='Moment')
    ax1.fill_between(x, M/1000.0, 0, color='#e74c3c', alpha=0.1)
    ax1.set_ylabel("Moment (kNm)", fontweight='bold')
    ax1.set_title("Bending Moment Diagram (BMD)", fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.axhline(0, color='black', linewidth=1)

    # 2. Shear Force Diagram (SFD)
    ax2.plot(x, V/1000.0, color='#2980b9', linewidth=2, label='Shear')
    ax2.fill_between(x, V/1000.0, 0, color='#2980b9', alpha=0.1)
    ax2.set_ylabel("Shear (kN)", fontweight='bold')
    ax2.set_title("Shear Force Diagram (SFD)", fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.axhline(0, color='black', linewidth=1)

    # 3. Deflection
    ax3.plot(x, D*1000.0, color='#27ae60', linewidth=2, label='Deflection')
    ax3.fill_between(x, D*1000.0, 0, color='#27ae60', alpha=0.1)
    ax3.set_ylabel("Deflection (mm)", fontweight='bold')
    ax3.set_xlabel("Position (m)", fontweight='bold')
    ax3.set_title("Elastic Deflection", fontweight='bold')
    ax3.grid(True, linestyle='--', alpha=0.6)
    ax3.axhline(0, color='black', linewidth=1)
    
    # Invert Y for deflection to look natural (downward = negative)
    # If solver returns negative for downward, we can leave it or invert axis.
    # Usually structural apps invert Y for deflection.
    # ax3.invert_yaxis() 

    # Add Support Markers on x-axis
    for s in spans:
        pass # Logic handled by x-ticks usually
        
    return fig

def render_design_view(res_package):
    """
    Main View Controller: แสดงผลลัพธ์การออกแบบทั้งหมด
    """
    if not res_package:
        st.error("No design results to display.")
        return

    # Unpack Data
    x = res_package['x']
    m = res_package['m']
    v = res_package['v']
    d = res_package['d']
    react = res_package['reactions']
    design_res = res_package['design_results']
    spans = res_package['spans']
    sup_df = res_package['supports']
    params = res_package['params']
    
    st.markdown("## 🏗️ Design Results Dashboard")
    
    # Create Tabs
    t1, t2, t3 = st.tabs(["📊 Analysis & Diagrams", "📐 Section Details", "📝 Report & BOQ"])
    
    # ==========================================
    # TAB 1: Analysis (SFD, BMD, Deflection)
    # ==========================================
    with t1:
        st.subheader("1. Internal Forces Diagrams")
        
        # Reuse the plotter function or create simple line charts
        # Since app.py might have already plotted the matplotlib figure, 
        # here we can use Streamlit native charts for interactivity.
        
        chart_data = pd.DataFrame({
            "Position (m)": x,
            "Moment (kNm)": m / 1000.0,
            "Shear (kN)": v / 1000.0,
            "Deflection (mm)": d * 1000.0
        })
        
        # Interactive Charts
        st.caption("Bending Moment (kNm)")
        st.line_chart(chart_data, x="Position (m)", y="Moment (kNm)", color="#FF4B4B")
        
        st.caption("Shear Force (kN)")
        st.line_chart(chart_data, x="Position (m)", y="Shear (kN)", color="#0068C9")
        
        st.caption("Deflection (mm)")
        st.line_chart(chart_data, x="Position (m)", y="Deflection (mm)", color="#29B09D")
        
        st.divider()
        
        # Longitudinal Section Plot (Interactive SVG)
        st.subheader("2. Longitudinal Reinforcement Profile")
        svg_long, _ = plot_longitudinal_section_detailed(spans, sup_df, design_res, params['h'], params.get('cover', 25))
        st.image(svg_long, use_container_width=True)
        
        # Reactions Table
        st.subheader("3. Support Reactions")
        r_data = [{"Support": k, "Vertical Reaction (kN)": f"{val/1000:.2f}"} for k, val in react.items()]
        st.dataframe(pd.DataFrame(r_data), use_container_width=True)

    # ==========================================
    # TAB 2: Section Design Details
    # ==========================================
    with t2:
        st.subheader("Detailed Section Design by Span")
        
        # Selector for Span
        span_opts = [f"Span {i+1}" for i in range(len(spans))]
        selected_span_idx = st.selectbox("Select Span to View:", range(len(spans)), format_func=lambda x: span_opts[x])
        
        res = design_res[selected_span_idx]
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("#### Cross Section")
            svg_cross = plot_cross_section(res)
            st.image(svg_cross, use_container_width=True)
            
        with col2:
            st.markdown("#### Design Parameters")
            st.info(f"**Moment Capacity:** {res['phi_Mn']:.2f} kNm (Mu: {res['Mu_pos']:.2f})")
            st.info(f"**Shear Capacity:** {res['phi_Vn']:.2f} kN (Vu: {res['Vu_max']:.2f})")
            
            st.markdown("---")
            st.markdown("**Reinforcement Required:**")
            
            # Formatted Text for Steel
            top_layers = res.get('top', {}).get('all_layers', [])
            bot_layers = res.get('bot', {}).get('all_layers', [])
            
            top_txt = " + ".join([f"{int(l['n'])}DB{int(l['db'])}" for l in top_layers if l['n']>0])
            bot_txt = " + ".join([f"{int(l['n'])}DB{int(l['db'])}" for l in bot_layers if l['n']>0])
            
            shear_d = res.get('shear', {})
            stir_txt = f"RB{int(shear_d.get('db', 6))} @ {int(shear_d.get('s', 200))} mm"
            
            st.write(f"**Top:** {top_txt if top_txt else '-'}")
            st.write(f"**Bottom:** {bot_txt if bot_txt else '-'}")
            st.write(f"**Stirrups:** {stir_txt}")

    # ==========================================
    # TAB 3: Report & Bill of Quantities (BOQ)
    # ==========================================
    with t3:
        st.header("📝 Project Summary & Estimation")
        
        # 1. Bill of Quantities (BOQ)
        st.subheader("1. Bill of Quantities (Estimated)")
        
        boq_df = calculate_boq_summary(design_res, spans)
        
        # Formatting for display
        st.dataframe(
            boq_df.style.format({"Quantity": "{:.2f}"}), 
            use_container_width=True,
            hide_index=True
        )
        
        # CSV Download Button for BOQ
        csv = boq_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download BOQ (CSV)",
            data=csv,
            file_name='beam_boq_estimate.csv',
            mime='text/csv',
        )
        
        st.divider()
        
        # 2. Detailed Calculation Report
        st.subheader("2. Detailed Calculation Report")
        
        # Loop through each span to render report using reporter.py
        for i, res in enumerate(design_res):
            with st.expander(f"📄 View Calculation Note: Span {i+1}", expanded=False):
                # We need to inject span info into res for reporter
                res['span_id'] = i
                res['L'] = spans[i]
                render_calculation_report(res)
