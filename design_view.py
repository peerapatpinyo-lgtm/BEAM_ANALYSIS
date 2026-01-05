import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def render_result_tables(df_res, reactions, spans, unit_force="kg", unit_len="m"):
    """
    ฟังก์ชันแสดงตารางผลลัพธ์ (รองรับ NumPy Array และแก้บั๊ก AttributeError)
    """
    st.markdown("---")
    st.subheader("📋 Analysis Results")

    col1, col2 = st.columns(2)

    # --- Col 1: Reactions ---
    with col1:
        st.markdown(f"**📍 Support Reactions ({unit_force})**")
        # ใช้ len() > 0 เพื่อป้องกัน Error กับ NumPy Array
        if reactions is not None and len(reactions) > 0:
            react_data = []
            for i, r in enumerate(reactions):
                react_data.append({
                    "Support": f"Support {i+1}",
                    "Reaction": f"{r:,.2f}"
                })
            df_react = pd.DataFrame(react_data)
            st.table(df_react)
        else:
            st.warning("No reaction data available.")

    # --- Col 2: Critical Values ---
    with col2:
        st.markdown(f"**📊 Critical Design Values**")
        if df_res is not None and not df_res.empty:
            st.dataframe(df_res.style.format({
                "Value": "{:,.2f}",
                "Position": "{:.2f}"
            }))
        else:
            st.info("No result data to display.")
    
    total_len = sum(spans) if isinstance(spans, list) else spans
    st.caption(f"Total Span Length: {total_len:.2f} {unit_len}")

def plot_professional_diagrams(L_total, loads, reactions_locs, shear_x, shear_y, moment_x, moment_y):
    """
    วาดกราฟแบบ Professional Engineering Style (เหมือนใน Textbook)
    - Support แยกออกจากคาน
    - Load Uniform แบบหวี (Comb style)
    - คานมีความหนา
    """
    # ตั้งค่า Font และ Style
    plt.rcParams['font.family'] = 'sans-serif'
    plt.style.use('default')
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True, 
                                        gridspec_kw={'height_ratios': [1, 1, 1]})
    
    # ==========================================
    # 1. Free Body Diagram (FBD)
    # ==========================================
    # กำหนดความหนาคานให้ดูสมส่วน
    beam_height = L_total * 0.06 
    if beam_height < 0.3: beam_height = 0.3 # ขั้นต่ำ
    
    # 1.1 วาดตัวคาน (Beam Rectangle)
    beam_rect = patches.Rectangle((0, -beam_height/2), L_total, beam_height, 
                                  linewidth=2, edgecolor='#333333', facecolor='#f9f9f9', zorder=2)
    ax1.add_patch(beam_rect)
    
    # 1.2 วาด Support (อยู่ *ใต้* คาน ไม่ทับ)
    support_h = beam_height * 0.8
    support_w = beam_height * 0.8
    
    for loc in reactions_locs:
        # วาดสามเหลี่ยม Support
        triangle = patches.Polygon([
            [loc, -beam_height/2],  # ยอด (แตะขอบล่างคาน)
            [loc - support_w/2, -beam_height/2 - support_h], # ฐานซ้าย
            [loc + support_w/2, -beam_height/2 - support_h]  # ฐานขวา
        ], closed=True, edgecolor='#333333', facecolor='#ffffff', linewidth=1.5, zorder=1)
        ax1.add_patch(triangle)
        
        # วาดเส้นพื้น (Ground)
        ax1.plot([loc - support_w, loc + support_w], 
                 [-beam_height/2 - support_h]*2, color='black', linewidth=1.5)
        
        # Hash marks (แรเงาพื้น)
        for i in np.linspace(-support_w, support_w, 4):
             ax1.plot([loc + i, loc + i - support_w/3], 
                      [-beam_height/2 - support_h, -beam_height/2 - support_h*1.3], 
                      color='black', linewidth=0.8)

    # 1.3 วาด Loads (ให้ดูโปร)
    max_load_h = beam_height * 2.0 
    
    for load in loads:
        l_type = load[0]
        val = load[1]
        
        if l_type == 'udl':
            start, end = load[2], load[3]
            
            # วาดเส้น Load ด้านบน (คานรับ Load)
            load_top_y = beam_height/2 + max_load_h
            ax1.plot([start, end], [load_top_y]*2, color='#005b96', linewidth=1.5)
            # ปิดหัวท้าย
            ax1.plot([start, start], [beam_height/2, load_top_y], color='#005b96', linewidth=1.5)
            ax1.plot([end, end], [beam_height/2, load_top_y], color='#005b96', linewidth=1.5)
            
            # วาดลูกศรถี่ๆ (Comb Style) ให้ดูเหมือน Textbook
            # คำนวณจำนวนลูกศรตามความยาว
            dist = end - start
            n_arrows = max(3, int(dist * 3)) # อย่างน้อย 3 ตัว หรือ 3 ตัวต่อเมตร
            x_arrows = np.linspace(start, end, n_arrows)
            
            for x in x_arrows:
                ax1.arrow(x, load_top_y, 0, -max_load_h*0.85, 
                          head_width=L_total*0.015, head_length=max_load_h*0.15, 
                          fc='#005b96', ec='#005b96', length_includes_head=True)
            
            # Text Label
            ax1.text((start+end)/2, load_top_y + beam_height*0.2, f"w = {val:,.0f}", 
                     ha='center', va='bottom', color='#005b96', fontweight='bold')

        elif l_type == 'point':
            pos = load[2]
            load_top_y = beam_height/2 + max_load_h
            # ลูกศรตัวใหญ่
            ax1.arrow(pos, load_top_y, 0, -max_load_h*0.9, 
                      head_width=L_total*0.02, head_length=max_load_h*0.2, 
                      fc='#d9534f', ec='#d9534f', width=L_total*0.003, length_includes_head=True)
            # Text Label
            ax1.text(pos, load_top_y + beam_height*0.2, f"P = {val:,.0f}", 
                     ha='center', va='bottom', color='#d9534f', fontweight='bold')

    ax1.set_title("Free Body Diagram", fontsize=14, fontweight='bold', pad=15)
    ax1.set_ylim(-beam_height*4, beam_height*5)
    ax1.axis('off')

    # ==========================================
    # 2. Shear Force Diagram (SFD)
    # ==========================================
    ax2.plot(shear_x, shear_y, color='#ff9f43', linewidth=2)
    ax2.fill_between(shear_x, shear_y, 0, facecolor='#ff9f43', alpha=0.15)
    ax2.axhline(0, color='black', linewidth=0.8)
    ax2.set_ylabel("Shear Force", fontsize=10, fontweight='bold')
    ax2.grid(True, linestyle=':', alpha=0.6)
    
    # Annotate Max/Min
    v_max_idx = np.argmax(shear_y)
    v_min_idx = np.argmin(shear_y)
    for idx in [v_max_idx, v_min_idx]:
        val = shear_y[idx]
        if abs(val) > 0.1:
            ax2.text(shear_x[idx], val, f"{val:,.0f}", ha='center', va='bottom' if val > 0 else 'top',
                     fontsize=9, bbox=dict(facecolor='white', edgecolor='#ff9f43', boxstyle='round,pad=0.2'))

    # ==========================================
    # 3. Bending Moment Diagram (BMD)
    # ==========================================
    ax3.plot(moment_x, moment_y, color='#54a0ff', linewidth=2)
    ax3.fill_between(moment_x, moment_y, 0, facecolor='#54a0ff', alpha=0.15)
    ax3.axhline(0, color='black', linewidth=0.8)
    ax3.set_ylabel("Moment", fontsize=10, fontweight='bold')
    ax3.set_xlabel("Beam Length (m)", fontsize=10)
    ax3.grid(True, linestyle=':', alpha=0.6)

    # Annotate Max/Min Moment
    m_max_idx = np.argmax(moment_y)
    m_min_idx = np.argmin(moment_y)
    for idx in [m_max_idx, m_min_idx]:
        val = moment_y[idx]
        if abs(val) > 0.1:
             ax3.text(moment_x[idx], val, f"{val:,.0f}", ha='center', va='bottom' if val > 0 else 'top',
                     fontsize=9, bbox=dict(facecolor='white', edgecolor='#54a0ff', boxstyle='round,pad=0.2'))

    plt.tight_layout()
    return fig
