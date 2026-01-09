import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# --- 🏗️ New Logical Setup ---
def plot_longitudinal_section_detailed(spans, sup_df, design_res, h_m, cover_mm):
    """
    แนวคิดใหม่: 
    1. บังคับ Aspect Ratio ให้ 'ผอมยาว' (Thin & Long) 
    2. แยก Layer เส้นคอนกรีตออกจากเส้นเหล็กด้วยระยะ Offset 15mm เสมอ
    """
    spans_mm = [s * 1000 for s in spans]
    total_L = sum(spans_mm)
    h_mm = h_m * 1000
    
    # แก้ปัญหาคานหนา: ใช้สัดส่วน Figure 15:2 (ยาวมากแต่เตี้ย)
    fig, ax = plt.subplots(figsize=(15, 2.5), dpi=120)
    
    # 1. วาดคอนกรีต (zorder=1)
    ax.add_patch(patches.Rectangle((0, 0), total_L, h_mm, lw=1.5, ec='black', fc='none', zorder=1))
    
    # 2. วาดเหล็กเสริม (zorder=5) 
    # บังคับห่างจากขอบ (cover + gap) เพื่อไม่ให้ทับเส้นขอบคานเด็ดขาด
    gap = cover_mm + 15
    y_top = h_mm - gap
    y_bot = gap
    
    x_curr = 0
    for i, span_L in enumerate(spans_mm):
        res = design_res[i]
        mid = x_curr + span_L/2
        
        # เหล็กบน (Main Top)
        ax.plot([x_curr, x_curr + span_L], [y_top, y_top], color='#e74c3c', lw=2, zorder=5)
        # เหล็กล่าง (Main Bot)
        ax.plot([x_curr + 50, x_curr + span_L - 50], [y_bot, y_bot], color='#27ae60', lw=2, zorder=5)
        
        # Label (วางนอกคาน ไม่ทับเนื้อคอนกรีต)
        ax.text(mid, h_mm + 80, f"{res['neg']['n']}-DB{int(res['top_db'])}", color='#e74c3c', ha='center', fontsize=8)
        ax.text(mid, -150, f"RB{int(res['stir_db'])}@{int(res['shear']['s'])}", color='#7f8c8d', ha='center', fontsize=8)
        
        x_curr += span_L

    # 3. วาด Support (ใต้คานเท่านั้น)
    if not sup_df.empty:
        for _, row in sup_df.iterrows():
            sx = row['x'] * 1000
            # วาดจาก y=0 ลงไปข้างล่าง
            ax.add_patch(patches.Rectangle((sx-60, -350), 120, 350, fc='#f1f2f6', ec='black', zorder=0))

    ax.set_aspect('auto') # บังคับให้คานยืดตามความยาว ไม่โดนบีบเป็นก้อนหนา
    ax.axis('off')
    ax.set_xlim(-500, total_L + 500)
    ax.set_ylim(-600, h_mm + 500)
    return fig
