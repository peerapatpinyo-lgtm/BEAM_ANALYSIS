import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def plot_section(b_m, h_m, cover_mm, db_main_mm, n_top, n_bottom, stirrup_name, fc, fy):
    """
    คืนค่า Section รูปแบบเดิมที่คุณมั่นใจ พร้อมข้อมูลวัสดุครบถ้วน
    """
    b, h = b_m * 1000, h_m * 1000
    cover = cover_mm
    ds = 6  
    db = db_main_mm
    
    fig, ax = plt.subplots(figsize=(5, 7))
    
    # 1. Concrete Frame (Hatch แบบเดิม)
    ax.add_patch(patches.Rectangle((0, 0), b, h, linewidth=2, edgecolor='black', facecolor='#f8f9fa', hatch='///', alpha=0.3))
    
    # 2. Stirrup (เส้นเหล็กปลอก)
    ax.add_patch(patches.Rectangle((cover, cover), b-2*cover, h-2*cover, linewidth=1.5, edgecolor='#2c3e50', facecolor='none'))
    
    # 3. Rebar Drawing Logic (กระจายเหล็กแบบเดิมที่ดูง่าย)
    def draw_bars(n, y_pos, color):
        if n < 2: 
            # กรณีเหล็กเส้นเดียวหรือไม่มี (แต่มาตรฐานคานต้องมีอย่างน้อย 2)
            if n == 1: ax.add_patch(plt.Circle((b/2, y_pos), db/2, color=color, zorder=5))
            return
        spacing = (b - 2*cover - 2*ds - db) / (n - 1)
        for i in range(n):
            x = cover + ds + db/2 + i*spacing
            ax.add_patch(plt.Circle((x, y_pos), db/2, color=color, zorder=5))

    draw_bars(n_bottom, cover + ds + db/2, '#c0392b') # เหล็กล่าง (สีแดง)
    draw_bars(n_top, h - cover - ds - db/2, '#2980b9') # เหล็กบน (สีน้ำเงิน)
    
    # 4. ข้อมูลวัสดุและรายละเอียด (กลับมาแสดงผลข้างรูปเหมือนเดิม)
    info_text = (
        f"BEAM SECTION\n"
        f"Size: {int(b)}x{int(h)} mm\n"
        f"Concrete: {fc} MPa\n"
        f"Main Steel: {fy} MPa\n"
        f"Top: {n_top}-DB{db}\n"
        f"Bottom: {n_bottom}-DB{db}\n"
        f"Stirrup: {stirrup_name}"
    )
    plt.text(b + 30, h, info_text, va='top', family='monospace', fontsize=10, 
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='#bdc3c7'))
    
    # 5. Dimension Lines (เส้นบอกขนาด)
    ax.annotate('', xy=(0, -30), xytext=(b, -30), arrowprops=dict(arrowstyle='<->'))
    ax.text(b/2, -60, f"{int(b)}", ha='center', size=9)
    ax.annotate('', xy=(-30, 0), xytext=(-30, h), arrowprops=dict(arrowstyle='<->'))
    ax.text(-60, h/2, f"{int(h)}", va='center', rotation=90, size=9)
    
    ax.set_xlim(-120, b + 250)
    ax.set_ylim(-120, h + 100)
    ax.set_aspect('equal')
    ax.axis('off')
    return fig

def plot_longitudinal_section_detailed(spans, sup_df, design_res, h_m, cover_mm):
    """
    Longitudinal Section ที่เน้นความชัดเจนของระยะ Span และตำแหน่งเหล็ก
    """
    h = h_m * 1000
    total_L = sum(spans) * 1000
    offsets = [0] + list(np.cumsum(spans) * 1000)
    
    fig, ax = plt.subplots(figsize=(15, 4))
    
    # คานคอนกรีต
    ax.add_patch(patches.Rectangle((0, 0), total_L, h, facecolor='#ffffff', edgecolor='black', lw=1.5))
    
    for i, span_l_m in enumerate(spans):
        L_mm = span_l_m * 1000
        x_s, x_e = offsets[i], offsets[i+1]
        
        # เหล็กล่าง (เน้นความยาวเต็ม Span)
        ax.plot([x_s+20, x_e-20], [cover_mm, cover_mm], color='#c0392b', lw=2)
        # เหล็กบน (เน้นช่วง Support)
        ax.plot([x_s, x_s + L_mm/3], [h-cover_mm, h-cover_mm], color='#2980b9', lw=2)
        ax.plot([x_e - L_mm/3, x_e], [h-cover_mm, h-cover_mm], color='#2980b9', lw=2)
        
        # ใส่ Label กำกับแต่ละ Span
        ax.text(x_s + L_mm/2, -100, f"Span {i+1}\n{span_l_m}m", ha='center', fontsize=9)

    # วาด Support
    for _, sup in sup_df.iterrows():
        ax.plot([sup['x']*1000, sup['x']*1000], [0, -50], 'k-', lw=2)
        ax.plot([sup['x']*1000-50, sup['x']*1000+50], [-50, -50], 'k-', lw=2)

    ax.set_xlim(-100, total_L + 100)
    ax.set_ylim(-200, h + 150)
    ax.axis('off')
    return fig
