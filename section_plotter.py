import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# กำหนดค่าคงที่สำหรับ Scale (Physical Unit Scaling)
# ให้ 500mm ในแบบ เท่ากับ 1.0 unit ใน figsize
SCALE_FACTOR = 500 

def plot_section(b_m, h_m, cover_mm, db_main_mm, n_top, n_bottom, stirrup_name, fc, fy):
    """
    Cross Section - มาตราส่วนเดียวกับรูปตัดตามยาว
    """
    b, h = b_m * 1000, h_m * 1000
    cover = cover_mm
    ds, db = 6, db_main_mm
    
    # คำนวณ figsize ตามขนาดจริงของวัตถุ
    width_inches = (b + 600) / SCALE_FACTOR 
    height_inches = (h + 600) / SCALE_FACTOR
    
    fig, ax = plt.subplots(figsize=(width_inches, height_inches), dpi=100)
    
    # คอนกรีต
    ax.add_patch(patches.Rectangle((0, 0), b, h, linewidth=2, edgecolor='#1a1a1a', facecolor='#ffffff'))
    # เหล็กปลอก
    ax.add_patch(patches.Rectangle((cover, cover), b-2*cover, h-2*cover, linewidth=1.5, edgecolor='#2c3e50', facecolor='none'))
    
    def draw_bars(n, y_pos, color):
        if n < 1: return
        spacing = (b - 2*cover - 2*ds - db) / (max(1, n - 1))
        for i in range(int(n)):
            x = (b/2) if n == 1 else (cover + ds + db/2 + i*spacing)
            ax.add_patch(plt.Circle((x, y_pos), db/2, color=color, zorder=10))

    y_bot, y_top = cover + ds + db/2, h - cover - ds - db/2
    draw_bars(n_bottom, y_bot, '#8e1c19')
    draw_bars(n_top, y_top, '#1a5276')
    
    # Label บอกเหล็ก (วางตำแหน่งคงที่นอกรูป)
    ax.text(b + 50, y_top, f"{int(n_top)}-DB{int(db)}", va='center', color='#1a5276', fontweight='bold', fontsize=12)
    ax.text(b + 50, y_bot, f"{int(n_bottom)}-DB{int(db)}", va='center', color='#8e1c19', fontweight='bold', fontsize=12)

    # Engineering Ticks Dimension
    def draw_tick_dim(p1, p2, text, vert=False):
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='black', lw=1.2)
        tick = 25
        for p in [p1, p2]:
            ax.plot([p[0]-tick, p[0]+tick], [p[1]-tick, p[1]+tick], color='black', lw=2)
        if vert:
            ax.text(p1[0]-60, (p1[1]+p2[1])/2, text, va='center', ha='right', rotation=90, fontsize=11)
        else:
            ax.text((p1[0]+p2[0])/2, p1[1]-60, text, ha='center', va='top', fontsize=11)

    draw_tick_dim([0, -80], [b, -80], f"{int(b)}")
    draw_tick_dim([-80, 0], [-80, h], f"{int(h)}", vert=True)
    
    ax.set_xlim(-250, b + 450)
    ax.set_ylim(-250, h + 250)
    ax.set_aspect('equal')
    ax.axis('off')
    return fig

def plot_longitudinal_section_detailed(spans, sup_df, design_res, h_m, cover_mm):
    """
    Longitudinal Section - มาตราส่วนเดียวกับรูปตัดขวาง
    """
    h_beam = h_m * 1000 
    total_L = sum(spans) * 1000
    offsets = [0] + list(np.cumsum(spans) * 1000)
    
    # คำนวณ figsize ตามขนาดจริงของวัตถุ (ใช้ SCALE_FACTOR เดียวกัน)
    width_inches = (total_L + 1000) / SCALE_FACTOR
    height_inches = (h_beam + 1200) / SCALE_FACTOR
    
    fig, ax = plt.subplots(figsize=(width_inches, height_inches), dpi=100)
    
    # ตัวคาน
    ax.add_patch(patches.Rectangle((0, 0), total_L, h_beam, linewidth=2, edgecolor='#1a1a1a', facecolor='#ffffff'))
    
    # Support Symbols (วาดตาม Type จริง)
    for _, sup in sup_df.iterrows():
        sx = sup['x'] * 1000
        stype = sup['type']
        if stype == 'Fixed':
            ax.add_patch(patches.Rectangle((sx-30, -100), 60, h_beam+200, facecolor='#bdc3c7', hatch='///', alpha=0.6))
        elif stype == 'Pin':
            poly = plt.Polygon([[sx-120, -150], [sx+120, -150], [sx, 0]], closed=True, facecolor='none', edgecolor='black', lw=1.5)
            ax.add_patch(poly)
            ax.plot([sx-150, sx+150], [-155, -155], color='black', lw=2)
        elif stype == 'Roller':
            poly = plt.Polygon([[sx-120, -120], [sx+120, -120], [sx, 0]], closed=True, facecolor='none', edgecolor='black', lw=1.5)
            ax.add_patch(poly)
            ax.add_patch(plt.Circle((sx, -150), 30, facecolor='none', edgecolor='black', lw=1.5))
            ax.plot([sx-150, sx+150], [-185, -185], color='black', lw=2)

    for i, span_l_m in enumerate(spans):
        L_mm, x_s, x_e = span_l_m * 1000, offsets[i], offsets[i+1]
        res = design_res[i]
        mid_x = (x_s + x_e) / 2
        
        # เหล็กล่าง (ความหนาเส้นต้องเท่ากับรูปตัดขวาง)
        ax.plot([x_s+50, x_e-50], [cover_mm, cover_mm], color='#8e1c19', lw=3)
        ax.text(mid_x, cover_mm + 50, f"{int(res['pos']['n'])}-DB{int(res['db'])}", ha='center', color='#8e1c19', fontsize=12, fontweight='bold')
        
        # เหล็กบน (แสดงครบถ้วน)
        y_top = h_beam - cover_mm
        ax.plot([x_s, x_s + L_mm*0.3], [y_top, y_top], color='#1a5276', lw=3)
        ax.plot([x_e - L_mm*0.3, x_e], [y_top, y_top], color='#1a5276', lw=3)
        ax.text(x_s + 100, y_top + 50, f"{int(res['neg']['n'])}-DB{int(res['db'])}", ha='left', color='#1a5276', fontsize=12, fontweight='bold')
        
        # เหล็กปลอก
        ax.text(mid_x, h_beam + 150, f"RB6@{int(res['shear']['s'])}", ha='center', color='#1d8348', fontsize=12, fontweight='bold')

    # Dimension Ticks (มาตราส่วนเดียวกับรูปตัดขวาง)
    ax.plot([0, total_L], [-h_beam*1.2, -h_beam*1.2], color='black', lw=1.2)
    for px in [0, total_L]:
        ax.plot([px-30, px+30], [-h_beam*1.2-30, -h_beam*1.2+30], color='black', lw=2)
    ax.text(total_L/2, -h_beam*1.2 - 150, f"Total Length = {total_L/1000} m", ha='center', fontweight='bold', fontsize=14)

    ax.set_xlim(-500, total_L + 500)
    ax.set_ylim(-h_beam*2.0, h_beam + 600)
    ax.set_aspect('equal')
    ax.axis('off')
    return fig
