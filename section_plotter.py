import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# --- 📏 Configuration ---
DPI_VALUE = 300     # ความคมชัดสูงป้องกันภาพแตก

def plot_section(b_m, h_m, cover_mm, db_main_mm, n_top, n_bottom, stirrup_name, fc, fy):
    """ Cross Section: แก้ไขให้รูปใหญ่ขึ้น ตัวหนังสือ 7.5 เท่าเดิม """
    # ⬇️ ปรับลดสเกลลงเพื่อให้รูปหน้าตัดขยายใหญ่ขึ้นในหน้าจอ (เดิมอาจจะเป็น 800-900)
    SECTION_SCALE = 500  
    
    b, h = b_m * 1000, h_m * 1000
    cover, ds, db = cover_mm, 6, db_main_mm
    
    # Viewport รอบหน้าตัด
    view_left, view_right = -300, b + 700
    view_bottom, view_top = -400, h + 400
    
    # คำนวณขนาดนิ้ว (Inches) ให้ใหญ่ขึ้นตาม SECTION_SCALE
    width_inches = (view_right - view_left) / SECTION_SCALE 
    height_inches = (view_top - view_bottom) / SECTION_SCALE
    
    fig, ax = plt.subplots(figsize=(width_inches, height_inches), dpi=DPI_VALUE)
    
    # วาดหน้าตัดคอนกรีต
    ax.add_patch(patches.Rectangle((0, 0), b, h, linewidth=1.2, edgecolor='#000000', facecolor='#ffffff'))
    # วาดเหล็กปลอก
    ax.add_patch(patches.Rectangle((cover, cover), b-2*cover, h-2*cover, linewidth=0.7, edgecolor='#2c3e50', facecolor='none', ls='--'))
    
    def draw_bars(n, y_pos, color):
        if n < 1: return
        spacing = (b - 2*cover - 2*ds - db) / (max(1, n - 1))
        for i in range(int(n)):
            x = (b/2) if n == 1 else (cover + ds + db/2 + i*spacing)
            ax.add_patch(plt.Circle((x, y_pos), db/2, color=color, zorder=10))

    y_bot, y_top = cover + ds + db/2, h - cover - ds - db/2
    draw_bars(n_bottom, y_bot, '#a93226')
    draw_bars(n_top, y_top, '#1f618d')
    
    # 📝 ตัวหนังสือขนาด 7.5 (ล็อคค่าคงที่)
    f_size = 7.5
    ax.text(b + 50, y_top, f"{int(n_top)}-DB{int(db)}", va='center', color='#1f618d', fontweight='bold', fontsize=f_size)
    ax.text(b + 50, y_bot, f"{int(n_bottom)}-DB{int(db)}", va='center', color='#a93226', fontweight='bold', fontsize=f_size)
    ax.text(b/2, h + 100, f"{stirrup_name}", ha='center', color='#1d8348', fontweight='bold', fontsize=f_size)

    # Dimension Lines
    def draw_tick_dim(p1, p2, text, vert=False):
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='#000000', lw=0.5)
        tick = 15
        for p in [p1, p2]: ax.plot([p[0]-tick, p[0]+tick], [p[1]-tick, p[1]+tick], color='#000000', lw=0.8)
        if vert: ax.text(p1[0]-60, (p1[1]+p2[1])/2, text, va='center', ha='right', rotation=90, fontsize=f_size-0.5)
        else: ax.text((p1[0]+p2[0])/2, p1[1]-60, text, ha='center', va='top', fontsize=f_size-0.5)

    draw_tick_dim([0, -120], [b, -120], f"{int(b)}")
    draw_tick_dim([-120, 0], [-120, h], f"{int(h)}", vert=True)
    
    ax.set_xlim(view_left, view_right)
    ax.set_ylim(view_bottom, view_top)
    ax.set_aspect('equal')
    ax.axis('off')
    return fig

def plot_longitudinal_section_detailed(spans, sup_df, design_res, h_m, cover_mm):
    """ Longitudinal Section: ใช้สเกลปกติเพื่อให้เห็นภาพรวมยาวๆ """
    LONG_SCALE = 850
    h_beam = h_m * 1000 
    total_L = sum(spans) * 1000
    offsets = [0] + list(np.cumsum(spans) * 1000)
    
    width_inches = (total_L + 1500) / LONG_SCALE
    height_inches = 2500 / LONG_SCALE 
    
    fig, ax = plt.subplots(figsize=(width_inches, height_inches), dpi=DPI_VALUE)
    ax.add_patch(patches.Rectangle((0, 0), total_L, h_beam, linewidth=1.2, edgecolor='#000000', facecolor='#ffffff'))
    
    f_size = 7.5
    for i, span_l_m in enumerate(spans):
        L_mm, x_s, x_e = span_l_m * 1000, offsets[i], offsets[i+1]
        res = design_res[i]
        mid_x = (x_s + x_e) / 2
        
        # เหล็กล่าง
        ax.plot([x_s+40, x_e-40], [cover_mm, cover_mm], color='#a93226', lw=1.5)
        ax.text(mid_x, cover_mm + 40, f"{int(res['pos']['n'])}-DB{int(res['db'])}", ha='center', color='#a93226', fontsize=f_size, fontweight='bold')
        
        # เหล็กบน
        y_top = h_beam - cover_mm
        ax.plot([x_s, x_s + L_mm*0.3], [y_top, y_top], color='#1f618d', lw=1.5)
        ax.plot([x_e - L_mm*0.3, x_e], [y_top, y_top], color='#1f618d', lw=1.5)
        ax.text(x_s + 60, y_top + 40, f"{int(res['neg']['n'])}-DB{int(res['db'])}", ha='left', color='#1f618d', fontsize=f_size, fontweight='bold')
        
        # เหล็กปลอก
        ax.text(mid_x, h_beam + 120, f"RB6@{int(res['shear']['s'])}", ha='center', color='#1d8348', fontsize=f_size, fontweight='bold')

    ax.set_xlim(-800, total_L + 800)
    ax.set_ylim(-1200, 1200)
    ax.set_aspect('equal')
    ax.axis('off')
    return fig
