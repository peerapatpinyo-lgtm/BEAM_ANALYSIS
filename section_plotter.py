import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# ล็อคสเกลกลาง: ตัวเลขยิ่งสูง รูปจะยิ่งละเอียดและตัวหนังสือจะดูเล็กลงเมื่อเทียบกับคาน
SCALE_FACTOR = 850 

def plot_section(b_m, h_m, cover_mm, db_main_mm, n_top, n_bottom, stirrup_name, fc, fy):
    b, h = b_m * 1000, h_m * 1000
    cover = cover_mm
    ds, db = 6, db_main_mm
    
    # เพิ่มระยะ Viewport ให้กว้างขึ้นเพื่อลดขนาดตัวหนังสือโดยรวม
    view_left, view_right = -500, b + 1200
    view_bottom, view_top = -600, h + 600
    
    width_inches = (view_right - view_left) / SCALE_FACTOR 
    height_inches = (view_top - view_bottom) / SCALE_FACTOR
    
    # ใช้ DPI 300 และเปิด Antialiasing เพื่อให้เส้นคมกริบ
    fig, ax = plt.subplots(figsize=(width_inches, height_inches), dpi=300)
    
    # คอนกรีต (lw=1.2)
    ax.add_patch(patches.Rectangle((0, 0), b, h, linewidth=1.2, edgecolor='#000000', facecolor='#ffffff'))
    # เหล็กปลอก
    ax.add_patch(patches.Rectangle((cover, cover), b-2*cover, h-2*cover, linewidth=0.6, edgecolor='#2c3e50', facecolor='none'))
    
    def draw_bars(n, y_pos, color):
        if n < 1: return
        spacing = (b - 2*cover - 2*ds - db) / (max(1, n - 1))
        for i in range(int(n)):
            x = (b/2) if n == 1 else (cover + ds + db/2 + i*spacing)
            ax.add_patch(plt.Circle((x, y_pos), db/2, color=color, zorder=10))

    y_bot, y_top = cover + ds + db/2, h - cover - ds - db/2
    draw_bars(n_bottom, y_bot, '#a93226')
    draw_bars(n_top, y_top, '#2e5984')
    
    # ปรับ Font เป็น 7.5 ให้ดูคมและเล็กแบบงานวิศวกรรม
    f_size = 7.5
    ax.text(b + 100, y_top, f"{int(n_top)}-DB{int(db)}", va='center', color='#2e5984', fontweight='bold', fontsize=f_size)
    ax.text(b + 100, y_bot, f"{int(n_bottom)}-DB{int(db)}", va='center', color='#a93226', fontweight='bold', fontsize=f_size)
    ax.text(b/2, h + 150, f"{stirrup_name}", ha='center', color='#1d8348', fontweight='bold', fontsize=f_size)

    # Engineering Ticks (ขยับระยะ text ให้ออกห่างจากเส้นนิดนึง)
    def draw_tick_dim(p1, p2, text, vert=False):
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='#000000', lw=0.5)
        tick = 20
        for p in [p1, p2]:
            ax.plot([p[0]-tick, p[0]+tick], [p[1]-tick, p[1]+tick], color='#000000', lw=0.8)
        if vert:
            ax.text(p1[0]-120, (p1[1]+p2[1])/2, text, va='center', ha='right', rotation=90, fontsize=f_size)
        else:
            ax.text((p1[0]+p2[0])/2, p1[1]-120, text, ha='center', va='top', fontsize=f_size)

    draw_tick_dim([0, -150], [b, -150], f"{int(b)}")
    draw_tick_dim([-150, 0], [-150, h], f"{int(h)}", vert=True)
    
    ax.set_xlim(view_left, view_right)
    ax.set_ylim(view_bottom, view_top)
    ax.set_aspect('equal')
    ax.axis('off')
    return fig
