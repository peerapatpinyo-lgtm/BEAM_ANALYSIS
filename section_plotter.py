import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# --- 📏 Global Configuration ---
DPI_VALUE = 300     
GLOBAL_FONT = 7.5

def plot_section(b_m, h_m, cover_mm, db_main_mm, n_top, n_bottom, stirrup_name, fc=None, fy=None, title="SECTION A-A"):
    """ วาดรูปตัดขวาง (Cross Section) """
    SECTION_SCALE = 500  
    b, h = b_m * 1000, h_m * 1000
    cover, ds, db = cover_mm, 6, db_main_mm
    
    view_left, view_right = -300, b + 750
    view_bottom, view_top = -450, h + 550
    
    width_inches = (view_right - view_left) / SECTION_SCALE 
    height_inches = (view_top - view_bottom) / SECTION_SCALE
    
    fig, ax = plt.subplots(figsize=(width_inches, height_inches), dpi=DPI_VALUE)
    
    # วาดคอนกรีต
    ax.add_patch(patches.Rectangle((0, 0), b, h, linewidth=1.5, edgecolor='#000000', facecolor='#ffffff'))
    ax.text(b/2, -350, title, ha='center', fontweight='bold', fontsize=GLOBAL_FONT + 1)
    
    # วาดเหล็กปลอก
    ax.add_patch(patches.Rectangle((cover, cover), b-2*cover, h-2*cover, linewidth=0.8, edgecolor='#2c3e50', ls='--'))
    
    def draw_bars(n, y_pos, color):
        if n < 1: return
        spacing = (b - 2*cover - 2*ds - db) / (max(1, n - 1))
        for i in range(int(n)):
            x = (b/2) if n == 1 else (cover + ds + db/2 + i*spacing)
            ax.add_patch(plt.Circle((x, y_pos), db/2, color=color, zorder=10))

    y_bot, y_top = cover + ds + db/2, h - cover - ds - db/2
    draw_bars(n_bottom, y_bot, '#a93226') # เหล็กล่าง
    draw_bars(n_top, y_top, '#1f618d')    # เหล็กบน
    
    # Text กำกับ
    ax.text(b + 70, y_top, f"{int(n_top)}-DB{int(db)}", va='center', color='#1f618d', fontsize=GLOBAL_FONT)
    ax.text(b + 70, y_bot, f"{int(n_bottom)}-DB{int(db)}", va='center', color='#a93226', fontsize=GLOBAL_FONT)
    ax.text(b/2, h + 120, stirrup_name, ha='center', color='#1d8348', fontsize=GLOBAL_FONT)

    # Dimension
    def draw_tick_dim(p1, p2, text, vert=False):
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='#000000', lw=0.6)
        if vert: ax.text(p1[0]-80, (p1[1]+p2[1])/2, text, va='center', ha='right', rotation=90, fontsize=GLOBAL_FONT-1)
        else: ax.text((p1[0]+p2[0])/2, p1[1]-80, text, ha='center', va='top', fontsize=GLOBAL_FONT-1)

    draw_tick_dim([0, -150], [b, -150], f"{int(b)}")
    draw_tick_dim([-150, 0], [-150, h], f"{int(h)}", vert=True)
    
    ax.set_xlim(view_left, view_right)
    ax.set_ylim(view_bottom, view_top)
    ax.set_aspect('equal')
    ax.axis('off')
    return fig

def plot_longitudinal_section_detailed(spans, sup_df, design_res, h_m, cover_mm):
    """ วาดรูปตัดตามยาวพร้อม Support (Hinge/Roller) """
    LONG_SCALE = 850
    h_beam = h_m * 1000 
    total_L = sum(spans) * 1000
    offsets = [0] + list(np.cumsum(spans) * 1000)
    
    width_inches = (total_L + 2000) / LONG_SCALE
    height_inches = 4500 / LONG_SCALE 
    fig, ax = plt.subplots(figsize=(width_inches, height_inches), dpi=DPI_VALUE)
    
    # 1. วาดตัวคาน
    ax.add_patch(patches.Rectangle((0, 0), total_L, h_beam, linewidth=1.5, edgecolor='#000000', facecolor='#ffffff', zorder=1))
    
    # 2. วาด Support (Hinge & Roller)
    for i, (_, row) in enumerate(sup_df.iterrows()):
        x_s = row['x'] * 1000
        if i == 0: # Hinge (สามเหลี่ยม + ฐานขีด)
            poly = plt.Polygon([[x_s-150, -300], [x_s+150, -300], [x_s, 0]], facecolor='#ffffff', edgecolor='black', lw=1, zorder=2)
            ax.add_patch(poly)
            ax.plot([x_s-250, x_s+250], [-300, -300], color='black', lw=1.5)
        else: # Roller (สามเหลี่ยม + วงกลม)
            poly = plt.Polygon([[x_s-150, -200], [x_s+150, -200], [x_s, 0]], facecolor='#ffffff', edgecolor='black', lw=1, zorder=2)
            ax.add_patch(poly)
            ax.add_patch(plt.Circle((x_s, -250), 50, color='black', fill=False, lw=1))
            ax.plot([x_s-250, x_s+250], [-310, -310], color='black', lw=1.2)

    # 3. วาดเส้นตัด Section A และ B
    sec_a_x = (offsets[0] + offsets[1]) / 2 
    ax.plot([sec_a_x, sec_a_x], [-700, h_beam + 700], color='#d35400', ls='--', lw=1)
    ax.text(sec_a_x, h_beam + 800, "A", color='#d35400', fontweight='bold', ha='center', fontsize=GLOBAL_FONT+2)
    
    sec_b_x = offsets[1] 
    ax.plot([sec_b_x, sec_b_x], [-700, h_beam + 700], color='#2980b9', ls='--', lw=1)
    ax.text(sec_b_x, h_beam + 800, "B", color='#2980b9', fontweight='bold', ha='center', fontsize=GLOBAL_FONT+2)

    # 4. วาดเหล็กเสริม
    for i, span_l_m in enumerate(spans):
        L_mm, x_s, x_e = span_l_m * 1000, offsets[i], offsets[i+1]
        res = design_res[i]
        # เหล็กล่าง
        ax.plot([x_s+50, x_e-50], [cover_mm, cover_mm], color='#a93226', lw=1.8)
        # เหล็กบน
        y_t = h_beam - cover_mm
        ax.plot([x_s, x_s + L_mm*0.3], [y_t, y_t], color='#1f618d', lw=1.8)
        ax.plot([x_e - L_mm*0.3, x_e], [y_t, y_t], color='#1f618d', lw=1.8)

    ax.set_xlim(-1000, total_L + 1000)
    ax.set_ylim(-2000, h_beam + 2000)
    ax.set_aspect('equal')
    ax.axis('off')
    return fig
