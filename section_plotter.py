import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# --- 📏 Configuration ---
DPI_VALUE = 300     
GLOBAL_FONT = 7.5

def plot_section(b_m, h_m, cover_mm, db_main_mm, n_top, n_bottom, stirrup_name, fc, fy):
    """ Cross Section: รูปใหญ่ ตัวหนังสือ 7.5 """
    SECTION_SCALE = 500  
    b, h = b_m * 1000, h_m * 1000
    cover, ds, db = cover_mm, 6, db_main_mm
    
    view_left, view_right = -300, b + 700
    view_bottom, view_top = -400, h + 400
    
    width_inches = (view_right - view_left) / SECTION_SCALE 
    height_inches = (view_top - view_bottom) / SECTION_SCALE
    
    fig, ax = plt.subplots(figsize=(width_inches, height_inches), dpi=DPI_VALUE)
    
    ax.add_patch(patches.Rectangle((0, 0), b, h, linewidth=1.2, edgecolor='#000000', facecolor='#ffffff'))
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
    
    ax.text(b + 50, y_top, f"{int(n_top)}-DB{int(db)}", va='center', color='#1f618d', fontweight='bold', fontsize=GLOBAL_FONT)
    ax.text(b + 50, y_bot, f"{int(n_bottom)}-DB{int(db)}", va='center', color='#a93226', fontweight='bold', fontsize=GLOBAL_FONT)
    ax.text(b/2, h + 120, f"{stirrup_name}", ha='center', color='#1d8348', fontweight='bold', fontsize=GLOBAL_FONT)

    def draw_tick_dim(p1, p2, text, vert=False):
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='#000000', lw=0.5)
        tick = 15
        for p in [p1, p2]: ax.plot([p[0]-tick, p[0]+tick], [p[1]-tick, p[1]+tick], color='#000000', lw=0.8)
        if vert: ax.text(p1[0]-60, (p1[1]+p2[1])/2, text, va='center', ha='right', rotation=90, fontsize=GLOBAL_FONT-0.5)
        else: ax.text((p1[0]+p2[0])/2, p1[1]-60, text, ha='center', va='top', fontsize=GLOBAL_FONT-0.5)

    draw_tick_dim([0, -120], [b, -120], f"{int(b)}")
    draw_tick_dim([-120, 0], [-120, h], f"{int(h)}", vert=True)
    
    ax.set_xlim(view_left, view_right)
    ax.set_ylim(view_bottom, view_top)
    ax.set_aspect('equal')
    ax.axis('off')
    return fig

def plot_longitudinal_section_detailed(spans, sup_df, design_res, h_m, cover_mm):
    """ Longitudinal Section: แก้ไข Support หาย และตัวหนังสือทับคาน """
    LONG_SCALE = 800
    h_beam = h_m * 1000 
    total_L = sum(spans) * 1000
    offsets = [0] + list(np.cumsum(spans) * 1000)
    
    # ขยายความสูงภาพ (height_inches) เพื่อให้มีพื้นที่วาด Support ด้านล่าง
    width_inches = (total_L + 1600) / LONG_SCALE
    height_inches = 4000 / LONG_SCALE 
    
    fig, ax = plt.subplots(figsize=(width_inches, height_inches), dpi=DPI_VALUE)
    
    # 1. วาดตัวคาน
    ax.add_patch(patches.Rectangle((0, 0), total_L, h_beam, linewidth=1.2, edgecolor='#000000', facecolor='#ffffff', zorder=1))
    
# 2. วาด Support (Hinge สำหรับจุดแรก, Roller สำหรับจุดที่เหลือ)
    for i, (_, row) in enumerate(sup_df.iterrows()):
        x_s = row['x'] * 1000
        
        if i == 0:
            # วาด Hinge (สามเหลี่ยมมีขีดฐาน)
            poly = plt.Polygon([[x_s-150, -300], [x_s+150, -300], [x_s, 0]], 
                               facecolor='#ffffff', edgecolor='black', lw=1, zorder=2)
            ax.add_patch(poly)
            # ขีดฐานแสดงความแน่น
            ax.plot([x_s-250, x_s+250], [-300, -300], color='black', lw=1.5)
            for j in range(6): # ขีดเฉียงใต้ฐาน
                ax.plot([x_s-250 + j*100, x_s-200 + j*100], [-350, -300], color='black', lw=0.8)
        else:
            # วาด Roller (สามเหลี่ยมมีช่องว่าง/วงกลมใต้ฐาน)
            poly = plt.Polygon([[x_s-150, -250], [x_s+150, -250], [x_s, 0]], 
                               facecolor='#ffffff', edgecolor='black', lw=1, zorder=2)
            ax.add_patch(poly)
            # วาดล้อ (วงกลมเล็กๆ 2 วง)
            ax.add_patch(plt.Circle((x_s-70, -280), 30, color='black', fill=False, lw=0.8))
            ax.add_patch(plt.Circle((x_s+70, -280), 30, color='black', fill=False, lw=0.8))
            # เส้นพื้นดิน
            ax.plot([x_s-250, x_s+250], [-310, -310], color='black', lw=1.2)
    
    # 3. วาดเหล็กและข้อความ
    for i, span_l_m in enumerate(spans):
        L_mm, x_s, x_e = span_l_m * 1000, offsets[i], offsets[i+1]
        res = design_res[i]
        mid_x = (x_s + x_e) / 2
        
        # --- เหล็กล่าง (ขยับ Text ลงมาด้านล่างเส้นเหล็ก) ---
        y_rebar_bot = cover_mm
        ax.plot([x_s+50, x_e-50], [y_rebar_bot, y_rebar_bot], color='#a93226', lw=1.5, zorder=3)
        ax.text(mid_x, y_rebar_bot - 120, f"{int(res['pos']['n'])}-DB{int(res['db'])}", 
                ha='center', va='top', color='#a93226', fontsize=GLOBAL_FONT, fontweight='bold')
        
        # --- เหล็กบน (ขยับ Text ขึ้นไปด้านบนเส้นเหล็ก) ---
        y_rebar_top = h_beam - cover_mm
        ax.plot([x_s, x_s + L_mm*0.3], [y_rebar_top, y_rebar_top], color='#1f618d', lw=1.5, zorder=3)
        ax.plot([x_e - L_mm*0.3, x_e], [y_rebar_top, y_rebar_top], color='#1f618d', lw=1.5, zorder=3)
        ax.text(mid_x, y_rebar_top + 120, f"{int(res['neg']['n'])}-DB{int(res['db'])}", 
                ha='center', va='bottom', color='#1f618d', fontsize=GLOBAL_FONT, fontweight='bold')
        
        # --- ข้อความเหล็กปลอก (อยู่เหนือคาน) ---
        ax.text(mid_x, h_beam + 250, f"RB6@{int(res['shear']['s'])}", 
                ha='center', va='bottom', color='#1d8348', fontsize=GLOBAL_FONT, fontweight='bold')

    # ปรับ Viewport ให้เห็น Support (ylim ด้านล่างต้องติดลบพอสมควร)
    ax.set_xlim(-800, total_L + 800)
    ax.set_ylim(-1500, h_beam + 1500) 
    ax.set_aspect('equal')
    ax.axis('off')
    return fig
