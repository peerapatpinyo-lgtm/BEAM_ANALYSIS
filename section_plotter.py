import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# --- 📏 Configuration เพื่อความคมชัดสูงและสเกลคงที่ ---
# ใช้ SCALE_FACTOR ชุดเดียวเพื่อให้ขนาดเส้นและตัวหนังสือสัมพันธ์กัน
SCALE_FACTOR = 800 

def plot_section(b_m, h_m, cover_mm, db_main_mm, n_top, n_bottom, stirrup_name, fc, fy):
    """
    Cross Section: เน้นเส้นคมกริบและตัวหนังสือขนาดจิ๋วที่เท่ากับ Long Section
    """
    b, h = b_m * 1000, h_m * 1000
    cover = cover_mm
    ds, db = 6, db_main_mm
    
    # กำหนดขอบเขตการมองเห็น (Viewport) ให้คงที่เพื่อให้สเกลรูปไม่กระโดด
    view_left, view_right = -400, b + 700
    view_bottom, view_top = -500, h + 500
    
    width_inches = (view_right - view_left) / SCALE_FACTOR 
    height_inches = (view_top - view_bottom) / SCALE_FACTOR
    
    # ใช้ DPI 300 เพื่อความชัด และปิดการบีบอัดภาพ
    fig, ax = plt.subplots(figsize=(width_inches, height_inches), dpi=300)
    
    # 1. Concrete Face & Stirrup (ใช้ linewidth ที่คมชัด)
    ax.add_patch(patches.Rectangle((0, 0), b, h, linewidth=1.2, edgecolor='#000000', facecolor='#ffffff'))
    ax.add_patch(patches.Rectangle((cover, cover), b-2*cover, h-2*cover, linewidth=0.7, edgecolor='#4d5656', facecolor='none'))
    
    # 2. Rebar Drawing
    def draw_bars(n, y_pos, color):
        if n < 1: return
        spacing = (b - 2*cover - 2*ds - db) / (max(1, n - 1))
        for i in range(int(n)):
            x = (b/2) if n == 1 else (cover + ds + db/2 + i*spacing)
            ax.add_patch(plt.Circle((x, y_pos), db/2, color=color, zorder=10))

    y_bot, y_top = cover + ds + db/2, h - cover - ds - db/2
    draw_bars(n_bottom, y_bot, '#a93226')
    draw_bars(n_top, y_top, '#1f618d')
    
    # 3. Rebar Labels (ล็อค f_size=7 ให้เล็กและคม)
    f_size = 7
    ax.text(b + 60, y_top, f"{int(n_top)}-DB{int(db)}", va='center', color='#1f618d', fontweight='bold', fontsize=f_size)
    ax.text(b + 60, y_bot, f"{int(n_bottom)}-DB{int(db)}", va='center', color='#a93226', fontweight='bold', fontsize=f_size)
    ax.text(b/2, h + 100, f"{stirrup_name}", ha='center', color='#186a3b', fontweight='bold', fontsize=f_size)

    # 4. Engineering Ticks (มิติ)
    def draw_tick_dim(p1, p2, text, vert=False):
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='#000000', lw=0.5)
        tick = 15
        for p in [p1, p2]:
            ax.plot([p[0]-tick, p[0]+tick], [p[1]-tick, p[1]+tick], color='#000000', lw=0.8)
        if vert:
            ax.text(p1[0]-60, (p1[1]+p2[1])/2, text, va='center', ha='right', rotation=90, fontsize=f_size)
        else:
            ax.text((p1[0]+p2[0])/2, p1[1]-60, text, ha='center', va='top', fontsize=f_size)

    draw_tick_dim([0, -100], [b, -100], f"{int(b)}")
    draw_tick_dim([-100, 0], [-100, h], f"{int(h)}", vert=True)
    
    ax.set_xlim(view_left, view_right)
    ax.set_ylim(view_bottom, view_top)
    ax.set_aspect('equal')
    ax.axis('off')
    return fig

def plot_longitudinal_section_detailed(spans, sup_df, design_res, h_m, cover_mm):
    """
    Longitudinal Section: คมชัดสูง ตัวหนังสือจิ๋วเท่ากับรูปหน้าตัด
    """
    h_beam = h_m * 1000 
    total_L = sum(spans) * 1000
    offsets = [0] + list(np.cumsum(spans) * 1000)
    
    # บังคับความสูง Figsize ให้สัมพันธ์กับสเกล
    width_inches = (total_L + 1500) / SCALE_FACTOR
    height_inches = 3000 / SCALE_FACTOR 
    
    fig, ax = plt.subplots(figsize=(width_inches, height_inches), dpi=300)
    
    # คอนกรีต
    ax.add_patch(patches.Rectangle((0, 0), total_L, h_beam, linewidth=1.2, edgecolor='#000000', facecolor='#ffffff'))
    
    # Supports
    for _, sup in sup_df.iterrows():
        sx = sup['x'] * 1000
        stype = sup['type']
        if stype == 'Fixed':
            ax.add_patch(patches.Rectangle((sx-25, -120), 50, h_beam+240, facecolor='#f8f9f9', hatch='///', alpha=0.5, edgecolor='#95a5a6', lw=0.5))
        elif stype == 'Pin':
            poly = plt.Polygon([[sx-100, -120], [sx+100, -120], [sx, 0]], closed=True, facecolor='none', edgecolor='#000000', lw=0.8)
            ax.add_patch(poly)
            ax.plot([sx-130, sx+130], [-122, -122], color='black', lw=0.8)
        elif stype == 'Roller':
            poly = plt.Polygon([[sx-100, -100], [sx+100, -100], [sx, 0]], closed=True, facecolor='none', edgecolor='#000000', lw=0.8)
            ax.add_patch(poly)
            ax.add_patch(plt.Circle((sx, -130), 25, facecolor='none', edgecolor='#000000', lw=0.8))
            ax.plot([sx-130, sx+130], [-160, -160], color='black', lw=0.8)

    # Detailing (f_size=7)
    f_size = 7
    for i, span_l_m in enumerate(spans):
        L_mm, x_s, x_e = span_l_m * 1000, offsets[i], offsets[i+1]
        res = design_res[i]
        mid_x = (x_s + x_e) / 2
        
        ax.plot([x_s+40, x_e-40], [cover_mm, cover_mm], color='#a93226', lw=1.5)
        ax.text(mid_x, cover_mm + 40, f"{int(res['pos']['n'])}-DB{int(res['db'])}", ha='center', color='#a93226', fontsize=f_size, fontweight='bold')
        
        y_top = h_beam - cover_mm
        ax.plot([x_s, x_s + L_mm*0.3], [y_top, y_top], color='#1f618d', lw=1.5)
        ax.plot([x_e - L_mm*0.3, x_e], [y_top, y_top], color='#1f618d', lw=1.5)
        ax.text(x_s + 80, y_top + 40, f"{int(res['neg']['n'])}-DB{int(res['db'])}", ha='left', color='#1f618d', fontsize=f_size, fontweight='bold')
        
        ax.text(mid_x, h_beam + 120, f"RB6@{int(res['shear']['s'])}", ha='center', color='#186a3b', fontsize=f_size, fontweight='bold')

    # Dim Ticks
    dim_y = -700
    ax.plot([0, total_L], [dim_y, dim_y], color='#000000', lw=0.5)
    for px in [0, total_L]:
        ax.plot([px-25, px+25], [dim_y-25, dim_y+25], color='#000000', lw=0.8)
    ax.text(total_L/2, dim_y - 120, f"L = {total_L/1000} m", ha='center', fontweight='bold', fontsize=f_size)

    ax.set_xlim(-800, total_L + 800)
    ax.set_ylim(-1500, 1500)
    ax.set_aspect('equal')
    ax.axis('off')
    return fig
