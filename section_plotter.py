import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# --- Global Scale Configuration ---
# กำหนดค่าคงที่เพื่อให้ 500mm ในพิกัดจริง = 1 นิ้วในหน้าจอ (ช่วยให้สเกล 2 รูปเท่ากัน)
SCALE_FACTOR = 500 

def plot_section(b_m, h_m, cover_mm, db_main_mm, n_top, n_bottom, stirrup_name, fc, fy):
    """
    Professional Cross Section: มาตราส่วนเท่ากับ Long Section, ใช้ Ticks, ชี้บอกเหล็กข้างรูป
    """
    b, h = b_m * 1000, h_m * 1000
    cover = cover_mm
    ds, db = 6, db_main_mm
    
    # คำนวณขนาด Figsize ให้สัมพันธ์กับสัดส่วนจริง
    width_inches = (b + 1000) / SCALE_FACTOR 
    height_inches = (h + 1200) / SCALE_FACTOR
    
    fig, ax = plt.subplots(figsize=(width_inches, height_inches), dpi=100)
    
    # 1. Concrete Face
    ax.add_patch(patches.Rectangle((0, 0), b, h, linewidth=2, edgecolor='#1a1a1a', facecolor='#ffffff'))
    
    # 2. Stirrup
    ax.add_patch(patches.Rectangle((cover, cover), b-2*cover, h-2*cover, linewidth=1.5, edgecolor='#2c3e50', facecolor='none'))
    
    # 3. Main Bars
    def draw_bars(n, y_pos, color):
        if n < 1: return
        spacing = (b - 2*cover - 2*ds - db) / (max(1, n - 1))
        for i in range(int(n)):
            x = (b/2) if n == 1 else (cover + ds + db/2 + i*spacing)
            ax.add_patch(plt.Circle((x, y_pos), db/2, color=color, zorder=10))

    y_bot, y_top = cover + ds + db/2, h - cover - ds - db/2
    draw_bars(n_bottom, y_bot, '#8e1c19')
    draw_bars(n_top, y_top, '#1a5276')
    
    # 4. Rebar Callouts (วางข้างๆ ระดับเดียวกับตัวเหล็ก)
    ax.text(b + 80, y_top, f"{int(n_top)}-DB{int(db)}", va='center', color='#1a5276', fontweight='bold', fontsize=12)
    ax.text(b + 80, y_bot, f"{int(n_bottom)}-DB{int(db)}", va='center', color='#8e1c19', fontweight='bold', fontsize=12)
    ax.text(b/2, h + 150, f"{stirrup_name}", ha='center', color='#1d8348', fontsize=12, fontweight='bold')

    # 5. Engineering Ticks Dimension
    def draw_tick_dim(p1, p2, text, vert=False):
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='black', lw=1.2)
        tick = 30
        for p in [p1, p2]:
            ax.plot([p[0]-tick, p[0]+tick], [p[1]-tick, p[1]+tick], color='black', lw=2)
        if vert:
            ax.text(p1[0]-80, (p1[1]+p2[1])/2, text, va='center', ha='right', rotation=90, fontsize=12)
        else:
            ax.text((p1[0]+p2[0])/2, p1[1]-80, text, ha='center', va='top', fontsize=12)

    draw_tick_dim([0, -150], [b, -150], f"{int(b)}")
    draw_tick_dim([-150, 0], [-150, h], f"{int(h)}", vert=True)
    
    ax.set_xlim(-500, b + 600)
    ax.set_ylim(-600, h + 600)
    ax.set_aspect('equal')
    ax.axis('off')
    return fig

def plot_longitudinal_section_detailed(spans, sup_df, design_res, h_m, cover_mm):
    """
    Professional Longitudinal Section: สัญลักษณ์ Support ครบถ้วน, สเกลเท่ากับ Cross Section
    """
    h_beam = h_m * 1000 
    total_L = sum(spans) * 1000
    offsets = [0] + list(np.cumsum(spans) * 1000)
    
    width_inches = (total_L + 1200) / SCALE_FACTOR
    height_inches = (h_beam + 1500) / SCALE_FACTOR
    
    fig, ax = plt.subplots(figsize=(width_inches, height_inches), dpi=100)
    
    # 1. Beam Outline
    ax.add_patch(patches.Rectangle((0, 0), total_L, h_beam, linewidth=2, edgecolor='#1a1a1a', facecolor='#ffffff'))
    
    # 2. Structural Support Symbols
    for _, sup in sup_df.iterrows():
        sx = sup['x'] * 1000
        stype = sup['type']
        if stype == 'Fixed':
            ax.add_patch(patches.Rectangle((sx-30, -150), 60, h_beam+300, facecolor='#bdc3c7', hatch='///', alpha=0.6))
        elif stype == 'Pin':
            poly = plt.Polygon([[sx-120, -150], [sx+120, -150], [sx, 0]], closed=True, facecolor='none', edgecolor='black', lw=1.5)
            ax.add_patch(poly)
            ax.plot([sx-150, sx+150], [-155, -155], color='black', lw=2)
        elif stype == 'Roller':
            poly = plt.Polygon([[sx-120, -120], [sx+120, -120], [sx, 0]], closed=True, facecolor='none', edgecolor='black', lw=1.5)
            ax.add_patch(poly)
            ax.add_patch(plt.Circle((sx, -150), 30, facecolor='none', edgecolor='black', lw=1.5))
            ax.plot([sx-150, sx+150], [-185, -185], color='black', lw=2)

    # 3. Bar Detailing
    for i, span_l_m in enumerate(spans):
        L_mm, x_s, x_e = span_l_m * 1000, offsets[i], offsets[i+1]
        res = design_res[i]
        mid_x = (x_s + x_e) / 2
        
        # Bottom Bars
        ax.plot([x_s+50, x_e-50], [cover_mm, cover_mm], color='#8e1c19', lw=3)
        ax.text(mid_x, cover_mm + 50, f"{int(res['pos']['n'])}-DB{int(res['db'])}", ha='center', color='#8e1c19', fontsize=12, fontweight='bold')
        
        # Top Bars (Curtailment 0.3L)
        y_top = h_beam - cover_mm
        ax.plot([x_s, x_s + L_mm*0.3], [y_top, y_top], color='#1a5276', lw=3)
        ax.plot([x_e - L_mm*0.3, x_e], [y_top, y_top], color='#1a5276', lw=3)
        ax.text(x_s + 100, y_top + 50, f"{int(res['neg']['n'])}-DB{int(res['db'])}", ha='left', color='#1a5276', fontsize=12, fontweight='bold')
        
        # Stirrup Label
        ax.text(mid_x, h_beam + 150, f"RB6@{int(res['shear']['s'])}", ha='center', color='#1d8348', fontsize=12, fontweight='bold')

    # 4. Total Length Dimension (Engineering Ticks)
    dim_y = -h_beam * 1.2
    ax.plot([0, total_L], [dim_y, dim_y], color='black', lw=1.2)
    for px in [0, total_L]:
        ax.plot([px-30, px+30], [dim_y-30, dim_y+30], color='black', lw=2)
    ax.text(total_L/2, dim_y - 150, f"L = {total_L/1000} m", ha='center', fontweight='bold', fontsize=14)

    ax.set_xlim(-600, total_L + 600)
    ax.set_ylim(-h_beam*2.2, h_beam + 800)
    ax.set_aspect('equal')
    ax.axis('off')
    return fig
