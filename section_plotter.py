import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# --- 📏 Master Scale Configuration ---
# กำหนดค่าคงที่เพื่อให้ 800mm ในพิกัดจริง = 1 นิ้วในหน้าจอ 
# ช่วยให้ตัวหนังสือและเส้นในทั้ง 2 รูปมีขนาดเท่ากันเมื่อแสดงผล
SCALE_FACTOR = 800 

def plot_section(b_m, h_m, cover_mm, db_main_mm, n_top, n_bottom, stirrup_name, fc, fy):
    """
    Cross Section: ซูมเข้าให้เห็นหน้าตัดชัดเจน แต่คงขนาด Font ให้เท่ากับรูป Long Section
    """
    b, h = b_m * 1000, h_m * 1000
    cover = cover_mm
    ds, db = 6, db_main_mm
    
    # 📐 คำนวณขอบเขตการมองเห็น (Viewport) ให้กระชับกับตัวคาน
    view_left, view_right = -300, b + 800
    view_bottom, view_top = -400, h + 400
    
    # คำนวณ Figsize ให้สัมพันธ์กับสัดส่วนจริง
    width_inches = (view_right - view_left) / SCALE_FACTOR 
    height_inches = (view_top - view_bottom) / SCALE_FACTOR
    
    fig, ax = plt.subplots(figsize=(width_inches, height_inches), dpi=100)
    
    # 1. Concrete Outline
    ax.add_patch(patches.Rectangle((0, 0), b, h, linewidth=1.2, edgecolor='#1a1a1a', facecolor='#ffffff'))
    
    # 2. Stirrup
    ax.add_patch(patches.Rectangle((cover, cover), b-2*cover, h-2*cover, linewidth=0.8, edgecolor='#34495e', facecolor='none'))
    
    # 3. Main Bars
    def draw_bars(n, y_pos, color):
        if n < 1: return
        spacing = (b - 2*cover - 2*ds - db) / (max(1, n - 1))
        for i in range(int(n)):
            x = (b/2) if n == 1 else (cover + ds + db/2 + i*spacing)
            ax.add_patch(plt.Circle((x, y_pos), db/2, color=color, zorder=10))

    y_bot, y_top = cover + ds + db/2, h - cover - ds - db/2
    draw_bars(n_bottom, y_bot, '#a93226')
    draw_bars(n_top, y_top, '#1f618d')
    
    # 4. Rebar Labels (Font 9 เท่ากับรูป Long Section)
    f_size = 9
    ax.text(b + 60, y_top, f"{int(n_top)}-DB{int(db)}", va='center', color='#1f618d', fontweight='bold', fontsize=f_size)
    ax.text(b + 60, y_bot, f"{int(n_bottom)}-DB{int(db)}", va='center', color='#a93226', fontweight='bold', fontsize=f_size)
    ax.text(b/2, h + 120, f"{stirrup_name}", ha='center', color='#1d8348', fontweight='bold', fontsize=f_size)

    # 5. Engineering Ticks (สัดส่วนเดียวกับรูปใหญ่)
    def draw_tick_dim(p1, p2, text, vert=False):
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='black', lw=0.6)
        tick = 20
        for p in [p1, p2]:
            ax.plot([p[0]-tick, p[0]+tick], [p[1]-tick, p[1]+tick], color='black', lw=0.8)
        if vert:
            ax.text(p1[0]-60, (p1[1]+p2[1])/2, text, va='center', ha='right', rotation=90, fontsize=f_size-1)
        else:
            ax.text((p1[0]+p2[0])/2, p1[1]-60, text, ha='center', va='top', fontsize=f_size-1)

    draw_tick_dim([0, -120], [b, -120], f"{int(b)}")
    draw_tick_dim([-120, 0], [-120, h], f"{int(h)}", vert=True)
    
    ax.set_xlim(view_left, view_right)
    ax.set_ylim(view_bottom, view_top)
    ax.set_aspect('equal')
    ax.axis('off')
    return fig

def plot_longitudinal_section_detailed(spans, sup_df, design_res, h_m, cover_mm):
    """
    Longitudinal Section: แสดงรายละเอียดเหล็กและสัญลักษณ์ Support ตามประเภทจริง
    """
    h_beam = h_m * 1000 
    total_L = sum(spans) * 1000
    offsets = [0] + list(np.cumsum(spans) * 1000)
    
    # 📏 คำนวณขนาด Figsize ให้สัมพันธ์กับ SCALE_FACTOR (ความสูงอ้างอิงที่ 4000mm)
    width_inches = (total_L + 2000) / SCALE_FACTOR
    height_inches = 4000 / SCALE_FACTOR 
    
    fig, ax = plt.subplots(figsize=(width_inches, height_inches), dpi=100)
    
    # 1. Beam Outline
    ax.add_patch(patches.Rectangle((0, 0), total_L, h_beam, linewidth=1.2, edgecolor='#1a1a1a', facecolor='#ffffff'))
    
    # 2. Support Drawing (วาดตามประเภท Pin, Roller, Fixed)
    for _, sup in sup_df.iterrows():
        sx = sup['x'] * 1000
        stype = sup['type']
        if stype == 'Fixed':
            ax.add_patch(patches.Rectangle((sx-30, -200), 60, h_beam+400, facecolor='#f2f4f4', hatch='///', alpha=0.4, edgecolor='#7f8c8d', lw=0.6))
        elif stype == 'Pin':
            poly = plt.Polygon([[sx-120, -150], [sx+120, -150], [sx, 0]], closed=True, facecolor='none', edgecolor='#1a1a1a', lw=1)
            ax.add_patch(poly)
            ax.plot([sx-150, sx+150], [-155, -155], color='black', lw=1)
        elif stype == 'Roller':
            poly = plt.Polygon([[sx-120, -120], [sx+120, -120], [sx, 0]], closed=True, facecolor='none', edgecolor='#1a1a1a', lw=1)
            ax.add_patch(poly)
            ax.add_patch(plt.Circle((sx, -150), 30, facecolor='none', edgecolor='#1a1a1a', lw=1))
            ax.plot([sx-150, sx+150], [-185, -185], color='black', lw=1)

    # 3. Bar Detailing
    f_size = 9
    for i, span_l_m in enumerate(spans):
        L_mm, x_s, x_e = span_l_m * 1000, offsets[i], offsets[i+1]
        res = design_res[i]
        mid_x = (x_s + x_e) / 2
        
        # Bottom Steel
        ax.plot([x_s+50, x_e-50], [cover_mm, cover_mm], color='#a93226', lw=1.8)
        ax.text(mid_x, cover_mm + 60, f"{int(res['pos']['n'])}-DB{int(res['db'])}", ha='center', color='#a93226', fontsize=f_size, fontweight='bold')
        
        # Top Steel (L/3 Curtailment)
        y_top = h_beam - cover_mm
        ax.plot([x_s, x_s + L_mm*0.3], [y_top, y_top], color='#1f618d', lw=1.8)
        ax.plot([x_e - L_mm*0.3, x_e], [y_top, y_top], color='#1f618d', lw=1.8)
        ax.text(x_s + 100, y_top + 60, f"{int(res['neg']['n'])}-DB{int(res['db'])}", ha='left', color='#1f618d', fontsize=f_size, fontweight='bold')
        
        # Stirrup
        ax.text(mid_x, h_beam + 150, f"RB6@{int(res['shear']['s'])}", ha='center', color='#1d8348', fontsize=f_size, fontweight='bold')

    # 4. Dimension Ticks
    dim_y = -800
    ax.plot([0, total_L], [dim_y, dim_y], color='black', lw=0.6)
    for px in [0, total_L]:
        ax.plot([px-30, px+30], [dim_y-30, dim_y+30], color='black', lw=0.8)
    ax.text(total_L/2, dim_y - 150, f"L = {total_L/1000} m", ha='center', fontweight='bold', fontsize=f_size+1)

    ax.set_xlim(-1000, total_L + 1000)
    ax.set_ylim(-2000, 2000)
    ax.set_aspect('equal')
    ax.axis('off')
    return fig
