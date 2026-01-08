import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def plot_section(b_m, h_m, cover_mm, db_main_mm, n_top, n_bottom, stirrup_name, fc, fy):
    """
    Matched-Scale Cross Section
    """
    b, h = b_m * 1000, h_m * 1000
    cover = cover_mm
    ds, db = 6, db_main_mm
    
    # คำนวณ figsize ให้สัมพันธ์กับสัดส่วนจริง (กว้าง 4 นิ้ว เป็นเกณฑ์)
    fig_w = 4
    fig_h = fig_w * (h + 300) / (b + 400)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=100)
    
    ax.add_patch(patches.Rectangle((0, 0), b, h, linewidth=1.5, edgecolor='#1a1a1a', facecolor='#ffffff'))
    ax.add_patch(patches.Rectangle((cover, cover), b-2*cover, h-2*cover, linewidth=1.2, edgecolor='#2c3e50', facecolor='none'))
    
    def draw_bars(n, y_pos, color):
        if n < 1: return
        spacing = (b - 2*cover - 2*ds - db) / (max(1, n - 1))
        for i in range(int(n)):
            x = (b/2) if n == 1 else (cover + ds + db/2 + i*spacing)
            ax.add_patch(plt.Circle((x, y_pos), db/2, color=color, zorder=10))

    y_bot, y_top = cover + ds + db/2, h - cover - ds - db/2
    draw_bars(n_bottom, y_bot, '#8e1c19')
    draw_bars(n_top, y_top, '#1a5276')
    
    # Text: ใช้ขนาดฟอนต์ 10 เท่ากันทั้งสองรูป
    ax.text(b + 50, y_top, f"{int(n_top)}-DB{int(db)}", va='center', color='#1a5276', fontweight='bold', fontsize=10)
    ax.text(b + 50, y_bot, f"{int(n_bottom)}-DB{int(db)}", va='center', color='#8e1c19', fontweight='bold', fontsize=10)
    ax.text(b/2, h + 80, f"{stirrup_name}", ha='center', color='#1d8348', fontsize=10, fontweight='bold')

    # Dimension Ticks
    def draw_tick_dim(p1, p2, text, vert=False):
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='black', lw=1)
        tick = 15
        for p in [p1, p2]:
            ax.plot([p[0]-tick, p[0]+tick], [p[1]-tick, p[1]+tick], color='black', lw=1.5)
        if vert:
            ax.text(p1[0]-50, (p1[1]+p2[1])/2, text, va='center', ha='right', rotation=90, fontsize=10)
        else:
            ax.text((p1[0]+p2[0])/2, p1[1]-50, text, ha='center', va='top', fontsize=10)

    draw_tick_dim([0, -60], [b, -60], f"{int(b)}")
    draw_tick_dim([-60, 0], [-60, h], f"{int(h)}", vert=True)
    
    ax.set_xlim(-200, b + 350)
    ax.set_ylim(-200, h + 200)
    ax.set_aspect('equal')
    ax.axis('off')
    return fig

def plot_longitudinal_section_detailed(spans, sup_df, design_res, h_m, cover_mm):
    """
    Matched-Scale Longitudinal Section
    """
    h_beam = h_m * 1000 
    total_L = sum(spans) * 1000
    offsets = [0] + list(np.cumsum(spans) * 1000)
    
    # บังคับความกว้างให้ยาวตามจำนวนสแปน เพื่อให้สเกลคงที่
    fig_w = len(spans) * 5
    fig_h = fig_w * (h_beam + 600) / (total_L + 600)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=100)
    
    ax.add_patch(patches.Rectangle((0, 0), total_L, h_beam, lw=1.5, edgecolor='#1a1a1a', facecolor='#ffffff'))
    
    # Support Icons
    for _, sup in sup_df.iterrows():
        sx = sup['x'] * 1000
        stype = sup['type']
        if stype == 'Fixed':
            ax.add_patch(patches.Rectangle((sx-25, -h_beam*0.2), 50, h_beam*1.4, facecolor='#bdc3c7', hatch='///', alpha=0.5))
        elif stype == 'Pin':
            poly = plt.Polygon([[sx-100, -120], [sx+100, -120], [sx, 0]], closed=True, facecolor='none', edgecolor='black', lw=1.2)
            ax.add_patch(poly)
            ax.plot([sx-120, sx+120], [-125, -125], color='black', lw=1.2)
        elif stype == 'Roller':
            poly = plt.Polygon([[sx-100, -100], [sx+100, -100], [sx, 0]], closed=True, facecolor='none', edgecolor='black', lw=1.2)
            ax.add_patch(poly)
            ax.add_patch(plt.Circle((sx, -125), 25, facecolor='none', edgecolor='black', lw=1.2))
            ax.plot([sx-120, sx+120], [-155, -155], color='black', lw=1.2)

    for i, span_l_m in enumerate(spans):
        L_mm, x_s, x_e = span_l_m * 1000, offsets[i], offsets[i+1]
        res = design_res[i]
        mid_x = (x_s + x_e) / 2
        
        # Bottom Steel
        ax.plot([x_s+30, x_e-30], [cover_mm, cover_mm], color='#8e1c19', lw=2.5)
        ax.text(mid_x, cover_mm + 40, f"{int(res['pos']['n'])}-DB{int(res['db'])}", ha='center', color='#8e1c19', fontsize=10, fontweight='bold')
        
        # Top Steel
        y_top = h_beam - cover_mm
        ax.plot([x_s, x_s + L_mm*0.3], [y_top, y_top], color='#1a5276', lw=2.5)
        ax.plot([x_e - L_mm*0.3, x_e], [y_top, y_top], color='#1a5276', lw=2.5)
        ax.text(x_s + 50, y_top + 40, f"{int(res['neg']['n'])}-DB{int(res['db'])}", ha='left', color='#1a5276', fontsize=10, fontweight='bold')
        
        # Stirrup
        ax.text(mid_x, h_beam + 100, f"RB6@{int(res['shear']['s'])}", ha='center', color='#1d8348', fontsize=10, fontweight='bold')

    # Dim Ticks
    ax.plot([0, total_L], [-h_beam*0.8, -h_beam*0.8], color='black', lw=1)
    for px in [0, total_L]:
        ax.plot([px-20, px+20], [-h_beam*0.8-20, -h_beam*0.8+20], color='black', lw=1.5)
    ax.text(total_L/2, -h_beam*1.1, f"Total L = {total_L/1000} m", ha='center', fontweight='bold', fontsize=11)

    ax.set_xlim(-500, total_L + 500)
    ax.set_ylim(-h_beam*1.5, h_beam + 400)
    ax.set_aspect('equal')
    ax.axis('off')
    return fig
