import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# --- ปรับจูนเพื่อความ Sharp (ยิ่งค่ามาก รูปยิ่งดูละเอียดและเล็กลง) ---
SCALE_FACTOR = 800 

def plot_section(b_m, h_m, cover_mm, db_main_mm, n_top, n_bottom, stirrup_name, fc, fy):
    b, h = b_m * 1000, h_m * 1000
    cover = cover_mm
    ds, db = 6, db_main_mm
    
    width_inches = (b + 1200) / SCALE_FACTOR 
    height_inches = (h + 1200) / SCALE_FACTOR
    
    fig, ax = plt.subplots(figsize=(width_inches, height_inches), dpi=120) # เพิ่ม DPI เพื่อความชัด
    
    # Concrete Face (เส้นบางลงเหลือ 1.2)
    ax.add_patch(patches.Rectangle((0, 0), b, h, linewidth=1.2, edgecolor='#1a1a1a', facecolor='#ffffff'))
    
    # Stirrup (เส้นบาง 0.8)
    ax.add_patch(patches.Rectangle((cover, cover), b-2*cover, h-2*cover, linewidth=0.8, edgecolor='#34495e', facecolor='none'))
    
    def draw_bars(n, y_pos, color):
        if n < 1: return
        spacing = (b - 2*cover - 2*ds - db) / (max(1, n - 1))
        for i in range(int(n)):
            x = (b/2) if n == 1 else (cover + ds + db/2 + i*spacing)
            ax.add_patch(plt.Circle((x, y_pos), db/2, color=color, zorder=10))

    y_bot, y_top = cover + ds + db/2, h - cover - ds - db/2
    draw_bars(n_bottom, y_bot, '#a93226')
    draw_bars(n_top, y_top, '#1f618d')
    
    # Labels (ลดขนาด Font เหลือ 9)
    txt_opt = {'fontweight': 'bold', 'fontsize': 9}
    ax.text(b + 60, y_top, f"{int(n_top)}-DB{int(db)}", va='center', color='#1f618d', **txt_opt)
    ax.text(b + 60, y_bot, f"{int(n_bottom)}-DB{int(db)}", va='center', color='#a93226', **txt_opt)
    ax.text(b/2, h + 120, f"{stirrup_name}", ha='center', color='#1d8348', **txt_opt)

    # Engineering Ticks (เส้นบางและ Tick เล็กลง)
    def draw_tick_dim(p1, p2, text, vert=False):
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='black', lw=0.6)
        tick = 20
        for p in [p1, p2]:
            ax.plot([p[0]-tick, p[0]+tick], [p[1]-tick, p[1]+tick], color='black', lw=1)
        if vert:
            ax.text(p1[0]-60, (p1[1]+p2[1])/2, text, va='center', ha='right', rotation=90, fontsize=8)
        else:
            ax.text((p1[0]+p2[0])/2, p1[1]-60, text, ha='center', va='top', fontsize=8)

    draw_tick_dim([0, -120], [b, -120], f"{int(b)}")
    draw_tick_dim([-120, 0], [-120, h], f"{int(h)}", vert=True)
    
    ax.set_xlim(-400, b + 600)
    ax.set_ylim(-500, h + 500)
    ax.set_aspect('equal')
    ax.axis('off')
    return fig

def plot_longitudinal_section_detailed(spans, sup_df, design_res, h_m, cover_mm):
    h_beam = h_m * 1000 
    total_L = sum(spans) * 1000
    offsets = [0] + list(np.cumsum(spans) * 1000)
    
    width_inches = (total_L + 1500) / SCALE_FACTOR
    height_inches = (h_beam + 2000) / SCALE_FACTOR
    
    fig, ax = plt.subplots(figsize=(width_inches, height_inches), dpi=120)
    
    # Beam Outline (เส้นบาง 1.2)
    ax.add_patch(patches.Rectangle((0, 0), total_L, h_beam, linewidth=1.2, edgecolor='#1a1a1a', facecolor='#ffffff'))
    
    # Support Symbols (เส้นบางลง)
    for _, sup in sup_df.iterrows():
        sx = sup['x'] * 1000
        stype = sup['type']
        if stype == 'Fixed':
            ax.add_patch(patches.Rectangle((sx-20, -100), 40, h_beam+200, facecolor='#f2f4f4', hatch='///', alpha=0.4, edgecolor='#7f8c8d', lw=0.5))
        elif stype == 'Pin':
            poly = plt.Polygon([[sx-100, -120], [sx+100, -120], [sx, 0]], closed=True, facecolor='none', edgecolor='#1a1a1a', lw=1)
            ax.add_patch(poly)
            ax.plot([sx-120, sx+120], [-125, -125], color='black', lw=1)
        elif stype == 'Roller':
            poly = plt.Polygon([[sx-100, -100], [sx+100, -100], [sx, 0]], closed=True, facecolor='none', edgecolor='#1a1a1a', lw=1)
            ax.add_patch(poly)
            ax.add_patch(plt.Circle((sx, -130), 25, facecolor='none', edgecolor='#1a1a1a', lw=1))
            ax.plot([sx-120, sx+120], [-160, -160], color='black', lw=1)

    # Bar Detailing
    for i, span_l_m in enumerate(spans):
        L_mm, x_s, x_e = span_l_m * 1000, offsets[i], offsets[i+1]
        res = design_res[i]
        mid_x = (x_s + x_e) / 2
        
        # Steel Bars (ลดความหนาเส้นเหล็กเหลือ 1.8)
        ax.plot([x_s+50, x_e-50], [cover_mm, cover_mm], color='#a93226', lw=1.8)
        ax.text(mid_x, cover_mm + 50, f"{int(res['pos']['n'])}-DB{int(res['db'])}", ha='center', color='#a93226', fontsize=9, fontweight='bold')
        
        y_top = h_beam - cover_mm
        ax.plot([x_s, x_s + L_mm*0.3], [y_top, y_top], color='#1f618d', lw=1.8)
        ax.plot([x_e - L_mm*0.3, x_e], [y_top, y_top], color='#1f618d', lw=1.8)
        ax.text(x_s + 80, y_top + 40, f"{int(res['neg']['n'])}-DB{int(res['db'])}", ha='left', color='#1f618d', fontsize=9, fontweight='bold')
        
        ax.text(mid_x, h_beam + 120, f"RB6@{int(res['shear']['s'])}", ha='center', color='#1d8348', fontsize=9, fontweight='bold')

    # Total Length Dimension
    dim_y = -h_beam * 1.5
    ax.plot([0, total_L], [dim_y, dim_y], color='black', lw=0.6)
    for px in [0, total_L]:
        ax.plot([px-25, px+25], [dim_y-25, dim_y+25], color='black', lw=0.8)
    ax.text(total_L/2, dim_y - 120, f"L = {total_L/1000} m", ha='center', fontweight='bold', fontsize=10)

    ax.set_xlim(-800, total_L + 800)
    ax.set_ylim(-h_beam*3, h_beam + 1000)
    ax.set_aspect('equal')
    ax.axis('off')
    return fig
