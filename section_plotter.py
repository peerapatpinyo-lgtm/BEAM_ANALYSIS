import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def plot_section(b_m, h_m, cover_mm, db_main_mm, n_top, n_bottom, stirrup_name, fc, fy):
    """
    Cross Section: Blueprint Style - No arrows, clean extension lines.
    """
    b, h = b_m * 1000, h_m * 1000
    cover = cover_mm
    ds, db = 6, db_main_mm
    
    fig, ax = plt.subplots(figsize=(6, 7), dpi=100)
    
    # Concrete Face (Clean outline)
    ax.add_patch(patches.Rectangle((0, 0), b, h, linewidth=1.2, edgecolor='#1a1a1a', facecolor='#ffffff'))
    
    # Stirrup
    ax.add_patch(patches.Rectangle((cover, cover), b-2*cover, h-2*cover, linewidth=1, edgecolor='#2c3e50', facecolor='none'))
    
    def draw_bars(n, y_pos, color):
        if n < 1: return
        spacing = (b - 2*cover - 2*ds - db) / (max(1, n - 1))
        for i in range(int(n)):
            x = (b/2) if n == 1 else (cover + ds + db/2 + i*spacing)
            ax.add_patch(plt.Circle((x, y_pos), db/2, color=color, zorder=10))

    y_bot = cover + ds + db/2
    y_top = h - cover - ds - db/2
    draw_bars(n_bottom, y_bot, '#8e1c19')
    draw_bars(n_top, y_top, '#1a5276')
    
    # --- Clean Labels (No Arrows) ---
    # Top Bars: วางเส้นระดับบางๆ ชี้ไปที่ข้อความ
    ax.plot([b-20, b+40], [y_top, y_top], color='#1a5276', lw=0.5, alpha=0.5)
    ax.text(b + 50, y_top, f"{int(n_top)}-DB{int(db)}", va='center', color='#1a5276', fontweight='bold', fontsize=10)
    
    # Bottom Bars:
    ax.plot([b-20, b+40], [y_bot, y_bot], color='#8e1c19', lw=0.5, alpha=0.5)
    ax.text(b + 50, y_bot, f"{int(n_bottom)}-DB{int(db)}", va='center', color='#8e1c19', fontweight='bold', fontsize=10)

    # Stirrup Label:
    ax.text(b/2, h + 40, f"STIRRUP: {stirrup_name}", ha='center', color='#1d8348', fontsize=9, fontweight='bold')

    # Dimensions (Architecture Ticks)
    def draw_dim(p1, p2, text, vert=False):
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='black', lw=0.7)
        if vert:
            ax.text(p1[0]-35, (p1[1]+p2[1])/2, text, va='center', ha='right', rotation=90, fontsize=9)
        else:
            ax.text((p1[0]+p2[0])/2, p1[1]-35, text, ha='center', va='top', fontsize=9)

    draw_dim([0, -30], [b, -30], f"B={int(b)}")
    draw_dim([-30, 0], [-30, h], f"H={int(h)}", vert=True)
    
    ax.set_xlim(-150, b + 250)
    ax.set_ylim(-150, h + 150)
    ax.set_aspect('equal')
    ax.axis('off')
    return fig

def plot_longitudinal_section_detailed(spans, sup_df, design_res, h_m, cover_mm):
    """
    Longitudinal: Professional Structural Profile with Realistic Supports.
    """
    h_beam = h_m * 1000 
    total_L = sum(spans) * 1000
    offsets = [0] + list(np.cumsum(spans) * 1000)
    
    fig, ax = plt.subplots(figsize=(15, 5), dpi=100)
    
    # 1. Beam Outline (Double line for top face)
    ax.add_patch(patches.Rectangle((0, 0), total_L, h_beam, lw=1.5, edgecolor='#1a1a1a', facecolor='#ffffff'))
    
    # 2. Realistic Structural Supports (Reinforced Concrete Pillar Style)
    for _, sup in sup_df.iterrows():
        sx = sup['x'] * 1000
        # Pillar Body
        ax.add_patch(patches.Rectangle((sx-80, -h_beam*0.8), 160, h_beam*0.8, facecolor='#f2f2f2', edgecolor='#7f8c8d', lw=0.8))
        # Pillar Hatch (Small dots)
        ax.add_patch(patches.Rectangle((sx-80, -h_beam*0.8), 160, h_beam*0.8, facecolor='none', hatch='...', alpha=0.2))
        # Support Label
        ax.text(sx, -h_beam*1.1, f"C-{sup['type'].upper()}", ha='center', fontsize=8, fontweight='bold', color='#2c3e50')

    # 3. Bar Detailing
    for i, span_l_m in enumerate(spans):
        L_mm = span_l_m * 1000
        x_s, x_e = offsets[i], offsets[i+1]
        res = design_res[i]
        
        # Bottom Steel
        ax.plot([x_s+20, x_e-20], [cover_mm, cover_mm], color='#8e1c19', lw=2, solid_capstyle='butt')
        ax.text((x_s+x_e)/2, cover_mm + 25, f"{int(res['pos']['n'])}-DB{int(res['db'])}", 
                ha='center', color='#8e1c19', fontsize=9, fontweight='bold')

        # Top Steel (Full Visibility)
        y_top = h_beam - cover_mm
        ax.plot([x_s, x_s + L_mm*0.3], [y_top, y_top], color='#1a5276', lw=2)
        ax.plot([x_e - L_mm*0.3, x_e], [y_top, y_top], color='#1a5276', lw=2)
        
        # Label Top Bars at supports
        ax.text(x_s + 20, y_top + 30, f"{int(res['neg']['n'])}-DB{int(res['db'])}", 
                ha='left', color='#1a5276', fontsize=8, fontweight='bold')

        # Stirrup Info
        ax.text((x_s+x_e)/2, h_beam + 50, f"STIRRUP: RB6@{int(res['shear']['s'])}", 
                ha='center', color='#1d8348', fontsize=8, fontweight='bold')

    # 4. Total Dimension
    ax.annotate('', xy=(0, -h_beam*0.4), xytext=(total_L, -h_beam*0.4), arrowprops=dict(arrowstyle='|-|', color='black', lw=0.8))
    ax.text(total_L/2, -h_beam*0.6, f"TOTAL LENGTH: {total_L/1000} m", ha='center', fontsize=9, fontweight='bold')

    ax.set_xlim(-400, total_L + 400)
    ax.set_ylim(-h_beam*1.5, h_beam + 200)
    ax.set_aspect('equal')
    ax.axis('off')
    return fig
