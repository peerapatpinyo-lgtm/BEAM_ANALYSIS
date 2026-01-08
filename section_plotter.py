import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def plot_section(b_m, h_m, cover_mm, db_main_mm, n_top, n_bottom, stirrup_name, fc, fy):
    """
    Cross Section - มาตรฐานเดิมที่ถูกต้อง
    """
    b, h = b_m * 1000, h_m * 1000
    cover = cover_mm
    ds, db = 6, db_main_mm
    
    fig, ax = plt.subplots(figsize=(5, 7))
    ax.add_patch(patches.Rectangle((0, 0), b, h, linewidth=2, edgecolor='black', facecolor='#f8f9fa', hatch='///', alpha=0.2))
    ax.add_patch(patches.Rectangle((cover, cover), b-2*cover, h-2*cover, linewidth=1.5, edgecolor='#2c3e50', facecolor='none'))
    
    def draw_bars(n, y_pos, color):
        if n < 1: return
        if n == 1:
            ax.add_patch(plt.Circle((b/2, y_pos), db/2, color=color, zorder=5))
        else:
            spacing = (b - 2*cover - 2*ds - db) / (n - 1)
            for i in range(n):
                ax.add_patch(plt.Circle((cover + ds + db/2 + i*spacing, y_pos), db/2, color=color, zorder=5))

    draw_bars(n_bottom, cover + ds + db/2, '#c0392b')
    draw_bars(n_top, h - cover - ds - db/2, '#2980b9')
    
    info_text = f"SECTION DETAIL\n{int(b)}x{int(h)} mm\nfc': {fc}\nTop: {int(n_top)}-DB{int(db)}\nBot: {int(n_bottom)}-DB{int(db)}"
    ax.text(b/2, h + 50, info_text, ha='center', va='bottom', family='monospace', fontsize=9, bbox=dict(facecolor='white', alpha=0.8, edgecolor='#bdc3c7'))
    
    ax.set_xlim(-150, b + 150)
    ax.set_ylim(-150, h + 250)
    ax.set_aspect('equal')
    ax.axis('off')
    return fig

def plot_longitudinal_section_detailed(spans, sup_df, design_res, h_m, cover_mm):
    """
    Professional Longitudinal Detail - แก้ปัญหาตัวหนังสือทับกัน
    """
    h_beam = h_m * 1000 
    total_L = sum(spans) * 1000
    offsets = [0] + list(np.cumsum(spans) * 1000)
    
    fig, ax = plt.subplots(figsize=(16, 6))
    
    # 1. Beam Outline
    ax.add_patch(patches.Rectangle((0, 0), total_L, h_beam, linewidth=2, edgecolor='#2c3e50', facecolor='#ffffff', zorder=1))

    # 2. Support Icons
    for _, sup in sup_df.iterrows():
        sx = sup['x'] * 1000
        stype = sup['type']
        if stype == 'Fixed':
            ax.add_patch(patches.Rectangle((sx-25, -20), 50, h_beam+40, facecolor='#bdc3c7', hatch='///', alpha=0.6))
        else:
            poly = plt.Polygon([[sx-100, -100], [sx+100, -100], [sx, 0]], closed=True, facecolor='#f8f9fa', edgecolor='black', lw=1.2)
            ax.add_patch(poly)

    # 3. Reinforcement & Non-Overlapping Labels
    for i, span_l_m in enumerate(spans):
        L_mm = span_l_m * 1000
        x_s, x_e = offsets[i], offsets[i+1]
        mid_x = (x_s + x_e) / 2
        res = design_res[i]
        
        # --- เหล็กล่าง (วาง Label ใต้คาน) ---
        ax.plot([x_s + 40, x_e - 40], [cover_mm, cover_mm], color='#c0392b', lw=2.5, solid_capstyle='round')
        ax.text(mid_x, cover_mm - 60, f"{int(res['pos']['n'])}-DB{int(res['db'])}", 
                ha='center', va='top', color='#c0392b', fontsize=10, fontweight='bold')

        # --- เหล็กบน (วาง Label เหนือคาน) ---
        y_top = h_beam - cover_mm
        ax.plot([x_s, x_s + L_mm*0.3], [y_top, y_top], color='#2980b9', lw=2.5)
        ax.plot([x_e - L_mm*0.3, x_e], [y_top, y_top], color='#2980b9', lw=2.5)
        ax.text(x_s + 50, h_beam + 40, f"{int(res['neg']['n'])}-DB{int(res['db'])}", 
                ha='left', color='#2980b9', fontsize=10, fontweight='bold')

        # --- เหล็กปลอก (ระบุรายละเอียดที่ด้านล่างสุด แยกจากเหล็กหลัก) ---
        # วาดเส้นสัญลักษณ์โซนเหล็กปลอก
        stirrup_x = np.linspace(x_s + 200, x_e - 200, 5)
        for sx in stirrup_x:
            ax.plot([sx, sx], [cover_mm, h_beam-cover_mm], color='#27ae60', lw=0.7, alpha=0.3)
        
        # ย้ายป้ายเหล็กปลอกมาไว้ด้านล่าง (ใต้เลขเหล็กหลัก) ไม่ให้ทับกัน
        ax.annotate(f"RB6 @{int(res['shear']['s'])} mm", xy=(mid_x, h_beam/2), xytext=(mid_x, -180),
                    arrowprops=dict(arrowstyle='->', color='#27ae60', lw=0.5),
                    ha='center', color='#27ae60', fontsize=9, fontweight='bold',
                    bbox=dict(facecolor='white', edgecolor='#27ae60', boxstyle='round,pad=0.2', alpha=0.9))

    # 4. Span Dimensions
    for i in range(len(spans)):
        ax.annotate('', xy=(offsets[i], h_beam + 150), xytext=(offsets[i+1], h_beam + 150),
                    arrowprops=dict(arrowstyle='<->', color='#7f8c8d'))
        ax.text((offsets[i]+offsets[i+1])/2, h_beam + 180, f"SPAN {i+1}: {spans[i]}m", ha='center', fontsize=9)

    ax.set_xlim(-400, total_L + 400)
    ax.set_ylim(-350, h_beam + 400)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.title("PROFESSIONAL LONGITUDINAL REINFORCEMENT DETAIL", fontsize=13, fontweight='bold', pad=25)
    
    return fig
