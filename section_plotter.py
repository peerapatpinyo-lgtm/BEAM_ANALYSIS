import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def plot_section(b_m, h_m, cover_mm, db_main_mm, n_top, n_bottom, stirrup_name, fc, fy):
    """
    หน้าตัดคาน (Cross Section) - รูปแบบมาตรฐานวิศวกรรม
    """
    b, h = b_m * 1000, h_m * 1000
    cover = cover_mm
    ds = 6  # Stirrup diameter
    db = db_main_mm
    
    fig, ax = plt.subplots(figsize=(5, 7))
    
    # วาดคอนกรีต (Concrete Outline)
    ax.add_patch(patches.Rectangle((0, 0), b, h, linewidth=2, edgecolor='black', facecolor='#f8f9fa', hatch='///', alpha=0.2))
    
    # วาดเหล็กปลอก (Stirrup)
    ax.add_patch(patches.Rectangle((cover, cover), b-2*cover, h-2*cover, linewidth=1.5, edgecolor='#2c3e50', facecolor='none'))
    
    # ฟังก์ชันวาดเหล็กเส้น
    def draw_bars(n, y_pos, color):
        if n < 1: return
        if n == 1:
            ax.add_patch(plt.Circle((b/2, y_pos), db/2, color=color, zorder=5))
        else:
            spacing = (b - 2*cover - 2*ds - db) / (n - 1)
            for i in range(n):
                x = cover + ds + db/2 + i*spacing
                ax.add_patch(plt.Circle((x, y_pos), db/2, color=color, zorder=5))

    draw_bars(n_bottom, cover + ds + db/2, '#c0392b') # เหล็กล่าง
    draw_bars(n_top, h - cover - ds - db/2, '#2980b9') # เหล็กบน
    
    # ข้อมูลประกอบ (Callouts)
    info_text = (
        f"BEAM SECTION DETAIL\n"
        f"Size: {int(b)}x{int(h)} mm\n"
        f"fc': {fc} MPa | fy: {fy} MPa\n"
        f"Top: {int(n_top)}-DB{int(db)}\n"
        f"Bottom: {int(n_bottom)}-DB{int(db)}\n"
        f"Stirrup: {stirrup_name}"
    )
    ax.text(b/2, h + 50, info_text, ha='center', va='bottom', family='monospace', 
            fontsize=9, bbox=dict(facecolor='white', alpha=0.8, edgecolor='#bdc3c7'))
    
    # เส้นบอกขนาด (Dimensions)
    ax.annotate('', xy=(0, -30), xytext=(b, -30), arrowprops=dict(arrowstyle='<->'))
    ax.text(b/2, -80, f"b={int(b)}", ha='center', size=9)
    ax.annotate('', xy=(-30, 0), xytext=(-30, h), arrowprops=dict(arrowstyle='<->'))
    ax.text(-100, h/2, f"h={int(h)}", va='center', rotation=90, size=9)
    
    ax.set_xlim(-150, b + 150)
    ax.set_ylim(-150, h + 250)
    ax.set_aspect('equal')
    ax.axis('off')
    return fig

def plot_longitudinal_section_detailed(spans, sup_df, design_res, h_m, cover_mm):
    """
    รูปตัดตามยาว (Longitudinal Section) - แสดงเหล็กเสริมและ Support ตามจริง
    """
    h_beam = h_m * 1000 
    total_L = sum(spans) * 1000
    offsets = [0] + list(np.cumsum(spans) * 1000)
    
    fig, ax = plt.subplots(figsize=(16, 6))
    
    # 1. ตัวคาน
    ax.add_patch(patches.Rectangle((0, 0), total_L, h_beam, linewidth=2, edgecolor='#2c3e50', facecolor='#ffffff', zorder=1))

    # 2. วาด Support ตามประเภท
    for _, sup in sup_df.iterrows():
        sx = sup['x'] * 1000
        stype = sup['type']
        if stype == 'Fixed':
            ax.add_patch(patches.Rectangle((sx-25, -50), 50, h_beam+100, facecolor='#bdc3c7', hatch='///'))
        else:
            # Pin / Roller
            poly = plt.Polygon([[sx-120, -120], [sx+120, -120], [sx, 0]], closed=True, facecolor='#ecf0f1', edgecolor='black', lw=1.5)
            ax.add_patch(poly)
            if stype == 'Roller': ax.plot([sx-120, sx+120], [-140, -140], 'k-', lw=2)

    # 3. วาดเหล็กเสริมและป้ายบอกรายละเอียด
    for i, span_l_m in enumerate(spans):
        L_mm = span_l_m * 1000
        x_s, x_e = offsets[i], offsets[i+1]
        res = design_res[i]
        
        # เหล็กล่าง (Main Bottom Steel)
        y_bot = cover_mm + 5
        ax.plot([x_s + 50, x_e - 50], [y_bot, y_bot], color='#c0392b', lw=3, solid_capstyle='round', zorder=3)
        ax.text(x_s + L_mm/2, y_bot + 20, f"{int(res['pos']['n'])}-DB{int(res['db'])}", 
                ha='center', color='#c0392b', fontsize=10, fontweight='bold')

        # เหล็กบน (Top Curtailment Steel)
        y_top = h_beam - cover_mm - 5
        cut_len = L_mm * 0.3 # ระยะตัดเหล็ก 0.3L
        ax.plot([x_s, x_s + cut_len], [y_top, y_top], color='#2980b9', lw=3, zorder=3)
        ax.plot([x_e - cut_len, x_e], [y_top, y_top], color='#2980b9', lw=3, zorder=3)
        ax.text(x_s + 50, y_top - 40, f"{int(res['neg']['n'])}-DB{int(res['db'])}", color='#2980b9', fontsize=9, fontweight='bold')

        # เหล็กปลอก (Stirrups) - วาดเส้นจำลองโซน
        s_val = res['shear']['s']
        num_vis = int(L_mm / 300) 
        for sx in np.linspace(x_s + 150, x_e - 150, num_vis):
            ax.plot([sx, sx], [cover_mm, h_beam-cover_mm], color='#27ae60', lw=1, alpha=0.4)
        ax.text(x_s + L_mm/2, h_beam/2, f"RB6 @{int(s_val)}", ha='center', va='center', rotation=90, 
                color='#27ae60', fontsize=9, bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))

    # 4. บอกระยะ Span
    for i in range(len(spans)):
        mid = (offsets[i] + offsets[i+1]) / 2
        ax.annotate('', xy=(offsets[i], -250), xytext=(offsets[i+1], -250), arrowprops=dict(arrowstyle='<->', color='gray'))
        ax.text(mid, -300, f"Span {i+1}: {spans[i]}m", ha='center', fontweight='bold')

    ax.set_xlim(-500, total_L + 500)
    ax.set_ylim(-400, h_beam + 300)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.title("DETAILED LONGITUDINAL REINFORCEMENT PROFILE", fontsize=14, fontweight='bold', pad=20)
    
    return fig
