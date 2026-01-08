import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def plot_section(b_m, h_m, cover_mm, db_main_mm, n_top, n_bottom, stirrup_name, fc, fy):
    """
    Cross Section: Callouts positioned beside the bars (Outside section)
    """
    b, h = b_m * 1000, h_m * 1000
    cover = cover_mm
    ds, db = 6, db_main_mm
    
    fig, ax = plt.subplots(figsize=(6, 7), dpi=100)
    
    # Concrete Face
    ax.add_patch(patches.Rectangle((0, 0), b, h, linewidth=1.5, edgecolor='#1a1a1a', facecolor='#fdfdfd'))
    
    # Stirrup
    ax.add_patch(patches.Rectangle((cover, cover), b-2*cover, h-2*cover, linewidth=1.2, edgecolor='#2c3e50', facecolor='none'))
    
    def draw_bars(n, y_pos, color):
        if n < 1: return
        spacing = (b - 2*cover - 2*ds - db) / (max(1, n - 1))
        for i in range(int(n)):
            x = (b/2) if n == 1 else (cover + ds + db/2 + i*spacing)
            ax.add_patch(plt.Circle((x, y_pos), db/2, color=color, zorder=10))

    y_bot_bars = cover + ds + db/2
    y_top_bars = h - cover - ds - db/2
    draw_bars(n_bottom, y_bot_bars, '#8e1c19')
    draw_bars(n_top, y_top_bars, '#1a5276')
    
    # --- ย้ายที่บอกเหล็กมาไว้ข้างๆ (Side Callouts) ---
    # เหล็กบน (Top Steel) - บอกที่ระดับความสูงของเหล็ก
    ax.text(b + 30, y_top_bars, f"← {int(n_top)}-DB{int(db)} (Top)", va='center', color='#1a5276', fontweight='bold')
    
    # เหล็กปลอก (Stirrup) - บอกที่กึ่งกลาง
    ax.text(b + 30, h/2, f"← {stirrup_name}", va='center', color='#1d8348', fontweight='bold')

    # เหล็กล่าง (Bottom Steel) - บอกที่ระดับความสูงของเหล็ก
    ax.text(b + 30, y_bot_bars, f"← {int(n_bottom)}-DB{int(db)} (Bot)", va='center', color='#8e1c19', fontweight='bold')

    # Dimensions (ISO Style)
    ax.plot([0, b], [-40, -40], color='black', lw=0.8) # Width Dim
    ax.text(b/2, -70, f"{int(b)}", ha='center', size=9)
    ax.plot([-40, -40], [0, h], color='black', lw=0.8) # Height Dim
    ax.text(-80, h/2, f"{int(h)}", va='center', rotation=90, size=9)
    
    ax.set_xlim(-120, b + 250)
    ax.set_ylim(-150, h + 150)
    ax.set_aspect('equal')
    ax.axis('off')
    return fig

def plot_longitudinal_section_detailed(spans, sup_df, design_res, h_m, cover_mm):
    """
    Longitudinal: Complete Bar Info (Top & Bottom) with Support Detail
    """
    h_beam = h_m * 1000 
    total_L = sum(spans) * 1000
    offsets = [0] + list(np.cumsum(spans) * 1000)
    
    fig, ax = plt.subplots(figsize=(15, 5), dpi=100)
    
    # 1. Beam Outline
    ax.add_patch(patches.Rectangle((0, 0), total_L, h_beam, lw=1.5, edgecolor='#1a1a1a', facecolor='#ffffff'))
    
    # 2. Support Columns
    for _, sup in sup_df.iterrows():
        sx = sup['x'] * 1000
        ax.add_patch(patches.Rectangle((sx-70, -h_beam*0.8), 140, h_beam*0.8, facecolor='#f2f4f4', edgecolor='#7f8c8d'))
        ax.text(sx, -h_beam*1.1, f"{sup['type']}", ha='center', fontsize=8, color='#7f8c8d')

    # 3. Bar Detailing (Top, Bottom, and Stirrups)
    for i, span_l_m in enumerate(spans):
        L_mm = span_l_m * 1000
        x_s, x_e = offsets[i], offsets[i+1]
        mid_x = (x_s + x_e) / 2
        res = design_res[i]
        
        # --- Bottom Steel ---
        y_bot = cover_mm
        ax.plot([x_s+30, x_e-30], [y_bot, y_bot], color='#8e1c19', lw=2)
        ax.text(mid_x, y_bot - 40, f"{int(res['pos']['n'])}-DB{int(res['db'])}", 
                ha='center', va='top', color='#8e1c19', fontsize=9, fontweight='bold')

        # --- Top Steel (Negative) - แก้ไขให้แสดงผลชัดเจน ---
        y_top = h_beam - cover_mm
        cut_len = L_mm * 0.25 # ระยะหยุดเหล็ก
        
        # วาดเส้นเหล็กบน
        ax.plot([x_s, x_s + cut_len], [y_top, y_top], color='#1a5276', lw=2)
        ax.plot([x_e - cut_len, x_e], [y_top, y_top], color='#1a5276', lw=2)
        
        # บอกรายละเอียดเหล็กบน (วางไว้เหนือคาน)
        ax.text(x_s + 50, h_beam + 30, f"{int(res['neg']['n'])}-DB{int(res['db'])} (Top)", 
                ha='left', va='bottom', color='#1a5276', fontsize=8, fontweight='bold')

        # --- Stirrups ---
        ax.text(mid_x, h_beam + 80, f"RB6@{int(res['shear']['s'])}", 
                ha='center', color='#1d8348', fontsize=8, fontweight='bold')
        for sx in np.linspace(x_s + 150, x_e - 150, 6):
            ax.plot([sx, sx], [cover_mm, h_beam-cover_mm], color='#1d8348', lw=0.6, alpha=0.2)

    # 4. Dimension
    ax.annotate('', xy=(0, -60), xytext=(total_L, -60), arrowprops=dict(arrowstyle='<->', color='#7f8c8d'))
    ax.text(total_L/2, -100, f"Total Length: {total_L/1000} m", ha='center', fontsize=9)

    ax.set_xlim(-300, total_L + 300)
    ax.set_ylim(-h_beam*1.5, h_beam*2)
    ax.set_aspect('equal')
    ax.axis('off')
    return fig
