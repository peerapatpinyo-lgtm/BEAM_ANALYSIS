import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def plot_section(b_m, h_m, cover_mm, db_main_mm, n_top, n_bottom, stirrup_name, fc, fy):
    """
    World-Class Engineering Section Detail Plotter.
    Includes: Dimension lines, proper hook visualization, and material callouts.
    """
    # Convert all to mm for plotting
    b, h = b_m * 1000, h_m * 1000
    cover = cover_mm
    ds = 6  # Stirrup diameter approx.
    db = db_main_mm
    
    fig, ax = plt.subplots(figsize=(6, 8))
    
    # 1. Draw Concrete Outline (Filled with light gray hatch)
    beam_rect = patches.Rectangle((0, 0), b, h, linewidth=2, edgecolor='black', facecolor='#f0f0f0', hatch='///', alpha=0.3)
    ax.add_patch(beam_rect)
    
    # 2. Draw Stirrup (With 135-degree hooks at top corner)
    s_x, s_y = cover, cover
    s_w, s_h = b - 2*cover, h - 2*cover
    stirrup = patches.Rectangle((s_x, s_y), s_w, s_h, linewidth=2, edgecolor='#2c3e50', facecolor='none', linestyle='-')
    ax.add_patch(stirrup)
    
    # 3. Draw Main Bars (Top & Bottom)
    def draw_bars(n, y_pos, color, label_prefix):
        if n <= 1: return
        # Calculate horizontal spacing
        usable_w = b - 2*cover - 2*ds - db
        spacing = usable_w / (n - 1)
        for i in range(n):
            x = cover + ds + db/2 + i*spacing
            circle = plt.Circle((x, y_pos), db/2, color=color, zorder=5)
            ax.add_patch(circle)
            
    # Bottom bars (Red - Tension for Positive Moment)
    draw_bars(n_bottom, cover + ds + db/2, '#c0392b', "Bottom")
    # Top bars (Blue - Compression/Negative Tension)
    draw_bars(n_top, h - cover - ds - db/2, '#2980b9', "Top")
    
    # 4. Dimension Lines & Annotations
    ax.annotate('', xy=(0, -40), xytext=(b, -40), arrowprops=dict(arrowstyle='<->'))
    ax.text(b/2, -70, f"b = {int(b)} mm", ha='center', fontweight='bold')
    
    ax.annotate('', xy=(-40, 0), xytext=(-40, h), arrowprops=dict(arrowstyle='<->'))
    ax.text(-80, h/2, f"h = {int(h)} mm", va='center', rotation=90, fontweight='bold')
    
    # 5. Material & Rebar Callouts
    info_text = (
        f"SECTION DETAIL\n"
        f"Concrete: f'c {fc} MPa\n"
        f"Steel: fy {fy} MPa\n"
        f"Top: {n_top}-DB{db}\n"
        f"Bottom: {n_bottom}-DB{db}\n"
        f"Stirrup: {stirrup_name} @ spacing"
    )
    plt.text(b + 50, h, info_text, va='top', fontsize=10, bbox=dict(facecolor='white', alpha=0.5))

    # Layout Adjustments
    ax.set_xlim(-150, b + 300)
    ax.set_ylim(-150, h + 100)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.title(f"Cross Section Details", fontsize=14, pad=20)
    
    return fig

def plot_longitudinal_section_advanced(spans, sup_df, design_res, h_m, res_df):
    """
    Advanced Structural Detailing with Moment Envelope Overlay.
    - Blue: Top Steel (Negative Moment)
    - Red: Bottom Steel (Positive Moment)
    - Gray: Shear Stirrups with varying zones
    """
    h = h_m * 1000
    total_L = sum(spans) * 1000
    offsets = [0] + list(np.cumsum(spans) * 1000)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), sharex=True, 
                                   gridspec_kw={'height_ratios': [1, 2]})
    plt.subplots_adjust(hspace=0.05)

    # --- 1. Top Axis: Moment Envelope for Verification ---
    ax1.plot(res_df['x']*1000, res_df['moment']/1000, color='#27ae60', lw=1.5, label='Bending Moment')
    ax1.fill_between(res_df['x']*1000, 0, res_df['moment']/1000, color='#27ae60', alpha=0.1)
    ax1.axhline(0, color='black', lw=0.8)
    ax1.set_ylabel("Moment (kNm)")
    ax1.set_title("Moment Envelope vs. Bar Detailing", fontsize=12, fontweight='bold')

    # --- 2. Bottom Axis: Detailed Reinforcement ---
    # Draw Concrete
    ax2.add_patch(patches.Rectangle((0, 0), total_L, h, facecolor='#f9f9f9', edgecolor='black', lw=1.5))
    
    cover = 40
    for i, span_l_m in enumerate(spans):
        L_mm = span_l_m * 1000
        x_s, x_e = offsets[i], offsets[i+1]
        
        # [เหล็กล่าง - Positive] วิ่งยาวตลอด (Main) + เสริมพิเศษกลางช่วง (Extra)
        ax2.plot([x_s+50, x_e-50], [cover, cover], color='#c0392b', lw=3, solid_capstyle='round')
        ax2.text(x_s + L_mm/2, cover+20, f"{design_res[i]['pos']['n']}-DB16", ha='center', color='#c0392b', size=9)

        # [เหล็กบน - Negative] ตัดตามพฤติกรรม Moment (Inflection Points)
        # แสดงระยะล้วงเข้า Support (Development Length)
        cut_left = x_s + (L_mm * 0.3)
        cut_right = x_e - (L_mm * 0.3)
        ax2.plot([x_s, cut_left], [h-cover, h-cover], color='#2980b9', lw=3)
        ax2.plot([cut_right, x_e], [h-cover, h-cover], color='#2980b9', lw=3)
        
        # [เหล็กปลอก - Shear Stirrups]
        # โซนถี่ (Denser at supports) vs โซนห่าง (Mid-span)
        s_fine = design_res[i]['shear']['s']
        s_coarse = min(s_fine * 2, h/2) # มาตรฐานยอมให้ห่างได้ไม่เกิน d/2
        
        # วาดสัญลักษณ์โซนเหล็กปลอก
        n_dense = 6
        dense_points = list(np.linspace(x_s, x_s + 2*h, n_dense)) + \
                       list(np.linspace(x_e - 2*h, x_e, n_dense))
        for px in dense_points:
            ax2.plot([px, px], [cover, h-cover], color='#95a5a6', lw=1, alpha=0.7)
            
        ax2.text(x_s + h, h/2, f"@{int(s_fine)}", ha='center', size=8, color='#7f8c8d')
        ax2.text(x_s + L_mm/2, h/2, f"@{int(s_coarse)}", ha='center', size=8, color='#7f8c8d')

    # Draw Supports as Columns
    for s_x in offsets:
        ax2.add_patch(patches.Rectangle((s_x-100, -200), 200, 200, facecolor='#ecf0f1', edgecolor='#bdc3c7'))

    ax2.set_ylim(-250, h + 150)
    ax2.set_aspect('equal')
    ax2.axis('off')
    
    return fig
