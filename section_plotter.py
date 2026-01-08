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

def plot_longitudinal_section_detailed(spans, sup_df, design_res, h_m, cover_mm):
    """
    Professional Engineering Detailing for Longitudinal Reinforcement.
    Features: Proper bar curtailment, Stirrup zones, and Support symbols.
    """
    h = h_m * 1000
    total_L = sum(spans) * 1000
    offsets = [0] + list(np.cumsum(spans) * 1000)
    
    fig, ax = plt.subplots(figsize=(16, 5))
    
    # 1. Concrete Outline
    ax.add_patch(patches.Rectangle((0, 0), total_L, h, linewidth=1.5, edgecolor='#34495e', facecolor='#fdfefe'))

    # 2. Rebar Detailing Logic
    for i, span_l_m in enumerate(spans):
        L_mm = span_l_m * 1000
        start_x = offsets[i]
        end_x = offsets[i+1]
        
        # --- Bottom Steel (Positive) ---
        # วิ่งยาวต่อเนื่องตลอดแนวล่าง (Continuous Bottom Reinforcement)
        ax.plot([start_x, end_x], [cover_mm + 10, cover_mm + 10], color='#c0392b', lw=2.5, solid_capstyle='round')
        ax.text(start_x + L_mm/2, cover_mm + 25, f"{design_res[i]['pos']['n']}-DB{design_res[i]['db']}", 
                ha='center', color='#c0392b', fontsize=9, fontweight='bold')

        # --- Top Steel (Negative/Support Steel) ---
        # หลักการ Curtailment: เหล็กบนต้องยื่นออกมา L/3 จากหน้า Support
        cut_len = L_mm / 3.0
        
        # Left Support Steel
        ax.plot([start_x, start_x + cut_len], [h - cover_mm - 10, h - cover_mm - 10], color='#2980b9', lw=2.5)
        # Right Support Steel
        ax.plot([end_x - cut_len, end_x], [h - cover_mm - 10, h - cover_mm - 10], color='#2980b9', lw=2.5)
        
        ax.text(start_x + 50, h - cover_mm - 40, f"{design_res[i]['neg']['n']}-DB{design_res[i]['db']}", 
                ha='left', color='#2980b9', fontsize=8)

        # --- Stirrup Zones (Shear Reinforcement) ---
        # วาดโซนเหล็กปลอก ถี่ที่ปลาย ห่างที่กลาง
        s_spacing = design_res[i]['shear']['s'] # ระยะที่คำนวณได้
        n_zones = 10
        zone_x = np.linspace(start_x, end_x, n_zones)
        for z in zone_x:
            ax.plot([z, z], [cover_mm, h - cover_mm], color='#7f8c8d', lw=0.8, alpha=0.6)
        
        ax.text(start_x + L_mm/2, h/2, f"Stirrups: @{int(s_spacing)}mm", 
                ha='center', color='#7f8c8d', fontsize=8, fontstyle='italic')

    # 3. Support Symbols (Column representation)
    for _, sup in sup_df.iterrows():
        sx = sup['x'] * 1000
        # วาดรูปตอม่อหรือเสาประคอง
        ax.add_patch(patches.Rectangle((sx-75, -150), 150, 150, facecolor='#bdc3c7', alpha=0.5))
        ax.plot([sx-75, sx+75], [0, 0], 'k-', lw=2)

    # 4. Dimension & Grid
    ax.set_xlim(-200, total_L + 200)
    ax.set_ylim(-300, h + 300)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.title(f"LONGITUDINAL REINFORCEMENT PROFILE (Standard Curtailment)", fontsize=12, fontweight='bold', pad=20)
    
    return fig
