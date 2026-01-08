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

def plot_longitudinal_section_detailed(spans, supports, design_res, h_m, cover_mm):
    """
    Professional Longitudinal Reinforcement Profile.
    Shows bar continuity and support positions.
    """
    h = h_m * 1000
    total_L = sum(spans) * 1000
    offsets = [0] + list(np.cumsum(spans) * 1000)
    
    fig, ax = plt.subplots(figsize=(15, 4))
    
    # Draw Beam Outline
    ax.add_patch(patches.Rectangle((0, 0), total_L, h, linewidth=2, edgecolor='black', facecolor='#f8f9fa'))
    
    # Draw Supports (World-class style: Symbol below beam)
    for s_x in offsets:
        ax.plot([s_x-100, s_x+100], [-50, -50], 'k-', lw=3)
        ax.plot([s_x, s_x], [0, -50], 'k--', lw=1)

    # Draw Reinforcement Continuity
    # Top bars (Negative Moment Zones - typically 1/3 of span)
    for i in range(len(design_res)):
        # Main Bottom Bars (Continuous)
        ax.plot([offsets[i]+50, offsets[i+1]-50], [cover_mm+10, cover_mm+10], color='#c0392b', lw=design_res[i]['pos']['n'], label='Bottom Steel')
        # Main Top Bars (Near supports)
        ax.plot([offsets[i], offsets[i]+(spans[i]*333)], [h-cover_mm-10, h-cover_mm-10], color='#2980b9', lw=design_res[i]['neg']['n'])

    # Labels
    for i, span_l in enumerate(spans):
        ax.text(offsets[i] + (span_l*500), h + 50, f"Span {i+1}\nL = {span_l}m", ha='center', fontsize=10)

    ax.set_xlim(-500, total_L + 500)
    ax.set_ylim(-200, h + 300)
    ax.set_aspect('equal')
    ax.axis('off')
    
    return fig
