import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def plot_section(b_m, h_m, cover_mm, db_main_mm, n_top, n_bottom, stirrup_name, fc, fy):
    """
    World-Class Cross Section Detail.
    """
    b, h = b_m * 1000, h_m * 1000
    cover = cover_mm
    ds = 6  
    db = db_main_mm
    
    fig, ax = plt.subplots(figsize=(5, 7))
    
    # Concrete Hatch
    ax.add_patch(patches.Rectangle((0, 0), b, h, linewidth=2, edgecolor='black', facecolor='#f8f9fa', hatch='...', alpha=0.5))
    
    # Stirrup
    ax.add_patch(patches.Rectangle((cover, cover), b-2*cover, h-2*cover, linewidth=2, edgecolor='#34495e', facecolor='none'))
    
    # Bars logic
    def draw_bars(n, y_pos, color):
        if n < 2: return
        spacing = (b - 2*cover - 2*ds - db) / (n - 1)
        for i in range(n):
            x = cover + ds + db/2 + i*spacing
            ax.add_patch(plt.Circle((x, y_pos), db/2, color=color, zorder=5))

    draw_bars(n_bottom, cover + ds + db/2, '#c0392b') # Bottom
    draw_bars(n_top, h - cover - ds - db/2, '#2980b9') # Top
    
    # Dimensions
    ax.text(b/2, -50, f"b={int(b)}", ha='center', fontweight='bold')
    ax.text(-50, h/2, f"h={int(h)}", va='center', rotation=90, fontweight='bold')
    
    ax.set_xlim(-100, b + 100)
    ax.set_ylim(-100, h + 100)
    ax.set_aspect('equal')
    ax.axis('off')
    return fig

def plot_longitudinal_section_detailed(spans, sup_df, design_res, h_m, cover_mm):
    """
    FIXED & UPGRADED: Engineering Grade Longitudinal Detail.
    This replaces the old function to fix the AttributeError.
    """
    h = h_m * 1000
    total_L = sum(spans) * 1000
    offsets = [0] + list(np.cumsum(spans) * 1000)
    
    fig, ax = plt.subplots(figsize=(15, 5))
    
    # 1. Concrete Outline (Double Line for professionalism)
    ax.add_patch(patches.Rectangle((0, 0), total_L, h, facecolor='#ffffff', edgecolor='#2c3e50', lw=2))
    
    # 2. Rebar Placement Logic
    for i, span_l_m in enumerate(spans):
        L_mm = span_l_m * 1000
        x_s, x_e = offsets[i], offsets[i+1]
        
        # --- Bottom Steel (Continuous Tension Bars) ---
        # วาดเหล็กเส้นล่างสีแดงเข้ม
        ax.plot([x_s + 40, x_e - 40], [cover_mm + 10, cover_mm + 10], color='#c0392b', lw=3, solid_capstyle='round')
        ax.text(x_s + L_mm/2, cover_mm + 35, f"{design_res[i]['pos']['n']}-DB{design_res[i]['db']}", 
                ha='center', color='#c0392b', fontsize=9, fontweight='bold')

        # --- Top Steel (Negative Moment Curtailment @ L/3) ---
        # วาดเหล็กบนสีน้ำเงิน (เสริมพิเศษที่หัวเสา)
        cut_len = L_mm / 3.0
        ax.plot([x_s, x_s + cut_len], [h - cover_mm - 10, h - cover_mm - 10], color='#2980b9', lw=3)
        ax.plot([x_e - cut_len, x_e], [h - cover_mm - 10, h - cover_mm - 10], color='#2980b9', lw=3)
        
        # Label for Top Bars
        ax.text(x_s + 50, h - cover_mm - 40, f"{design_res[i]['neg']['n']}-DB{design_res[i]['db']}", 
                color='#2980b9', fontsize=8, fontweight='bold')

        # --- Stirrups Zones (Visualizing Shear Reinforcement) ---
        s_val = design_res[i]['shear']['s']
        # วาดโซนเหล็กปลอกถี่ (Support zones)
        n_stirrups = 12
        stirrup_locs = np.linspace(x_s, x_e, n_stirrups)
        for sx in stirrup_locs:
            ax.plot([sx, sx], [cover_mm, h - cover_mm], color='#95a5a6', lw=0.8, alpha=0.5)
        
        ax.text(x_s + L_mm/2, h/2, f"Stirrups @{int(s_val)}", ha='center', color='#7f8c8d', fontsize=8)

    # 3. Enhanced Support Symbols
    for _, sup in sup_df.iterrows():
        sx = sup['x'] * 1000
        ax.add_patch(patches.Rectangle((sx-80, -180), 160, 180, facecolor='#bdc3c7', alpha=0.3))
        ax.plot([sx-80, sx+80], [0, 0], color='black', lw=3)

    ax.set_xlim(-300, total_L + 300)
    ax.set_ylim(-300, h + 300)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.title("REINFORCEMENT PROFILE: LONGITUDINAL SECTION", fontsize=14, fontweight='bold', pad=20)
    
    return fig
