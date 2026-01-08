import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def plot_longitudinal_section_detailed(spans, sup_df, design_res, h_m, cover_mm):
    """
    World-Class Longitudinal Detailing.
    - Dynamic Support Symbols
    - Precise Rebar Callouts
    - Professional Dimensioning
    """
    h_beam = h_m * 1000  # Convert to mm
    total_L = sum(spans) * 1000
    offsets = [0] + list(np.cumsum(spans) * 1000)
    
    fig, ax = plt.subplots(figsize=(18, 6))
    
    # 1. วาดตัวคาน (Concrete Outline)
    ax.add_patch(patches.Rectangle((0, 0), total_L, h_beam, 
                                   linewidth=2, edgecolor='#2c3e50', facecolor='#fdfefe', zorder=1))

    # 2. วาด Support ตามตำแหน่งและประเภทจริง
    for _, sup in sup_df.iterrows():
        sx = sup['x'] * 1000
        stype = sup['type']
        
        if stype == 'Fixed':
            # วาดสัญลักษณ์ผนัง Fixed
            ax.add_patch(patches.Rectangle((sx-20, -50), 40, h_beam+100, facecolor='#bdc3c7', hatch='///'))
        else:
            # วาดสัญลักษณ์ Roller/Pin (Triangle)
            poly = plt.Polygon([[sx-100, -100], [sx+100, -100], [sx, 0]], 
                               closed=True, fill=True, facecolor='#ecf0f1', edgecolor='black', lw=1.5)
            ax.add_patch(poly)
            if stype == 'Roller':
                ax.plot([sx-100, sx+100], [-120, -120], 'k-', lw=2)

    # 3. วาดเหล็กเสริม (Reinforcement)
    for i, span_l_m in enumerate(spans):
        L_mm = span_l_m * 1000
        x_s, x_e = offsets[i], offsets[i+1]
        res = design_res[i]
        
        # --- เหล็กล่าง (Positive Moment) ---
        y_bot = cover_mm
        ax.plot([x_s + 50, x_e - 50], [y_bot, y_bot], color='#c0392b', lw=3, solid_capstyle='round', zorder=3)
        ax.annotate(f"{res['pos']['n']}-DB{res['db']}", xy=(x_s + L_mm/2, y_bot), xytext=(0, -25),
                    textcoords='offset points', ha='center', color='#c0392b', fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='#c0392b'))

        # --- เหล็กบน (Negative Moment / Support Bars) ---
        y_top = h_beam - cover_mm
        cut_len = L_mm * 0.30 # ระยะตัดเหล็ก 0.3L
        
        # วาดเหล็กเสริมช่วงหัวเสา
        ax.plot([x_s, x_s + cut_len], [y_top, y_top], color='#2980b9', lw=3, zorder=3)
        ax.plot([x_e - cut_len, x_e], [y_top, y_top], color='#2980b9', lw=3, zorder=3)
        
        # ป้ายบอกเหล็กบน (เฉพาะช่วงที่มีเหล็ก)
        ax.text(x_s + 100, y_top + 20, f"{res['neg']['n']}-DB{res['db']}", 
                color='#2980b9', fontsize=9, fontweight='bold')

        # --- เหล็กปลอก (Stirrups) ---
        s_val = res['shear']['s']
        # วาดเส้นจำลองเหล็กปลอก (สุ่มแสดงผลเพื่อความสวยงาม ไม่ให้แน่นเกินไป)
        n_vis_stirrups = int(L_mm / 250) # แสดงทุกๆ 250mm ในแบบ
        stirrup_x = np.linspace(x_s + 100, x_e - 100, n_vis_stirrups)
        for sx in stirrup_x:
            ax.plot([sx, sx], [cover_mm, h_beam - cover_mm], color='#27ae60', lw=0.8, alpha=0.4, zorder=2)
        
        # ป้ายบอกระยะเหล็กปลอก
        ax.text(x_s + L_mm/2, h_beam/2, f"RB6 @{int(s_val)} mm", 
                ha='center', va='center', rotation=90, color='#27ae60', fontsize=8, 
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))

    # 4. Dimension Lines (บอกระยะ Span)
    for i in range(len(spans)):
        x_m = (offsets[i] + offsets[i+1]) / 2
        ax.annotate('', xy=(offsets[i], h_beam + 150), xytext=(offsets[i+1], h_beam + 150),
                    arrowprops=dict(arrowstyle='<->', color='gray'))
        ax.text(x_m, h_beam + 180, f"L = {spans[i]} m", ha='center', fontweight='bold')

    # ปรับแต่งขอบรูป
    ax.set_xlim(-500, total_L + 500)
    ax.set_ylim(-350, h_beam + 400)
    ax.set_aspect('equal')
    ax.axis('off')
    
    plt.title("DETAILED LONGITUDINAL SECTION & REINFORCEMENT", fontsize=14, fontweight='bold', pad=30)
    
    return fig
