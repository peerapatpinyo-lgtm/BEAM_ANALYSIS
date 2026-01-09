import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def plot_longitudinal_section_detailed(spans, sup_df, design_res, h_m, cover_mm):
    """
    วาดรูปตัดตามยาวโดยใช้ Vector Rendering 
    คมชัดระดับสูงสุด ซูมไม่แตก และแก้ปัญหาคานหนา
    """
    spans_mm = [s * 1000 for s in spans]
    total_L = sum(spans_mm)
    v_h = 400  # บังคับความสูงคานให้เพรียวบางที่สุด (Professional Look)
    
    # 1. ตั้งค่า Canvas ให้กว้างพิเศษ และใช้ความละเอียดสูง
    fig_w = max(18, total_L / 350)
    fig, ax = plt.subplots(figsize=(fig_w, 4))
    
    # --- 🛠️ เคล็ดลับความคม: ปิด Anti-aliasing สำหรับเส้นตรงหลัก ---
    # วาดขอบคอนกรีต
    beam = patches.Rectangle((0, 0), total_L, v_h, lw=2, ec='black', fc='white', zorder=2, antialiased=False)
    ax.add_patch(beam)
    
    # 2. วาด Grid Lines และหัวเสา (A, B, C)
    curr_x = 0
    for i, s_mm in enumerate(spans_mm + [0]):
        # เส้น Grid ประ
        ax.plot([curr_x, curr_x], [-600, v_h + 400], color='#bdc3c7', ls=(0, (5, 5)), lw=1, zorder=1)
        # หัว Grid
        ax.annotate(chr(65+i), xy=(curr_x, v_h + 500), ha='center', va='center',
                    bbox=dict(boxstyle='circle', fc='white', ec='black', lw=1.5), fontsize=14, fontweight='bold')
        
        # Dimension Line (เส้นบอกระยะ Span)
        if i < len(spans_mm):
            ax.annotate('', xy=(curr_x, v_h + 250), xytext=(curr_x + s_mm, v_h + 250),
                        arrowprops=dict(arrowstyle='<->', color='#2980b9', lw=1.2))
            ax.text(curr_x + s_mm/2, v_h + 300, f"{s_mm/1000:.2f} m", 
                    ha='center', color='#2980b9', fontsize=11, fontweight='bold')
            curr_x += s_mm

    # 3. วาด Support ตามประเภท (Fixed, Pin, Roller)
    if not sup_df.empty:
        for _, row in sup_df.iterrows():
            sx = row['x'] * 1000
            stype = str(row.get('type', 'PIN')).upper()
            
            if stype == 'FIXED':
                ax.add_patch(patches.Rectangle((sx-100, -350), 200, 350, fc='#dfe6e9', ec='black', lw=1.5, hatch='////'))
            elif stype == 'ROLLER':
                pts = np.array([[sx, 0], [sx-90, -180], [sx+90, -180]])
                ax.add_patch(patches.Polygon(pts, fc='white', ec='black', lw=1.5))
                ax.add_patch(patches.Circle((sx, -210), 25, fc='black'))
            else: # PIN
                pts = np.array([[sx, 0], [sx-90, -180], [sx+90, -180]])
                ax.add_patch(patches.Polygon(pts, fc='#2c3e50', ec='black', lw=1.5))
            
            ax.text(sx, -500, f"S{row['id']}: {stype}", ha='center', fontweight='bold', fontsize=10, color='#34495e')

    # 4. วาดเหล็กเสริม (Reinforcement) - เน้นเส้นคมและสีสด
    y_t, y_b = v_h * 0.82, v_h * 0.18
    x_curr = 0
    for i, span_L in enumerate(spans_mm):
        res = design_res[i]
        mid = x_curr + span_L/2
        
        # เหล็กเมน (ปิด antialiased เพื่อให้เส้นคมกริบแบบ CAD)
        ax.plot([x_curr, x_curr + span_L], [y_t, y_t], color='#e74c3c', lw=3.5, zorder=10, antialiased=False)
        ax.plot([x_curr + 40, x_curr + span_L - 40], [y_b, y_b], color='#27ae60', lw=3.5, zorder=10, antialiased=False)
        
        # ป้ายบอกเหล็ก (ใช้พื้นหลังขาวเพื่อตัดเส้น)
        label_style = dict(ha='center', fontweight='bold', fontsize=11, bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=0.5))
        ax.text(mid, v_h + 80, f"{int(res['neg']['n'])}-DB{int(res['top_db'])} (TOP)", color='#c0392b', **label_style)
        ax.text(mid, y_b - 50, f"{int(res['pos']['n'])}-DB{int(res['bot_db'])} (BOT)", color='#1e8449', va='top', **label_style)

        x_curr += span_L

    ax.set_aspect('auto')
    ax.axis('off')
    ax.set_xlim(-1000, total_L + 1000)
    ax.set_ylim(-800, v_h + 800)
    
    return fig
