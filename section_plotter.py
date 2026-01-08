import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# --- 📏 Config ---
DPI_VALUE = 300     
GLOBAL_FONT = 8

def plot_section(b_m, h_m, cover_mm, db_main_mm, n_top, n_bottom, stirrup_name, fc, fy, title="SECTION A-A"):
    """ วาด Cross Section พร้อมรายละเอียดวัสดุและการเสริมเหล็ก """
    SECTION_SCALE = 500  
    b, h = b_m * 1000, h_m * 1000
    cover, ds, db = cover_mm, 6, db_main_mm
    
    view_left, view_right = -400, b + 800
    view_bottom, view_top = -500, h + 600
    
    width_inches = (view_right - view_left) / SECTION_SCALE 
    height_inches = (view_top - view_bottom) / SECTION_SCALE
    
    fig, ax = plt.subplots(figsize=(width_inches, height_inches), dpi=DPI_VALUE)
    
    # 1. Concrete Outline
    ax.add_patch(patches.Rectangle((0, 0), b, h, linewidth=2, edgecolor='#2c3e50', facecolor='#fdfefe'))
    
    # 2. Stirrup Line
    ax.add_patch(patches.Rectangle((cover, cover), b-2*cover, h-2*cover, linewidth=1, edgecolor='#7f8c8d', ls='--'))
    
    # 3. Main Bars
    def draw_bars(n, y_pos, color):
        if n < 1: return
        # Calculate spacing: if n=1, center it.
        if n == 1:
            x_positions = [b/2]
        else:
            start_x = cover + ds + db/2
            end_x = b - cover - ds - db/2
            spacing = (end_x - start_x) / (n - 1)
            x_positions = [start_x + i*spacing for i in range(int(n))]
            
        for x in x_positions:
            circle = plt.Circle((x, y_pos), db/2, color=color, zorder=10, ec='black', lw=0.5)
            ax.add_patch(circle)

    y_bot = cover + ds + db/2
    y_top = h - cover - ds - db/2
    
    draw_bars(n_bottom, y_bot, '#c0392b') # Bottom Steel (Red)
    draw_bars(n_top, y_top, '#2980b9')    # Top Steel (Blue)
    
    # 4. Annotations (Leaders & Text)
    # Top Steel Label
    ax.plot([b/2, b+100], [y_top, y_top+100], color='#2980b9', lw=1)
    ax.text(b+110, y_top+100, f"{int(n_top)}-DB{int(db)} (Top)", va='center', color='#2980b9', fontsize=GLOBAL_FONT, fontweight='bold')
    
    # Bottom Steel Label
    ax.plot([b/2, b+100], [y_bot, y_bot-100], color='#c0392b', lw=1)
    ax.text(b+110, y_bot-100, f"{int(n_bottom)}-DB{int(db)} (Bot)", va='center', color='#c0392b', fontsize=GLOBAL_FONT, fontweight='bold')
    
    # Stirrup Label
    ax.plot([b-cover, b+150], [h/2, h/2], color='#27ae60', lw=1, ls=':')
    ax.text(b+160, h/2, f"Stirrup: {stirrup_name}", va='center', color='#27ae60', fontsize=GLOBAL_FONT)
    
    # Material Properties (Bottom Right)
    mat_text = f"fc' = {fc} MPa\nfy = {fy} MPa"
    ax.text(view_right-50, view_bottom+50, mat_text, ha='right', va='bottom', fontsize=GLOBAL_FONT-2, color='gray', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    # Dimensions
    # Width
    ax.plot([0, b], [-100, -100], color='black', lw=0.8)
    ax.plot([0, 0], [-80, -120], color='black', lw=0.8)
    ax.plot([b, b], [-80, -120], color='black', lw=0.8)
    ax.text(b/2, -180, f"{int(b)} mm", ha='center', va='top', fontsize=GLOBAL_FONT)
    
    # Height
    ax.plot([-100, -100], [0, h], color='black', lw=0.8)
    ax.plot([-80, -120], [0, 0], color='black', lw=0.8)
    ax.plot([-80, -120], [h, h], color='black', lw=0.8)
    
    # --- จุดที่แก้ไข (Fixed Line) ---
    # เปลี่ยน [-180] เป็น -180 เพื่อให้เป็น Scalar
    ax.text(-180, h/2, f"{int(h)} mm", ha='right', va='center', rotation=90, fontsize=GLOBAL_FONT)

    # Title
    ax.text(b/2, view_top - 100, title, ha='center', fontweight='bold', fontsize=GLOBAL_FONT+2)

    ax.set_xlim(view_left, view_right)
    ax.set_ylim(view_bottom, view_top)
    ax.set_aspect('equal')
    ax.axis('off')
    return fig

def plot_longitudinal_section_detailed(spans, sup_df, design_res, h_m, cover_mm):
    """ วาดรูปตัดตามยาว พร้อมระบุตำแหน่งหน้าตัด (Sections) """
    LONG_SCALE = 850
    h_beam = h_m * 1000 
    total_L = sum(spans) * 1000
    offsets = [0] + list(np.cumsum(spans) * 1000)
    
    width_inches = (total_L + 3000) / LONG_SCALE # เผื่อที่ด้านข้างเยอะหน่อย
    height_inches = 5000 / LONG_SCALE 
    fig, ax = plt.subplots(figsize=(width_inches, height_inches), dpi=DPI_VALUE)
    
    # 1. วาดตัวคานหลัก
    ax.add_patch(patches.Rectangle((0, 0), total_L, h_beam, linewidth=2, edgecolor='#000000', facecolor='#ffffff', zorder=1))
    
    # 2. วาด Support (Hinge & Roller)
    for i, (_, row) in enumerate(sup_df.iterrows()):
        x_s = row['x'] * 1000
        # Symbol drawing logic
        if i == 0: # Hinge (Triangle)
            poly = plt.Polygon([[x_s-150, -300], [x_s+150, -300], [x_s, 0]], facecolor='#ecf0f1', edgecolor='black', lw=1.5, zorder=2)
            ax.add_patch(poly)
            # Base marks
            for k in range(-200, 201, 50):
                ax.plot([x_s+k, x_s+k-50], [-300, -400], color='black', lw=1)
            ax.plot([x_s-250, x_s+250], [-300, -300], color='black', lw=2)
        else: # Roller (Circle on base)
            poly = plt.Polygon([[x_s-150, -200], [x_s+150, -200], [x_s, 0]], facecolor='#ecf0f1', edgecolor='black', lw=1.5, zorder=2)
            ax.add_patch(poly)
            ax.add_patch(plt.Circle((x_s, -250), 50, color='black', fill=False, lw=1.5))
            ax.plot([x_s-250, x_s+250], [-310, -310], color='black', lw=2)

    # 3. วาดเหล็กเสริม (Reinforcement)
    for i, span_l_m in enumerate(spans):
        L_mm = span_l_m * 1000
        x_s = offsets[i]
        x_e = offsets[i+1]
        
        # Bottom Steel (วิ่งยาวตลอดช่วง แต่หยุดก่อนถึง support นิดหน่อยในแบบ simplified)
        ax.plot([x_s+50, x_e-50], [cover_mm, cover_mm], color='#c0392b', lw=2.5, label='Bottom Bars' if i==0 else "")
        
        # Top Steel (Extra bars at supports) - ประมาณ 0.25-0.3L
        y_t = h_beam - cover_mm
        ax.plot([x_s, x_s + L_mm*0.3], [y_t, y_t], color='#2980b9', lw=2.5, label='Top Bars' if i==0 else "")
        ax.plot([x_e - L_mm*0.3, x_e], [y_t, y_t], color='#2980b9', lw=2.5)

    # 4. วาดเส้นแนวตัด (Cut Lines) - แสดงทุก Span เพื่อความสวยงาม
    for i in range(len(spans)):
        x_start = offsets[i]
        x_end = offsets[i+1]
        
        # Cut A-A (Mid Span)
        mid_x = (x_start + x_end) / 2
        ax.plot([mid_x, mid_x], [-800, h_beam + 800], color='#e67e22', ls='-.', lw=1.5)
        ax.text(mid_x, h_beam + 900, f"A (S{i+1})", color='#e67e22', fontweight='bold', ha='center', fontsize=GLOBAL_FONT+2)
        
        # Cut B-B (Near Support - End of span)
        # ตัดที่ support ขวาของ span นั้น (ยกเว้น span สุดท้ายตัดซ้ายก็ได้ แต่นิยมตัดหัวเสา)
        sup_x = x_end - (x_end-x_start)*0.1 # ตัดแถวๆ support
        ax.plot([sup_x, sup_x], [-800, h_beam + 800], color='#8e44ad', ls='-.', lw=1.5)
        ax.text(sup_x, h_beam + 900, f"B (S{i+1})", color='#8e44ad', fontweight='bold', ha='center', fontsize=GLOBAL_FONT+2)

    # 5. Dimension Line รวม
    ax.plot([0, total_L], [h_beam + 1500, h_beam + 1500], color='black', lw=1)
    ax.plot([0, 0], [h_beam + 1400, h_beam + 1600], color='black', lw=1)
    ax.plot([total_L, total_L], [h_beam + 1400, h_beam + 1600], color='black', lw=1)
    ax.text(total_L/2, h_beam + 1600, f"Total Length = {total_L/1000:.2f} m", ha='center', va='bottom', fontsize=GLOBAL_FONT+2)

    ax.set_xlim(-1000, total_L + 1000)
    ax.set_ylim(-1500, h_beam + 2500)
    ax.set_aspect('equal')
    ax.axis('off')
    
    return fig
