import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# --- 📏 Global Configuration ---
DPI_VALUE = 300      # ความคมชัดระดับสูงสำหรับงาน Engineering
GLOBAL_FONT = 7.5    # ขนาดตัวหนังสือที่ต้องการ (เล็กแต่คม)

def plot_section(b_m, h_m, cover_mm, db_main_mm, n_top, n_bottom, stirrup_name, fc, fy):
    """ 
    วาดรูปตัดขวาง (Cross Section): 
    ขยายรูปให้ใหญ่ขึ้นโดยลดค่า SECTION_SCALE แต่ล็อคตัวหนังสือให้เล็กเท่าเดิม
    """
    # ⬇️ ปรับลดค่านี้เพื่อให้รูปในหน่วยนิ้วใหญ่ขึ้น (ยิ่งน้อย รูปยิ่งใหญ่)
    SECTION_SCALE = 550  
    
    # แปลงหน่วยเป็นมิลลิเมตรเพื่อใช้ในการวาด
    b, h = b_m * 1000, h_m * 1000
    cover = cover_mm
    ds = 6  # สมมติขนาดเหล็กปลอก 6mm
    db = db_main_mm
    
    # กำหนดขอบเขตการมองเห็น (Viewport) ให้มี Padding รอบตัวคานพอดีๆ
    view_left, view_right = -350, b + 750
    view_bottom, view_top = -450, h + 450
    
    # คำนวณขนาดภาพจริงในหน่วยนิ้ว
    width_inches = (view_right - view_left) / SECTION_SCALE 
    height_inches = (view_top - view_bottom) / SECTION_SCALE
    
    fig, ax = plt.subplots(figsize=(width_inches, height_inches), dpi=DPI_VALUE)
    
    # 1. วาดคอนกรีต (Concrete Outline)
    ax.add_patch(patches.Rectangle((0, 0), b, h, linewidth=1.5, edgecolor='#000000', facecolor='#ffffff', zorder=1))
    
    # 2. วาดเหล็กปลอก (Stirrups) - เส้นประแสดงแนวเหล็กปลอก
    ax.add_patch(patches.Rectangle((cover, cover), b-2*cover, h-2*cover, 
                                   linewidth=0.8, edgecolor='#2c3e50', ls='--', facecolor='none', zorder=2))
    
    # 3. ฟังก์ชันวาดเหล็กเสริม (Reinforcing Bars)
    def draw_bars(n, y_pos, color):
        if n < 1: return
        # คำนวณระยะห่างระหว่างเหล็ก
        if n == 1:
            spacings = [b/2]
        else:
            spacing = (b - 2*cover - 2*ds - db) / (n - 1)
            spacings = [cover + ds + db/2 + i*spacing for i in range(int(n))]
            
        for x in spacings:
            ax.add_patch(plt.Circle((x, y_pos), db/2, color=color, zorder=10))

    # ตำแหน่งเหล็กบนและล่าง
    y_bot = cover + ds + db/2
    y_top = h - cover - ds - db/2
    
    draw_bars(n_bottom, y_bot, '#a93226') # เหล็กล่าง (สีแดงเข้ม)
    draw_bars(n_top, y_top, '#1f618d')    # เหล็กบน (สีน้ำเงินเข้ม)
    
    # 4. เขียนข้อความกำกับ (Annotations) - ล็อคขนาดที่ GLOBAL_FONT
    ax.text(b + 70, y_top, f"{int(n_top)}-DB{int(db)}", va='center', color='#1f618d', fontweight='bold', fontsize=GLOBAL_FONT)
    ax.text(b + 70, y_bot, f"{int(n_bottom)}-DB{int(db)}", va='center', color='#a93226', fontweight='bold', fontsize=GLOBAL_FONT)
    ax.text(b/2, h + 120, f"{stirrup_name}", ha='center', color='#1d8348', fontweight='bold', fontsize=GLOBAL_FONT)

    # 5. วาดเส้นมิติ (Dimension Lines)
    def draw_tick_dim(p1, p2, text, vert=False):
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='#000000', lw=0.6)
        tick = 20
        # วาดขีดเฉียง (Architectural Ticks)
        for p in [p1, p2]:
            ax.plot([p[0]-tick, p[0]+tick], [p[1]-tick, p[1]+tick], color='#000000', lw=1)
        
        if vert:
            ax.text(p1[0]-80, (p1[1]+p2[1])/2, text, va='center', ha='right', rotation=90, fontsize=GLOBAL_FONT-1)
        else:
            ax.text((p1[0]+p2[0])/2, p1[1]-80, text, ha='center', va='top', fontsize=GLOBAL_FONT-1)

    draw_tick_dim([0, -150], [b, -150], f"{int(b)}")
    draw_tick_dim([-150, 0], [-150, h], f"{int(h)}", vert=True)
    
    # ตั้งค่ากราฟ
    ax.set_xlim(view_left, view_right)
    ax.set_ylim(view_bottom, view_top)
    ax.set_aspect('equal')
    ax.axis('off')
    
    return fig

def plot_longitudinal_section_detailed(spans, sup_df, design_res, h_m, cover_mm):
    """
    วาดรูปตัดตามยาว (Longitudinal Section): 
    สเกลตัวหนังสือเท่าเดิม แต่สัดส่วนคานยาวตามจริง
    """
    LONG_SCALE = 850 # สเกลปกติสำหรับรูปตามยาวเพื่อให้เห็นครบทุก Span
    h_beam = h_m * 1000 
    total_L = sum(spans) * 1000
    offsets = [0] + list(np.cumsum(spans) * 1000)
    
    width_inches = (total_L + 2000) / LONG_SCALE
    height_inches = 3000 / LONG_SCALE 
    
    fig, ax = plt.subplots(figsize=(width_inches, height_inches), dpi=DPI_VALUE)
    
    # วาดตัวคาน
    ax.add_patch(patches.Rectangle((0, 0), total_L, h_beam, linewidth=1.2, edgecolor='#000000', facecolor='#ffffff'))
    
    # วาด Support และเหล็กเสริม
    for i, span_l_m in enumerate(spans):
        L_mm = span_l_m * 1000
        x_s, x_e = offsets[i], offsets[i+1]
        res = design_res[i]
        mid_x = (x_s + x_e) / 2
        
        # วาดเหล็กล่าง (เส้นตรงตลอดช่วง)
        ax.plot([x_s+50, x_e-50], [cover_mm, cover_mm], color='#a93226', lw=1.5)
        ax.text(mid_x, cover_mm + 60, f"{int(res['pos']['n'])}-DB{int(res['db'])}", 
                ha='center', color='#a93226', fontsize=GLOBAL_FONT, fontweight='bold')
        
        # วาดเหล็กบน (เหล็กเสริมพิเศษที่ Support)
        y_top = h_beam - cover_mm
        ax.plot([x_s, x_s + L_mm*0.3], [y_top, y_top], color='#1f618d', lw=1.5)
        ax.plot([x_e - L_mm*0.3, x_e], [y_top, y_top], color='#1f618d', lw=1.5)
        ax.text(x_s + 100, y_top + 60, f"{int(res['neg']['n'])}-DB{int(res['db'])}", 
                ha='left', color='#1f618d', fontsize=GLOBAL_FONT, fontweight='bold')
        
        # ข้อความเหล็กปลอก
        ax.text(mid_x, h_beam + 150, f"RB6@{int(res['shear']['s'])}", 
                ha='center', color='#1d8348', fontsize=GLOBAL_FONT, fontweight='bold')

    # เส้นบอกระยะรวม
    ax.plot([0, total_L], [-800, -800], color='#000000', lw=0.5)
    ax.text(total_L/2, -950, f"Total Length = {total_L/1000:.2f} m", ha='center', fontsize=GLOBAL_FONT)

    ax.set_xlim(-1000, total_L + 1000)
    ax.set_ylim(-1500, 1500)
    ax.set_aspect('equal')
    ax.axis('off')
    
    return fig
