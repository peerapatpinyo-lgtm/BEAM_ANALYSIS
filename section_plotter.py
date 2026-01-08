import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# --- 🎨 Style Constants ---
COLOR_CONCRETE = '#F5F5F5'  # สีคอนกรีต (เทาอ่อน)
COLOR_STIRRUP = '#2980b9'   # สีเหล็กปลอก (น้ำเงิน)
COLOR_TOP = '#c0392b'       # สีเหล็กบน (แดงเข้ม)
COLOR_BOT = '#27ae60'       # สีเหล็กล่าง (เขียว)
FONT_SIZE = 11              # ขนาดตัวอักษรมาตรฐาน
DIM_OFFSET = 0.15           # ระยะห่างเส้นบอกระยะ (คิดเป็น % ของความลึก)

def _setup_figure():
    """Create a clean canvas"""
    fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    return fig, ax

def plot_section(b_m, h_m, cover_mm, db_top_mm, db_bot_mm, n_top, n_bot, stir_text, fc, fy, title="SECTION"):
    """
    วาดรูปตัดขวาง (Cross Section) แบบ Smart Scaling
    """
    # 1. Convert Units to mm
    b = b_m * 1000.0
    h = h_m * 1000.0
    
    fig, ax = _setup_figure()
    
    # 2. Draw Concrete Face
    rect = patches.Rectangle((0, 0), b, h, linewidth=2, edgecolor='black', facecolor=COLOR_CONCRETE, zorder=0)
    ax.add_patch(rect)
    
    # 3. Draw Stirrup ( เส้นประ )
    sw = b - 2*cover_mm
    sh = h - 2*cover_mm
    stirrup = patches.Rectangle((cover_mm, cover_mm), sw, sh, linewidth=1.5, 
                                edgecolor=COLOR_STIRRUP, linestyle='--', fill=False, zorder=1)
    ax.add_patch(stirrup)
    
    # --- Helper: Draw Rebar ---
    def plot_rebar_layer(n, y_pos, db, color, label_text, is_top=True):
        if n < 2: n = 2 # วาดอย่างน้อย 2 เส้นเพื่อความสวยงาม
        radius = db / 2.0
        
        # คำนวณตำแหน่ง X (กระจายตัวสม่ำเสมอ)
        start_x = cover_mm + db
        end_x = b - cover_mm - db
        
        if n == 1:
            x_locs = [b/2]
        else:
            x_locs = np.linspace(start_x, end_x, int(n))
            
        # วาดวงกลม
        for x in x_locs:
            c = patches.Circle((x, y_pos), radius, facecolor=color, edgecolor='black', linewidth=1, zorder=5)
            ax.add_patch(c)
            
        # วาดลูกศรชี้ (Annotation)
        # ชี้ไปที่เหล็กเส้นแรก หรือตรงกลาง
        target_x = x_locs[-1] if x_locs[-1] < b/2 else x_locs[0]
        
        # ตำแหน่งข้อความ (Offset ออกไปนอกคาน)
        text_y = h + (h*0.15) if is_top else -(h*0.15)
        text_x = b/2
        
        ax.annotate(label_text, 
                    xy=(target_x, y_pos), 
                    xytext=(text_x, text_y),
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.5, connectionstyle="arc3,rad=0.2"),
                    ha='center', va='center', fontsize=FONT_SIZE, fontweight='bold', color=color,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color, alpha=0.9))

    # 4. Draw Top Bars
    y_top = h - cover_mm - 8 - (db_top_mm/2) # 8mm approx stirrup dia
    plot_rebar_layer(n_top, y_top, db_top_mm, COLOR_TOP, f"{int(n_top)}-DB{int(db_top_mm)} (Top)", is_top=True)
    
    # 5. Draw Bottom Bars
    y_bot = cover_mm + 8 + (db_bot_mm/2)
    plot_rebar_layer(n_bot, y_bot, db_bot_mm, COLOR_BOT, f"{int(n_bot)}-DB{int(db_bot_mm)} (Bot)", is_top=False)
    
    # 6. Dimensions (เส้นบอกระยะ)
    # Width (Bottom)
    offset_dim = h * 0.25
    ax.annotate(f"{int(b)}", xy=(0, -20), xytext=(b, -20),
                arrowprops=dict(arrowstyle='<|-|>', color='black', lw=1),
                ha='center', va='bottom', fontsize=FONT_SIZE)
    
    # Height (Left)
    ax.annotate(f"{int(h)}", xy=(-20, 0), xytext=(-20, h),
                arrowprops=dict(arrowstyle='<|-|>', color='black', lw=1),
                ha='right', va='center', rotation=90, fontsize=FONT_SIZE)
    
    # 7. Material Info & Stirrup
    info_box = (
        f"Stirrup: {stir_text}\n"
        f"Cover: {cover_mm} mm\n"
        f"fc': {fc} MPa\n"
        f"fy: {fy} MPa"
    )
    ax.text(b*1.3, h/2, info_box, fontsize=FONT_SIZE-1, color='#555',
            bbox=dict(boxstyle="round,pad=0.5", fc="#f9f9f9", ec="#ddd"),
            ha='left', va='center')

    # 8. Final Settings
    ax.set_title(title, fontsize=FONT_SIZE+2, fontweight='bold', pad=20)
    ax.axis('equal') # สำคัญมาก! ทำให้สัดส่วน กว้างxยาว ไม่เพี้ยน
    ax.axis('off')   # ปิดแกน x,y
    
    # Adjust limits to see everything (Auto margin)
    margin_x = b * 0.5
    margin_y = h * 0.4
    ax.set_xlim(-margin_x, b + margin_x*1.5) # เผื่อที่ขวาเยอะหน่อยสำหรับ Text
    ax.set_ylim(-margin_y, h + margin_y)
    
    return fig

def plot_longitudinal_section_detailed(spans, sup_df, design_res, h_m, cover_mm):
    """
    วาดรูปตัดยาว (Longitudinal) แบบสัดส่วนสมจริง (mm Unit)
    """
    # Convert all to mm
    spans_mm = [s * 1000 for s in spans]
    total_L = sum(spans_mm)
    h_mm = h_m * 1000
    
    # Setup Figure (Wide)
    fig, ax = plt.subplots(figsize=(12, 4), dpi=100)
    fig.patch.set_facecolor('white')
    
    # 1. Beam Body
    beam = patches.Rectangle((0, 0), total_L, h_mm, facecolor='#f0f0f0', edgecolor='black', lw=1.5)
    ax.add_patch(beam)
    
    # 2. Supports
    sup_w = 250  # ขนาดฐานรองรับ (mm)
    for _, row in sup_df.iterrows():
        x_pos = row['x'] * 1000
        # Draw Triangle
        tri = patches.Polygon([[x_pos-sup_w/2, -sup_w], [x_pos+sup_w/2, -sup_w], [x_pos, 0]], 
                              closed=True, facecolor='#7f8c8d', edgecolor='black')
        ax.add_patch(tri)
        ax.text(x_pos, -sup_w - 50, row.get('id', ''), ha='center', fontsize=FONT_SIZE-1)

    # 3. Rebars (Simplified Representation)
    x_cursor = 0
    for i, span_L in enumerate(spans_mm):
        end_cursor = x_cursor + span_L
        res = design_res[i]
        
        mid_x = x_cursor + span_L/2
        
        # --- Top Bar (Red) ---
        ax.plot([x_cursor, end_cursor], [h_mm-cover_mm-15, h_mm-cover_mm-15], 
                color=COLOR_TOP, lw=3, solid_capstyle='round')
        ax.text(mid_x, h_mm+50, f"{res['neg']['n']}-DB{int(res['top_db'])}", 
                color=COLOR_TOP, ha='center', fontweight='bold', fontsize=9)
        
        # --- Bottom Bar (Green) ---
        # เว้นระยะจาก Support นิดหน่อยให้ดูเหมือนเหล็กเสริมล่าง
        gap = span_L * 0.1 
        ax.plot([x_cursor+gap, end_cursor-gap], [cover_mm+15, cover_mm+15], 
                color=COLOR_BOT, lw=3, solid_capstyle='round')
        ax.text(mid_x, -100, f"{res['pos']['n']}-DB{int(res['bot_db'])}", 
                color=COLOR_BOT, ha='center', fontweight='bold', fontsize=9)
        
        # --- Stirrup Text (Middle) ---
        ax.text(mid_x, h_mm/2, f"Stir: RB{int(res['stir_db'])}@{int(res['shear']['s'])}", 
                color=COLOR_STIRRUP, ha='center', va='center', fontsize=8,
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))

        x_cursor += span_L

    # 4. Dimensions Line (Total Length)
    ax.annotate(f"Total Length = {total_L/1000:.2f} m", 
                xy=(0, h_mm+200), xytext=(total_L, h_mm+200),
                arrowprops=dict(arrowstyle='|-|', color='black'),
                ha='center', va='bottom')

    # Settings
    ax.axis('equal')
    ax.axis('off')
    ax.set_xlim(-500, total_L + 500)
    ax.set_ylim(-600, h_mm + 400)
    
    return fig
