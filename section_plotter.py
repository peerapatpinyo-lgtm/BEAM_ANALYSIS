import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# --- 🎨 Professional Style Config ---
COLOR_CONCRETE = '#FFFFFF'     # พื้นหลังขาวสะอาด
COLOR_STIRRUP = '#2c3e50'      # เหล็กปลอก (สีน้ำเงินเข้มเกือบดำ)
COLOR_TOP = '#c0392b'          # เหล็กบน (แดงเข้ม)
COLOR_BOT = '#27ae60'          # เหล็กล่าง (เขียวเข้ม)
COLOR_DIM = '#000000'          # เส้นบอกระยะ (ดำ)
FONT_MAIN = 10                 # ขนาดฟอนต์หลัก
FONT_DIM = 9                   # ขนาดฟอนต์บอกระยะ

def _setup_figure(figsize):
    """สร้าง Canvas พื้นหลังขาว"""
    fig, ax = plt.subplots(figsize=figsize, dpi=120) # DPI สูงขึ้นเพื่อความคมชัด
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    return fig, ax

def _draw_dimension(ax, start, end, text, offset_vec, rotate=0):
    """ฟังก์ชันวาดเส้นบอกระยะแบบมืออาชีพ (ตัวเลขอยู่ตรงกลาง)"""
    # 1. เส้นหลัก (Dimension Line)
    p1 = (start[0] + offset_vec[0], start[1] + offset_vec[1])
    p2 = (end[0] + offset_vec[0], end[1] + offset_vec[1])
    
    ax.annotate("", xy=p1, xytext=p2, 
                arrowprops=dict(arrowstyle='<|-|>', color=COLOR_DIM, lw=0.8, shrinkA=0, shrinkB=0))
    
    # 2. เส้นต่อขา (Extension Lines)
    ax.plot([start[0], p1[0]], [start[1], p1[1]], color=COLOR_DIM, lw=0.5)
    ax.plot([end[0], p2[0]], [end[1], p2[1]], color=COLOR_DIM, lw=0.5)
    
    # 3. ตัวหนังสือ (Centered Text) โดยมีพื้นหลังขาวบังเส้น
    mid_x = (p1[0] + p2[0]) / 2
    mid_y = (p1[1] + p2[1]) / 2
    
    ax.text(mid_x, mid_y, text, ha='center', va='center', rotation=rotate, 
            fontsize=FONT_DIM, color=COLOR_DIM,
            bbox=dict(facecolor='white', edgecolor='none', pad=2.0))

def plot_section(b_m, h_m, cover_mm, db_top_mm, db_bot_mm, n_top, n_bot, stir_text, fc, fy, title="SECTION"):
    """ 
    Cross Section แบบ Professional 
    - ตัวเลขบอกระยะอยู่ตรงกลาง
    - แยก Text เหล็กบน/ล่าง/ปลอก ไม่ให้ทับกัน
    """
    b = b_m * 1000.0
    h = h_m * 1000.0
    
    # Canvas Layout Setup
    fig, ax = _setup_figure((7, 6))
    
    # 1. Concrete Shape
    ax.add_patch(patches.Rectangle((0, 0), b, h, lw=2, ec='black', fc='#F9F9F9', zorder=0))
    
    # 2. Stirrup Shape
    ax.add_patch(patches.Rectangle((cover_mm, cover_mm), b-2*cover_mm, h-2*cover_mm, 
                                   lw=1.5, ec=COLOR_STIRRUP, ls='--', fill=False, zorder=1))
    
    # 3. Rebars Drawing Helper
    def draw_rebars(n, y_pos, db, color):
        if n < 2: n = 2
        r = db/2
        xs = np.linspace(cover_mm + db, b - cover_mm - db, int(n)) if n > 1 else [b/2]
        for x in xs:
            ax.add_patch(patches.Circle((x, y_pos), r, fc=color, ec='black', lw=0.8, zorder=10))
        return xs[-1] # Return last bar position for annotation

    # Draw Top Bars
    y_top = h - cover_mm - 10 - (db_top_mm/2)
    last_x_top = draw_rebars(n_top, y_top, db_top_mm, COLOR_TOP)
    
    # Draw Bottom Bars
    y_bot = cover_mm + 10 + (db_bot_mm/2)
    last_x_bot = draw_rebars(n_bot, y_bot, db_bot_mm, COLOR_BOT)
    
    # --- 4. Annotations (จัดวางไม่ให้ทับกัน) ---
    
    # Label: Top Bars (ชี้ไปทางขวาบน)
    ax.annotate(f"{int(n_top)}-DB{int(db_top_mm)} (Top)", 
                xy=(last_x_top, y_top), xytext=(b + 100, h - 50),
                arrowprops=dict(arrowstyle='->', color=COLOR_TOP, connectionstyle="angle,angleA=0,angleB=90,rad=10"),
                ha='left', va='center', fontsize=FONT_MAIN, fontweight='bold', color=COLOR_TOP)

    # Label: Bottom Bars (ชี้ไปทางขวาล่าง)
    ax.annotate(f"{int(n_bot)}-DB{int(db_bot_mm)} (Bot)", 
                xy=(last_x_bot, y_bot), xytext=(b + 100, 50),
                arrowprops=dict(arrowstyle='->', color=COLOR_BOT, connectionstyle="angle,angleA=0,angleB=90,rad=10"),
                ha='left', va='center', fontsize=FONT_MAIN, fontweight='bold', color=COLOR_BOT)

    # Label: Stirrup (ชี้ไปที่มุมปลอก - ด้านซ้ายบน)
    ax.annotate(f"Stirrup: {stir_text}", 
                xy=(cover_mm, h-cover_mm), xytext=(-150, h + 50),
                arrowprops=dict(arrowstyle='->', color=COLOR_STIRRUP),
                ha='right', va='center', fontsize=FONT_MAIN, color=COLOR_STIRRUP)

    # --- 5. Professional Dimensions (ตัวเลขอยู่ตรงกลาง) ---
    
    # Dimension Width (ด้านล่าง)
    _draw_dimension(ax, (0, 0), (b, 0), f"{int(b)} mm", (0, -60))
    
    # Dimension Height (ด้านซ้าย)
    _draw_dimension(ax, (0, 0), (0, h), f"{int(h)} mm", (-60, 0), rotate=90)

    # Material Info Box (มุมขวาล่าง นอกรูป)
    info_text = f"Cover: {cover_mm} mm\nfc': {fc} MPa\nfy: {fy} MPa"
    ax.text(b + 100, h/2, info_text, fontsize=9, color='#666', va='center', 
            bbox=dict(facecolor='#f0f0f0', edgecolor='none', pad=5))

    # Title
    ax.set_title(title, fontsize=12, fontweight='bold', pad=30)
    
    # Final View Settings
    ax.axis('equal')
    ax.axis('off')
    ax.set_xlim(-150, b + 300) # เผื่อที่ขวาเยอะๆ ให้ Text
    ax.set_ylim(-150, h + 150)
    
    return fig

def plot_longitudinal_section_detailed(spans, sup_df, design_res, h_m, cover_mm):
    """
    Longitudinal Section แบบ Professional
    - เส้น BB มาครบ
    - ตัวหนังสือ Stirrup ไม่ทับเหล็กหลัก (ย้ายลงล่าง)
    """
    # Unit Setup
    spans_mm = [s * 1000 for s in spans]
    total_L = sum(spans_mm)
    h_mm = h_m * 1000
    
    # Canvas
    fig, ax = _setup_figure((12, 5))
    
    # 1. Beam Body
    ax.add_patch(patches.Rectangle((0, 0), total_L, h_mm, lw=2, ec='black', fc='#FDFDFD', zorder=0))
    
    # 2. Supports
    sup_w = 300
    for _, row in sup_df.iterrows():
        x = row['x'] * 1000
        # Triangle
        ax.add_patch(patches.Polygon([[x-sup_w/2, -sup_w], [x+sup_w/2, -sup_w], [x, 0]], 
                                     closed=True, fc='#bdc3c7', ec='black'))
        # Text ID
        ax.text(x, -sup_w - 60, str(row.get('id','')), ha='center', fontsize=9, fontweight='bold')

    # 3. Span Loop
    x_cursor = 0
    for i, span_L in enumerate(spans_mm):
        res = design_res[i]
        end_cursor = x_cursor + span_L
        mid_span = x_cursor + span_L/2
        
        # --- A. Reinforcement ---
        
        # Bottom Bar (สีเขียว) - ยาวเกือบเต็มช่วง
        bot_y = cover_mm + 25
        ax.plot([x_cursor + 100, end_cursor - 100], [bot_y, bot_y], color=COLOR_BOT, lw=3)
        # Text Bottom (วางเหนือเส้นนิดหน่อย)
        ax.text(mid_span, bot_y + 40, f"{res['pos']['n']}-DB{int(res['bot_db'])}", 
                color=COLOR_BOT, ha='center', fontsize=9, fontweight='bold')

        # Top Bar (สีแดง) - เน้นช่วงหัวเสา
        top_y = h_mm - cover_mm - 25
        L_neg = span_L * 0.25 # ระยะล้วงเหล็กบน (25% ของช่วงคาน)
        
        # Left Support (ต่อเนื่อง)
        ax.plot([x_cursor, x_cursor + L_neg], [top_y, top_y], color=COLOR_TOP, lw=3)
        # Right Support (ต่อเนื่อง)
        ax.plot([end_cursor - L_neg, end_cursor], [top_y, top_y], color=COLOR_TOP, lw=3)
        
        # Text Top (วางใต้เส้นนิดหน่อย ตรงช่วงหัวเสาขวา)
        ax.text(end_cursor - L_neg/2, top_y - 60, f"{res['neg']['n']}-DB{int(res['top_db'])}", 
                color=COLOR_TOP, ha='center', fontsize=9, fontweight='bold')

        # --- B. Stirrup Text (ย้ายลงมาใต้คาน ไม่ให้ทับเหล็ก) ---
        stir_text = f"Stir: RB{int(res['stir_db'])}@{int(res['shear']['s'])}"
        ax.text(mid_span, -100, stir_text, color=COLOR_STIRRUP, ha='center', va='top', fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=COLOR_STIRRUP, lw=0.5))
        # เส้นชี้ขึ้นไปที่คานเพื่อให้รู้ว่าเป็นของช่วงนี้
        ax.annotate("", xy=(mid_span, 0), xytext=(mid_span, -90), 
                    arrowprops=dict(arrowstyle='-', color=COLOR_STIRRUP, lw=0.5, linestyle=':'))

        # --- C. Section Lines (เส้นแนวตัด) ---
        
        # Line A-A (Mid Span)
        ax.vlines(mid_span, -sup_w, h_mm + 200, colors='purple', linestyles='dashdot', lw=1)
        ax.text(mid_span, h_mm + 220, f"A-{i+1}", color='purple', ha='center', fontweight='bold')

        # Line B-B (Right Support Face) - บังคับวาดทุก Span
        sec_b_x = end_cursor - 100 # ถอยจาก support นิดนึงให้เห็นชัด
        ax.vlines(sec_b_x, -sup_w, h_mm + 200, colors='orange', linestyles='dashdot', lw=1)
        ax.text(sec_b_x, h_mm + 220, f"B-{i+1}", color='orange', ha='center', fontweight='bold')

        x_cursor += span_L

    # 4. Total Dimension (ตัวเลขอยู่ตรงกลาง)
    _draw_dimension(ax, (0, h_mm), (total_L, h_mm), f"Total Length = {total_L/1000:.2f} m", (0, 350))

    # Final Settings
    ax.axis('equal')
    ax.axis('off')
    ax.set_xlim(-500, total_L + 500)
    ax.set_ylim(-500, h_mm + 500)
    
    return fig
