import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# --- 🎨 Style Constants ---
COLOR_CONCRETE = '#FFFFFF'  # พื้นหลังคอนกรีต
COLOR_STIRRUP = '#2980b9'   # เหล็กปลอก (น้ำเงิน)
COLOR_TOP = '#c0392b'       # เหล็กบน (แดง)
COLOR_BOT = '#27ae60'       # เหล็กล่าง (เขียว)
FONT_SIZE = 10
DPI_VALUE = 100

def _setup_figure(figsize):
    fig, ax = plt.subplots(figsize=figsize, dpi=DPI_VALUE)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    return fig, ax

def plot_section(b_m, h_m, cover_mm, db_top_mm, db_bot_mm, n_top, n_bot, stir_text, fc, fy, title="SECTION"):
    """ วาด Cross Section พร้อมสเกลที่ถูกต้อง """
    # Convert to mm
    b = b_m * 1000.0
    h = h_m * 1000.0
    
    # Auto Scale Canvas
    fig, ax = _setup_figure((6, 6))
    
    # 1. Concrete Face
    ax.add_patch(patches.Rectangle((0, 0), b, h, lw=2, ec='black', fc='#F9F9F9', zorder=0))
    
    # 2. Stirrup
    sw = b - 2*cover_mm
    sh = h - 2*cover_mm
    ax.add_patch(patches.Rectangle((cover_mm, cover_mm), sw, sh, lw=1.5, ec=COLOR_STIRRUP, ls='--', fill=False, zorder=1))
    
    # 3. Helper to draw bars
    def draw_layer(n, y, db, color, label, is_top):
        if n < 2: n = 2
        radius = db/2
        start_x = cover_mm + db
        end_x = b - cover_mm - db
        
        if n == 1: x_locs = [b/2]
        else: x_locs = np.linspace(start_x, end_x, int(n))
        
        for x in x_locs:
            ax.add_patch(patches.Circle((x, y), radius, fc=color, ec='black', lw=0.5, zorder=5))
            
        # Label with arrow
        text_y = h + h*0.15 if is_top else -h*0.15
        mid_x = b/2
        
        ax.annotate(label, xy=(x_locs[-1], y), xytext=(mid_x, text_y),
                    arrowprops=dict(arrowstyle='->', color=color),
                    ha='center', va='center', fontsize=FONT_SIZE, fontweight='bold', color=color,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color))

    # Draw Bars
    y_top = h - cover_mm - 9 - (db_top_mm/2)
    draw_layer(n_top, y_top, db_top_mm, COLOR_TOP, f"{int(n_top)}-DB{int(db_top_mm)} (Top)", True)
    
    y_bot = cover_mm + 9 + (db_bot_mm/2)
    draw_layer(n_bot, y_bot, db_bot_mm, COLOR_BOT, f"{int(n_bot)}-DB{int(db_bot_mm)} (Bot)", False)
    
    # 4. Dimensions & Info
    # Width
    ax.annotate(f"{int(b)}", xy=(0, -30), xytext=(b, -30), arrowprops=dict(arrowstyle='<|-|>', lw=1), ha='center', va='bottom')
    # Height
    ax.annotate(f"{int(h)}", xy=(-30, 0), xytext=(-30, h), arrowprops=dict(arrowstyle='<|-|>', lw=1), ha='right', va='center', rotation=90)
    
    # Info Box
    info = f"Stirrup: {stir_text}\nCover: {cover_mm} mm\nfc': {fc} MPa\nfy: {fy} MPa"
    ax.text(b*1.2, h/2, info, fontsize=9, bbox=dict(fc='white', ec='#ccc'), va='center')
    
    ax.set_title(title, fontsize=12, fontweight='bold', pad=20)
    ax.axis('equal')
    ax.axis('off')
    
    # Set limits
    ax.set_xlim(-b*0.5, b*1.8)
    ax.set_ylim(-h*0.5, h*1.5)
    
    return fig

def plot_longitudinal_section_detailed(spans, sup_df, design_res, h_m, cover_mm):
    """
    วาดรูปตัดยาว (Longitudinal) ที่ถูกต้องตามหลักวิศวกรรม
    - เหล็กบน (Top): เน้นที่หัวเสา (Support)
    - เหล็กล่าง (Bottom): วิ่งยาวช่วงคาน
    - แสดงแนวตัด A-A และ B-B
    """
    # Convert to mm
    spans_mm = [s * 1000 for s in spans]
    total_L = sum(spans_mm)
    h_mm = h_m * 1000
    
    # Canvas Size
    width_in = max(10, total_L / 800)
    fig, ax = _setup_figure((width_in, 5))
    
    # 1. Beam
    ax.add_patch(patches.Rectangle((0, 0), total_L, h_mm, lw=2, ec='black', fc='#Fcfcfc', zorder=0))
    
    # 2. Supports
    sup_w = 250
    for _, row in sup_df.iterrows():
        x = row['x'] * 1000
        tri = patches.Polygon([[x-sup_w/2, -sup_w], [x+sup_w/2, -sup_w], [x, 0]], closed=True, fc='#bdc3c7', ec='black')
        ax.add_patch(tri)
        ax.text(x, -sup_w-80, f"Sup {row.get('id','')}", ha='center')

    # 3. Reinforcement & Sections
    x_cursor = 0
    
    for i, span_L in enumerate(spans_mm):
        res = design_res[i]
        end_cursor = x_cursor + span_L
        
        # --- A. Reinforcement ---
        
        # 1. Bottom Bar (Main Moment +) - วิ่งยาวเกือบเต็มช่วง
        bot_y = cover_mm + 20
        # เว้นจากขอบเสานิดหน่อย
        ax.plot([x_cursor + 50, end_cursor - 50], [bot_y, bot_y], 
                color=COLOR_BOT, lw=3, label='Bottom')
        ax.text(x_cursor + span_L/2, bot_y + 40, f"{res['pos']['n']}-DB{int(res['bot_db'])}", 
                color=COLOR_BOT, ha='center', fontsize=9, fontweight='bold')

        # 2. Top Bar (Main Moment -) - เน้นที่หัวเสา
        top_y = h_mm - cover_mm - 20
        
        # ช่วงความยาวเหล็กบน (ประมาณ L/3 หรือ L/4 จาก Support)
        L_top = span_L * 0.3 
        
        # Top Bar @ Left Support (ของ Span นี้)
        if i == 0: # Span แรก วาดแค่สั้นๆ หรือ Hook
             ax.plot([x_cursor, x_cursor + L_top], [top_y, top_y], color=COLOR_TOP, lw=3)
        else:
             # ต่อเนื่องจาก Span ก่อนหน้า (วาดข้าม Support)
             # (ในที่นี้วาดแยก Span ใคร Span มัน แต่ให้เห็นภาพว่าอยู่ตรง Support)
             ax.plot([x_cursor, x_cursor + L_top], [top_y, top_y], color=COLOR_TOP, lw=3)
             
        # Top Bar @ Right Support (ของ Span นี้)
        ax.plot([end_cursor - L_top, end_cursor], [top_y, top_y], color=COLOR_TOP, lw=3)
        
        # Label Top Bar (วางไว้ตรงแนว B-B)
        label_x_top = end_cursor - L_top/2
        ax.text(label_x_top, top_y - 60, f"{res['neg']['n']}-DB{int(res['top_db'])}", 
                color=COLOR_TOP, ha='center', fontsize=9, fontweight='bold')

        # 3. Stirrup Text (กลางคาน)
        ax.text(x_cursor + span_L/2, h_mm/2, f"Stir: RB{int(res['stir_db'])}@{int(res['shear']['s'])}", 
                ha='center', va='center', fontsize=8, color=COLOR_STIRRUP,
                bbox=dict(fc='white', ec='none', alpha=0.8))

        # --- B. Section Cut Lines (A-A, B-B) ---
        
        # Line A-A (Mid Span)
        sec_a_x = x_cursor + span_L/2
        ax.vlines(sec_a_x, -200, h_mm + 200, colors='purple', linestyles='dashdot', lw=1)
        ax.text(sec_a_x, h_mm + 250, f"A (S{i+1})", color='purple', ha='center', fontweight='bold', fontsize=10)
        ax.text(sec_a_x, -250, "A", color='purple', ha='center', fontweight='bold', fontsize=10)

        # Line B-B (Near Support - Right Side)
        # ตัดที่ระยะ L_top/2 จากขวาสุด (บริเวณที่มีเหล็กบนเยอะๆ)
        sec_b_x = end_cursor - (span_L * 0.05) # ใกล้ Support ขวา
        if i < len(spans) - 1: # ไม่วาดที่ Support ริมสุดขวา (หรือจะวาดก็ได้)
             ax.vlines(sec_b_x, -200, h_mm + 200, colors='orange', linestyles='dashdot', lw=1)
             ax.text(sec_b_x, h_mm + 250, f"B (S{i+1})", color='orange', ha='center', fontweight='bold', fontsize=10)
             ax.text(sec_b_x, -250, "B", color='orange', ha='center', fontweight='bold', fontsize=10)

        x_cursor += span_L

    # 4. Dimension Total
    ax.annotate(f"Total Length = {total_L/1000:.2f} m", 
                xy=(0, h_mm+400), xytext=(total_L, h_mm+400),
                arrowprops=dict(arrowstyle='|-|', color='black'), ha='center')

    ax.axis('equal')
    ax.axis('off')
    ax.set_xlim(-500, total_L + 500)
    ax.set_ylim(-500, h_mm + 600)
    
    return fig
