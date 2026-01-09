import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# --- 🏗️ Engineering Config ---
COLOR_CONCRETE = '#000000'
COLOR_STIRRUP  = '#95a5a6' # สีเทาสำหรับเหล็กปลอก
COLOR_TOP      = '#e74c3c' # สีแดงเหล็กบน
COLOR_BOT      = '#27ae60' # สีเขียวเหล็กล่าง

def _setup_figure(figsize):
    fig, ax = plt.subplots(figsize=figsize, dpi=120)
    ax.set_facecolor('white')
    return fig, ax

def _draw_support_symbol(ax, x, y, sup_id):
    """วาดเสารองรับจากใต้ท้องคานลงไป (ไม่ทับเนื้อคาน)"""
    # เสา (Column) เริ่มจากใต้คานพอดี
    rect = patches.Rectangle((x-75, y-400), 150, 400, fc='#f8f9fa', ec='black', lw=1, zorder=0)
    ax.add_patch(rect)
    ax.text(x, y - 550, f"S{sup_id}", ha='center', va='top', fontweight='bold', fontsize=9)

def plot_section(b_m, h_m, cover_mm, db_top_mm, db_bot_mm, n_top, n_bot, fc, fy, title="SECTION A-A"):
    """หน้าตัดขวาง: บังคับให้เหล็กอยู่ข้างใน ไม่ทับเส้นขอบ"""
    b, h = b_m * 1000.0, h_m * 1000.0
    fig, ax = _setup_figure((5, 6))
    
    # 1. วาดคอนกรีต (Outline เท่านั้น)
    ax.add_patch(patches.Rectangle((0, 0), b, h, lw=2, ec=COLOR_CONCRETE, fc='none', zorder=1))
    
    # 2. วาดเหล็กปลอก (เส้นประจางๆ)
    st_off = cover_mm
    ax.add_patch(patches.Rectangle((st_off, st_off), b-2*st_off, h-2*st_off, 
                                   lw=1, ec=COLOR_STIRRUP, ls='--', fill=False, zorder=2))
    
    # 3. คำนวณพิกัดเหล็ก (ต้องบวกระยะห่างเพิ่ม 10mm ไม่ให้ทับเหล็กปลอก)
    y_top = h - (st_off + 15 + db_top_mm/2)
    y_bot = st_off + 15 + db_bot_mm/2
    
    def draw_rebars(n, y, db, color):
        if n <= 0: return
        # กระจายเหล็กในระยะที่ปลอดภัย
        xs = np.linspace(st_off + 20, b - st_off - 20, int(n)) if n > 1 else [b/2]
        for x in xs:
            ax.add_patch(patches.Circle((x, y), db/2, fc=color, ec='black', lw=0.5, zorder=5))

    draw_rebars(n_top, y_top, db_top_mm, COLOR_TOP)
    draw_rebars(n_bot, y_bot, db_bot_mm, COLOR_BOT)
    
    ax.set_title(title, pad=20, fontweight='bold')
    ax.axis('equal')
    ax.axis('off')
    ax.set_xlim(-100, b + 100)
    ax.set_ylim(-100, h + 100)
    return fig

def plot_longitudinal_section_detailed(spans, sup_df, design_res, h_m, cover_mm):
    """หน้าตัดตามยาว: ปรับสเกลให้สมส่วน และแยกเส้นเหล็กออกจากขอบคาน"""
    spans_mm = [s * 1000 for s in spans]
    total_L = sum(spans_mm)
    h_mm = h_m * 1000
    
    # สเกลหน้าจอ: กว้างขึ้นตามความยาวคาน (1 เมตร = 3.5 นิ้ว โดยประมาณ)
    fig_w = max(12, total_L / 450)
    fig, ax = _setup_figure((fig_w, 4))
    
    # 1. ตัวคาน (ขอบนอก)
    ax.add_patch(patches.Rectangle((0, 0), total_L, h_mm, lw=1.5, ec=COLOR_CONCRETE, fc='none', zorder=10))
    
    # 2. Support (ใต้คาน)
    if not sup_df.empty:
        for _, row in sup_df.iterrows():
            _draw_support_symbol(ax, row['x']*1000, 0, row.get('id', ''))

    # 3. เหล็กเสริม (พิกัด Y ต้องไม่เท่ากับ 0 หรือ h_mm)
    y_top_line = h_mm - (cover_mm + 15)
    y_bot_line = cover_mm + 15
    
    x_offset = 0
    for i, span_L in enumerate(spans_mm):
        res = design_res[i]
        mid = x_offset + span_L/2
        
        # เหล็กบน (Solid Line)
        ax.plot([x_offset, x_offset + span_L], [y_top_line, y_top_line], color=COLOR_TOP, lw=2.5, zorder=15)
        # เหล็กล่าง (Solid Line)
        ax.plot([x_offset + 50, x_offset + span_L - 50], [y_bot_line, y_bot_line], color=COLOR_BOT, lw=2.5, zorder=15)
        
        # Label ด้านนอกคาน
        ax.text(mid, h_mm + 100, f"{res['neg']['n']}-DB{int(res['top_db'])}", color=COLOR_TOP, ha='center', fontsize=9)
        ax.text(mid, -200, f"RB{int(res['stir_db'])}@{int(res['shear']['s'])}", color='#546e7a', ha='center', fontsize=8)

        x_offset += span_L

    # บังคับรูปให้ยาวออกแนวราบ
    ax.set_aspect('auto')
    ax.axis('off')
    ax.set_xlim(-500, total_L + 500)
    ax.set_ylim(-800, h_mm + 600)
    return fig
