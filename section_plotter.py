import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# --- 🏗️ Configuration (Strict Layers) ---
Z_SUPPORT  = 0   # อยู่ล่างสุด
Z_CONCRETE = 1   # ตัวคาน
Z_STIRRUP  = 2   # เหล็กปลอก
Z_REBAR    = 5   # เหล็กเมน (บนสุด)

COLOR_TOP = '#e74c3c'
COLOR_BOT = '#27ae60'

def _setup_figure(figsize):
    fig, ax = plt.subplots(figsize=figsize, dpi=120)
    ax.set_facecolor('white')
    return fig, ax

def _draw_support_symbol(ax, x, y_bottom_beam, sup_id):
    """แนวคิดใหม่: วาดจากใต้ท้องคานลงไป 100% ไม่มีการทับซ้อน"""
    # วาดสี่เหลี่ยมรองรับ เริ่มที่ y_bottom_beam และขยายลงไปด้านล่างเท่านั้น
    rect = patches.Rectangle((x-70, y_bottom_beam - 400), 140, 400, 
                             fc='#f1f2f6', ec='black', lw=1, zorder=Z_SUPPORT)
    ax.add_patch(rect)
    ax.text(x, y_bottom_beam - 550, f"S{sup_id}", ha='center', fontweight='bold', fontsize=9)

def plot_section(b_m, h_m, cover_mm, db_top_mm, db_bot_mm, n_top, n_bot, fc, fy, title="SECTION A-A"):
    """หน้าตัดขวาง: ใช้ระบบ Offset ไม่ให้เหล็กแตะเส้นขอบ"""
    b, h = b_m * 1000.0, h_m * 1000.0
    fig, ax = _setup_figure((5, 6))
    
    # 1. เส้นขอบคอนกรีต (zorder=1)
    ax.add_patch(patches.Rectangle((0, 0), b, h, lw=2, ec='black', fc='none', zorder=Z_CONCRETE))
    
    # 2. ระยะปลอดภัย (Gap) - บังคับให้เหล็กห่างจากขอบคอนกรีตมากกว่า cover
    safety_gap = cover_mm + 15 
    
    def draw_clean_bars(n, y_pos, db, color):
        if n <= 0: return
        # กระจายเหล็กโดยให้ห่างจากขอบซ้ายขวาเท่ากับ safety_gap
        xs = np.linspace(safety_gap, b - safety_gap, int(n)) if n > 1 else [b/2]
        for x in xs:
            ax.add_patch(patches.Circle((x, y_pos), db/2, fc=color, ec='black', lw=0.6, zorder=Z_REBAR))

    # ตำแหน่ง Y ของเหล็กบนและล่าง (คำนวณให้ลอยอยู่ในคาน)
    y_top = h - safety_gap
    y_bot = safety_gap
    
    draw_clean_bars(n_top, y_top, db_top_mm, COLOR_TOP)
    draw_clean_bars(n_bot, y_bot, db_bot_mm, COLOR_BOT)
    
    ax.set_title(title, pad=20, fontweight='bold')
    ax.axis('equal')
    ax.axis('off')
    ax.set_xlim(-100, b + 100)
    return fig

def plot_longitudinal_section_detailed(spans, sup_df, design_res, h_m, cover_mm):
    """หน้าตัดตามยาว: แก้ไขสัดส่วนให้คานผอมยาว และเหล็กไม่ทับขอบ"""
    spans_mm = [s * 1000 for s in spans]
    total_L = sum(spans_mm)
    h_mm = h_m * 1000
    
    # บังคับสัดส่วนรูปภาพให้เป็นแนวนอน (15 นิ้ว x 3 นิ้ว)
    fig, ax = _setup_figure((15, 3))
    
    # 1. ขอบคอนกรีต (zorder=1)
    ax.add_patch(patches.Rectangle((0, 0), total_L, h_mm, lw=1.5, ec='black', fc='none', zorder=Z_CONCRETE))
    
    # 2. ระยะห่างเหล็กจากขอบบน/ล่าง (Offset)
    rebar_gap = cover_mm + 15
    y_top = h_mm - rebar_gap
    y_bot = rebar_gap
    
    # 3. วาด Support (เริ่มจาก y=0 ลงไป)
    if not sup_df.empty:
        for _, row in sup_df.iterrows():
            _draw_support_symbol(ax, row['x']*1000, 0, row.get('id', ''))

    x_cursor = 0
    for i, span_L in enumerate(spans_mm):
        res = design_res[i]
        mid = x_cursor + span_L/2
        
        # เหล็กบน (Solid Line) - ไม่ทับเส้นขอบคานแน่นอน
        ax.plot([x_cursor, x_cursor + span_L], [y_top, y_top], color=COLOR_TOP, lw=2.5, zorder=Z_REBAR)
        
        # เหล็กล่าง (Solid Line)
        ax.plot([x_cursor + 50, x_cursor + span_L - 50], [y_bot, y_bot], color=COLOR_BOT, lw=2.5, zorder=Z_REBAR)
        
        # ตัวหนังสือ (อยู่นอกตัวคาน)
        ax.text(mid, h_mm + 100, f"{res['neg']['n']}-DB{int(res['top_db'])}", color=COLOR_TOP, ha='center', fontsize=8)
        ax.text(mid, -200, f"RB{int(res['stir_db'])}@{int(res['shear']['s'])}", color='#576574', ha='center', fontsize=8)

        x_cursor += span_L

    # บังคับสเกลให้ยืดตามแนวนอน (ไม่ใช้ equal)
    ax.set_aspect('auto')
    ax.axis('off')
    ax.set_xlim(-500, total_L + 500)
    ax.set_ylim(-700, h_mm + 600)
    return fig
