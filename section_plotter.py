import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# --- 🏗️ Config ---
COLOR_CONCRETE = '#000000' # เส้นขอบคาน
COLOR_STIRRUP  = '#7f8c8d' # เส้นเหล็กปลอก
COLOR_TOP      = '#c0392b' # เหล็กบน
COLOR_BOT      = '#27ae60' # เหล็กล่าง

def _setup_figure(figsize):
    fig, ax = plt.subplots(figsize=figsize, dpi=120)
    ax.set_facecolor('white')
    return fig, ax

def _draw_support_symbol(ax, x, y, sup_type, sup_id):
    """วาด Support ให้ต่อจากใต้ท้องคานพอดี ไม่ทับเข้าไปในคาน"""
    # วาดเสารองรับจากใต้คานลงไป (y คือท้องคาน)
    ax.add_patch(patches.Rectangle((x-60, y-400), 120, 400, fc='#ecf0f1', ec='black', lw=1, zorder=0))
    ax.text(x, y - 550, f"S{sup_id}", ha='center', fontweight='bold', fontsize=9)

def plot_section(b_m, h_m, cover_mm, db_top_mm, db_bot_mm, n_top, n_bot, stir_text, fc, fy, title="SECTION A-A"):
    b, h = b_m * 1000.0, h_m * 1000.0
    fig, ax = _setup_figure((5, 6))
    
    # 1. Concrete Outline (zorder=1)
    ax.add_patch(patches.Rectangle((0, 0), b, h, lw=2, ec='black', fc='none', zorder=1))
    
    # 2. Stirrup (ต้องอยู่ข้างใน Concrete)
    st_off = cover_mm
    ax.add_patch(patches.Rectangle((st_off, st_off), b-2*st_off, h-2*st_off, lw=1, ec=COLOR_STIRRUP, fill=False, zorder=2))
    
    # 3. Main Rebars (zorder=3 - อยู่บนสุดและเยื้องจากขอบ)
    # เหล็กบน: เยื้องลงมาจากขอบบน = cover + 10mm
    y_t = h - st_off - 10 - (db_top_mm/2)
    # เหล็กล่าง: เยื้องขึ้นมาจากขอบล่าง = cover + 10mm
    y_b = st_off + 10 + (db_bot_mm/2)
    
    def draw_bars(n, y, db, color):
        if n <= 0: return
        xs = np.linspace(st_off+15, b-st_off-15, int(n)) if n > 1 else [b/2]
        for x in xs:
            ax.add_patch(patches.Circle((x, y), db/2, fc=color, ec='black', lw=0.5, zorder=3))

    draw_bars(n_top, y_t, db_top_mm, COLOR_TOP)
    draw_bars(n_bot, y_b, db_bot_mm, COLOR_BOT)
    
    ax.text(b/2, h+80, title, ha='center', fontweight='bold')
    ax.axis('equal')
    ax.axis('off')
    ax.set_xlim(-100, b+100)
    return fig

def plot_longitudinal_section_detailed(spans, sup_df, design_res, h_m, cover_mm):
    spans_mm = [s * 1000 for s in spans]
    total_L = sum(spans_mm)
    h_mm = h_m * 1000
    
    # ปรับ figsize ให้คานดูผอมยาว (กว้าง 15 นิ้ว สูง 3 นิ้ว)
    fig, ax = _setup_figure((15, 3))
    
    # 1. Beam Outline (zorder=5)
    ax.add_patch(patches.Rectangle((0, 0), total_L, h_mm, lw=1.5, ec='black', fc='none', zorder=5))
    
    # 2. Support (วาดจาก y=0 ลงไป ไม่ทับคาน)
    if not sup_df.empty:
        for _, row in sup_df.iterrows():
            _draw_support_symbol(ax, row['x']*1000, 0, row.get('type','Pin'), row.get('id',''))

    # 3. เหล็กเสริม (พิกัด Y ต้องอยู่ "ข้างใน" 0 ถึง h_mm)
    x_curr = 0
    # ให้เหล็กห่างจากขอบบน/ล่าง = cover + 15mm เพื่อความชัดเจน
    y_t = h_mm - cover_mm - 15
    y_b = cover_mm + 15

    for i, span_L in enumerate(spans_mm):
        res = design_res[i]
        mid = x_curr + span_L/2
        
        # เหล็กบน (zorder=10)
        ax.plot([x_curr, x_curr + span_L], [y_t, y_t], color=COLOR_TOP, lw=2, zorder=10)
        # เหล็กล่าง (zorder=10)
        ax.plot([x_curr + 50, x_curr + span_L - 50], [y_b, y_b], color=COLOR_BOT, lw=2, zorder=10)
        
        # Label (ขยับออกไปนอกตัวคาน)
        ax.text(mid, h_mm + 50, f"{res['neg']['n']}-DB{int(res['top_db'])}", color=COLOR_TOP, ha='center', fontsize=8)
        ax.text(mid, -150, f"RB{int(res['stir_db'])}@{int(res['shear']['s'])}", color=COLOR_STIRRUP, ha='center', fontsize=8)

        x_curr += span_L

    ax.set_aspect('auto') # หัวใจสำคัญที่ทำให้คานดูผอมยาว
    ax.axis('off')
    ax.set_xlim(-500, total_L + 500)
    ax.set_ylim(-700, h_mm + 500)
    return fig
