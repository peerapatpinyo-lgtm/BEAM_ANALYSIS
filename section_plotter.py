import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.path as mpath
import numpy as np

# --- 🏗️ Engineering Standard Config (Original) ---
COLOR_CONCRETE = '#FFFFFF'
COLOR_DIM      = '#000000'
COLOR_STIRRUP  = '#2c3e50'
COLOR_TOP      = '#c0392b'
COLOR_BOT      = '#27ae60'
FONT_MAIN      = 10
FONT_DIM       = 9

def _setup_figure(figsize):
    fig, ax = plt.subplots(figsize=figsize, dpi=120)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    return fig, ax

def _draw_dim_line(ax, p1, p2, text, offset=0, is_vert=False):
    """วาดเส้น Dimension - คงเดิมทุกบรรทัดตาม Code คุณ"""
    if is_vert:
        x_pos = p1[0] - offset
        mid_y = (p1[1] + p2[1]) / 2
        ax.annotate("", xy=(x_pos, p1[1]), xytext=(x_pos, p2[1]),
                    arrowprops=dict(arrowstyle='<|-|>', color=COLOR_DIM, lw=0.7))
        ax.plot([p1[0], x_pos], [p1[1], p1[1]], color=COLOR_DIM, lw=0.5)
        ax.plot([p2[0], x_pos], [p2[1], p2[1]], color=COLOR_DIM, lw=0.5)
        ax.text(x_pos - 10, mid_y, text, ha='right', va='center', rotation=90, fontsize=FONT_DIM,
                bbox=dict(facecolor='white', edgecolor='none', pad=2))
    else:
        y_pos = p1[1] + offset
        mid_x = (p1[0] + p2[0]) / 2
        ax.annotate("", xy=(p1[0], y_pos), xytext=(p2[0], y_pos),
                    arrowprops=dict(arrowstyle='<|-|>', color=COLOR_DIM, lw=0.7))
        ax.plot([p1[0], p1[0]], [p1[1], y_pos], color=COLOR_DIM, lw=0.5)
        ax.plot([p2[0], p2[0]], [p2[1], y_pos], color=COLOR_DIM, lw=0.5)
        ax.text(mid_x, y_pos, text, ha='center', va='center', fontsize=FONT_DIM,
                bbox=dict(facecolor='white', edgecolor='none', pad=2))

def _draw_support_symbol(ax, x, y, sup_type, sup_id):
    """วาดสัญลักษณ์ Support - คง Logic เดิม แต่ปรับตำแหน่งให้พ้นแนวคาน"""
    size = 200 
    # ขยับ Text ลงมาด้านล่าง
    ax.text(x, y - size - 180, f"S{sup_id}", ha='center', fontsize=9, fontweight='bold')
    
    if sup_type == 'Fixed':
        w, h = 100, 450
        rect = patches.Rectangle((x-w/2, y-h/2), w, h, facecolor='#bdc3c7', edgecolor='black', hatch='///', zorder=5)
        ax.add_patch(rect)
    elif sup_type == 'Pin':
        tri = patches.Polygon([[x, y], [x-size/2, y-size], [x+size/2, y-size]], 
                              closed=True, facecolor='#bdc3c7', edgecolor='black', zorder=5)
        ax.add_patch(tri)
        ax.plot([x-size, x+size], [y-size, y-size], color='black', lw=2, zorder=6)
    elif sup_type == 'Roller':
        tri = patches.Polygon([[x, y], [x-size/2, y-size], [x+size/2, y-size]], 
                              closed=True, facecolor='#bdc3c7', edgecolor='black', zorder=5)
        ax.add_patch(tri)
        wheel_r = 30
        ax.add_patch(patches.Circle((x, y-size-wheel_r), wheel_r, fc='white', ec='black', zorder=6))
        ax.plot([x-size, x+size], [y-size-2*wheel_r, y-size-2*wheel_r], color='black', lw=2, zorder=5)
    else:
        rect = patches.Rectangle((x-80, y-350), 160, 350, fc='#eee', ec='black', zorder=4)
        ax.add_patch(rect)

def plot_section(b_m, h_m, cover_mm, db_top_mm, db_bot_mm, n_top, n_bot, stir_text, fc, fy, title="SECTION A-A"):
    """รูปตัดขวาง - แก้ไขไม่ให้หมุน 90 องศา"""
    b, h = b_m * 1000.0, h_m * 1000.0
    fig, ax = _setup_figure((6, 6))
    
    # Concrete & Stirrup (b=กว้างแนวนอน, h=สูงแนวตั้ง)
    ax.add_patch(patches.Rectangle((0, 0), b, h, lw=2, ec='black', fc='#FAFAFA', zorder=1))
    st_off = cover_mm
    ax.add_patch(patches.Rectangle((st_off, st_off), b-2*st_off, h-2*st_off, 
                                   lw=1.5, ec=COLOR_STIRRUP, fill=False, zorder=2))
    
    # วางเหล็กเรียงตามแนวนอน (แกน X)
    def draw_bars(n, y_pos, db, color):
        if n <= 0: return b/2
        if n == 1:
            xs = [b/2]
        else:
            side_clear = st_off + 10 + db/2
            xs = np.linspace(side_clear, b - side_clear, int(n))
        for x in xs:
            ax.add_patch(patches.Circle((x, y_pos), db/2, fc=color, ec='black', lw=0.8, zorder=10))
        return xs[-1]

    y_top = h - (cover_mm + 10 + db_top_mm/2)
    y_bot = cover_mm + 10 + db_bot_mm/2
    
    last_x_top = draw_bars(n_top, y_top, db_top_mm, COLOR_TOP)
    last_x_bot = draw_bars(n_bot, y_bot, db_bot_mm, COLOR_BOT)
    
    # Labels (คงเดิมตามของคุณ)
    ax.annotate(f"{int(n_top)}-DB{int(db_top_mm)} (Top)", xy=(last_x_top, y_top), 
                xytext=(b+50, h-40), arrowprops=dict(arrowstyle='->', color=COLOR_TOP),
                ha='left', va='center', fontsize=FONT_MAIN, color=COLOR_TOP, fontweight='bold')
    
    ax.annotate(f"{int(n_bot)}-DB{int(db_bot_mm)} (Bot)", xy=(last_x_bot, y_bot), 
                xytext=(b+50, 40), arrowprops=dict(arrowstyle='->', color=COLOR_BOT),
                ha='left', va='center', fontsize=FONT_MAIN, color=COLOR_BOT, fontweight='bold')

    _draw_dim_line(ax, (0, 0), (b, 0), f"{int(b)}", offset=-70)
    _draw_dim_line(ax, (0, 0), (0, h), f"{int(h)}", offset=70, is_vert=True)

    ax.axis('equal')
    ax.axis('off')
    ax.set_xlim(-150, b + 250)
    ax.set_ylim(-150, h + 150)
    return fig

def plot_longitudinal_section_detailed(spans, sup_df, design_res, h_m, cover_mm):
    """รูปตัดตามยาว - แก้ไขการยืดตัวแนวตั้ง และ Text ซ้อนทับ"""
    spans_mm = [s * 1000 for s in spans]
    total_L = sum(spans_mm)
    h_mm = h_m * 1000
    
    # 1. ปรับขนาดรูปให้ยาวออก (Width > Height)
    fig_w = max(12, total_L / 400)
    fig, ax = _setup_figure((fig_w, 5)) 
    
    # 2. วาดโครงคาน
    ax.add_patch(patches.Rectangle((0, 0), total_L, h_mm, lw=2, ec='black', fc='none', zorder=10))
    
    # 3. Supports (ใช้ฟังก์ชันเดิมของคุณ)
    if not sup_df.empty:
        for _, row in sup_df.iterrows():
            _draw_support_symbol(ax, row['x']*1000, 0, row.get('type', 'Pin'), row.get('id', ''))

    x_cursor = 0
    y_top_main = h_mm - (cover_mm + 15)
    y_bot_main = cover_mm + 15

    for i, span_L in enumerate(spans_mm):
        res = design_res[i]
        mid_span = x_cursor + span_L/2
        
        # วาดเหล็กปลอก (Stirrups)
        s_spacing = res['shear']['s']
        num_stir = int(span_L / s_spacing)
        for sx in np.linspace(x_cursor + 50, x_cursor + span_L - 50, num_stir):
            ax.plot([sx, sx], [cover_mm, h_mm-cover_mm], color=COLOR_STIRRUP, lw=0.6, alpha=0.4)

        # เหล็กเสริมบน (Negative)
        L_neg = span_L * 0.30
        ax.plot([x_cursor, x_cursor + L_neg], [y_top_main, y_top_main], color=COLOR_TOP, lw=3, zorder=15)
        ax.plot([x_cursor + span_L - L_neg, x_cursor + span_L], [y_top_main, y_top_main], color=COLOR_TOP, lw=3, zorder=15)
        
        # Labels - ปรับตำแหน่งไม่ให้ซ้อน
        ax.text(mid_span, h_mm + 120, f"{res['neg']['n']}-DB{int(res['top_db'])}", color=COLOR_TOP, ha='center', fontsize=9)
        ax.text(mid_span, y_bot_main + 60, f"{res['pos']['n']}-DB{int(res['bot_db'])}", color=COLOR_BOT, ha='center', fontsize=9)
        # ขยับ Text เหล็กปลอกลงไปข้างล่าง
        ax.text(mid_span, -250, f"RB{int(res['stir_db'])}@{int(s_spacing)}", color=COLOR_STIRRUP, ha='center', fontsize=8)

        x_cursor += span_L

    _draw_dim_line(ax, (0, h_mm), (total_L, h_mm), f"Total L = {total_L/1000:.2f} m", offset=350)
    
    # สำคัญ: เปลี่ยนจาก equal เป็น auto เพื่อให้คานยาวออกแนวข้าง
    ax.set_aspect('auto') 
    ax.axis('off')
    ax.set_xlim(-500, total_L + 500)
    ax.set_ylim(-600, h_mm + 700)
    return fig
