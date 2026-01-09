import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path

# --- 📐 Global Sharpness Settings ---
plt.rcParams['text.antialiased'] = True
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']

def _draw_advanced_support(ax, x, y_bottom, sup_type, sup_id):
    """วาดสัญลักษณ์ Support ระดับตำราวิศวกรรม (Engineering Textbooks)"""
    s = 150 # Scale constant
    
    if sup_type.lower() == 'fixed':
        # สัญลักษณ์การยึดแน่น (Hatched Boundary)
        ax.plot([x-120, x+120], [y_bottom, y_bottom], color='black', lw=2.5)
        for i in range(-120, 130, 30):
            ax.plot([x+i, x+i-30], [y_bottom, y_bottom-60], color='black', lw=1)
            
    elif sup_type.lower() == 'roller':
        # Roller: สามเหลี่ยม + เส้นฐาน + วงกลมเล็ก
        poly = plt.Polygon([[x, y_bottom], [x-80, y_bottom-150], [x+80, y_bottom-150]], 
                           fc='white', ec='black', lw=1.5, zorder=5)
        ax.add_patch(poly)
        ax.add_patch(plt.Circle((x, y_bottom-180), 25, fc='black', zorder=6))
        ax.plot([x-120, x+120], [y_bottom-210, y_bottom-210], color='black', lw=2)
        
    else: # Pin / Hinge
        # Pin: สามเหลี่ยม + เส้นฐานแบบมีรอยขีด (Hatched Base)
        poly = plt.Polygon([[x, y_bottom], [x-90, y_bottom-180], [x+90, y_bottom-180]], 
                           fc='#ecf0f1', ec='black', lw=1.5, zorder=5)
        ax.add_patch(poly)
        ax.plot([x-130, x+130], [y_bottom-180, y_bottom-180], color='black', lw=2)
        for i in range(-120, 140, 40):
            ax.plot([x+i, x+i-20], [y_bottom-180, y_bottom-210], color='black', lw=1)

    # วางชื่อ Support ให้คมชัด
    ax.text(x, y_bottom - 350, f"S{sup_id}", ha='center', va='top', 
            fontsize=12, fontweight='bold', color='#2c3e50')

def plot_longitudinal_section_detailed(spans, sup_df, design_res, h_m, cover_mm):
    """
    รูปตัดตามยาวฉบับสมบูรณ์: ตัวหนังสือคมชัด คานผอมยาว และ Support ถูกต้อง
    """
    spans_mm = [s * 1000 for s in spans]
    total_L = sum(spans_mm)
    v_h = 600 # บังคับความสูงคานให้บาง (Thin Beam Ratio)
    
    # ใช้ DPI 300 และขยายขนาด Figure เพื่อเพิ่มพื้นที่ Pixel ให้ตัวหนังสือ
    fig, ax = plt.subplots(figsize=(20, 5), dpi=300)
    
    # 1. วาดโครงสร้างคาน (Main Beam Body)
    ax.add_patch(patches.Rectangle((0, 0), total_L, v_h, lw=2.5, ec='black', fc='#f9f9f9', zorder=2))
    
    # 2. วาด Support (Engineering Style)
    if not sup_df.empty:
        for _, row in sup_df.iterrows():
            _draw_advanced_support(ax, row['x']*1000, 0, row.get('type', 'Pin'), row.get('id', ''))

    # 3. วาดเหล็กเสริม (Reinforcement)
    # ใช้ระยะ Offset ที่แน่นอน ไม่ทับเส้นขอบ 100%
    y_top = v_h * 0.82
    y_bot = v_h * 0.18
    
    x_curr = 0
    for i, span_L in enumerate(spans_mm):
        res = design_res[i]
        mid = x_curr + span_L/2
        
        # เหล็กบน (Main Top)
        ax.plot([x_curr, x_curr + span_L], [y_top, y_top], color='#e74c3c', lw=3.5, zorder=10)
        # เหล็กล่าง (Main Bot)
        ax.plot([x_curr + 50, x_curr + span_L - 50], [y_bot, y_bot], color='#27ae60', lw=3.5, zorder=10)
        
        # --- ✍️ การจัดการตัวหนังสือ (Text) เพื่อความคมชัดสูงสุด ---
        # ใช้พื้นหลังสีขาวจางๆ (Bbox) ช่วยให้ตัวหนังสือเด่นและไม่แตก
        txt_style = dict(fontweight='bold', fontsize=12, ha='center', 
                         bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
        
        ax.text(mid, v_h + 120, f"{int(res['neg']['n'])}-DB{int(res['top_db'])} (TOP)", color='#c0392b', **txt_style)
        ax.text(mid, y_bot + 60, f"{int(res['pos']['n'])}-DB{int(res['bot_db'])} (BOT)", color='#1e8449', **txt_style)
        
        # รายละเอียดเหล็กปลอก (Stirrups)
        ax.text(mid, -120, f"RB{int(res['stir_db'])} @ {int(res['shear']['s'])} mm", 
                color='#535c68', fontsize=10, style='italic', ha='center')

        x_curr += span_L

    # 4. Final Layout ปรับปรุงสัดส่วน
    ax.set_aspect('auto')
    ax.axis('off')
    
    # ปรับ Margin ให้พอดีกับตัวหนังสือ
    ax.set_xlim(-1000, total_L + 1000)
    ax.set_ylim(-800, v_h + 600)
    
    plt.tight_layout()
    return fig
