import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_section(b, h, cover, db, n_top, n_bot, stirrup_info, fc, fy):
    fig, ax = plt.subplots(figsize=(4, 5))
    ax.add_patch(patches.Rectangle((0, 0), b, h, facecolor='#f0f0f0', edgecolor='black', lw=2))
    
    # Draw bars
    for i in range(int(n_top)):
        x = (cover/1000) + i * ((b - 2*cover/1000)/(n_top-1 if n_top>1 else 1))
        ax.add_patch(patches.Circle((x, h-cover/1000), db/2000, color='red'))
        
    for i in range(int(n_bot)):
        x = (cover/1000) + i * ((b - 2*cover/1000)/(n_bot-1 if n_bot>1 else 1))
        ax.add_patch(patches.Circle((x, cover/1000), db/2000, color='blue'))
        
    ax.set_aspect('equal')
    ax.axis('off')
    plt.title(f"Section {int(b*100)}x{int(h*100)} cm\n{stirrup_info}")
    return fig

def plot_longitudinal_section(spans, supports_df, design_data, h, cover):
    # ใช้ Logic เดิมที่คุณส่งมาในการวาดแนวคานยาว
    fig, ax = plt.subplots(figsize=(10, 2))
    total_l = sum(spans)
    ax.add_patch(patches.Rectangle((0, 0), total_l, h, facecolor='#fffde6', edgecolor='black'))
    ax.set_xlim(-0.5, total_l + 0.5)
    ax.set_ylim(-0.5, h + 0.5)
    ax.axis('off')
    return fig
