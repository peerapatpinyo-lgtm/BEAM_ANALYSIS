import streamlit as st
import numpy as np
import pandas as pd
import design_view  # เรียกใช้ไฟล์ view ใหม่ของเรา

# --- Mock Calculation Function (จำลองการคำนวณ) ---
# หมายเหตุ: คุณต้องแทนที่ฟังก์ชันนี้ด้วย Library ของคุณ (เช่น IndetermBeam)
# แต่ผมเขียนตรงนี้เพื่อให้ Code นี้ "Run ได้ทันที" เพื่อดูผลกราฟครับ
def calculate_beam(L1, L2, load_type, w, P, x_pos, start_loc, end_loc):
    # สร้างแกน X ละเอียดๆ สำหรับ Plot
    total_len = L1 + L2
    x_plot = np.linspace(0, total_len, 200)
    
    # สร้าง Dummy Data (เส้นกราฟสมมติ) เพื่อทดสอบการวาดกราฟ
    # *** (เปลี่ยนตรงนี้เป็น beam.get_shear() ของจริงของคุณ) ***
    if load_type == "Distributed Load (UDL)":
        # จำลองกราฟ Shear แบบเส้นตรง (Load แผ่)
        shear_y = 2500 - (1000 * x_plot) # Dummy
        shear_y[x_plot > end_loc] = -1500 # ตัด Load
        moment_y = 2000 * x_plot - (500 * x_plot**2) # Dummy Parabola
    else:
        # จำลองกราฟ Point Load
        shear_y = np.where(x_plot < x_pos, 1000, -1000)
        moment_y = np.where(x_plot < x_pos, 1000*x_plot, 1000*x_pos - 1000*(x_plot-x_pos))

    # จำลอง Reactions
    reactions = np.array([1500.0, 3000.0, 1500.0])
    
    # จำลอง Table Data
    df_res = pd.DataFrame({
        "Type": ["Max Shear", "Max Moment"],
        "Value": [np.max(np.abs(shear_y)), np.max(np.abs(moment_y))],
        "Position": [0.0, total_len/2]
    })
    
    return x_plot, shear_y, moment_y, reactions, df_res

# --- Main Application ---
def main():
    st.set_page_config(page_title="Pro Beam Analysis", layout="wide")
    st.title("🏗️ Professional Beam Analysis")

    # 1. Inputs
    with st.sidebar:
        st.header("⚙️ Settings")
        L1 = st.number_input("Span 1 Length (m)", 2.0, 20.0, 5.0)
        L2 = st.number_input("Span 2 Length (m)", 2.0, 20.0, 5.0)
        
        st.markdown("---")
        st.subheader("Load Configuration")
        load_type = st.selectbox("Type", ["Distributed Load (UDL)", "Point Load"])
        
        # ตัวแปรสำหรับเก็บ Load ไปวาดกราฟ
        vis_loads = []
        w, P, x_pos = 0, 0, 0
        start_loc, end_loc = 0, 0

        if load_type == "Distributed Load (UDL)":
            w = st.number_input("Load (kg/m)", value=1000.0)
            span_opt = st.radio("Apply to:", ["Span 1", "Span 2", "Both"])
            
            # --- FIXED LOGIC: ป้องกัน Load เกินช่วงคาน ---
            if span_opt == "Span 1":
                start_loc, end_loc = 0.0, L1
            elif span_opt == "Span 2":
                start_loc, end_loc = L1, L1 + L2
            else:
                start_loc, end_loc = 0.0, L1 + L2
            
            # เก็บข้อมูลเพื่อส่งไปวาดกราฟ ('udl', value, start, end)
            vis_loads.append(('udl', w, start_loc, end_loc))

        else:
            P = st.number_input("Point Load (kg)", value=2000.0)
            x_pos = st.number_input("Position (m)", 0.0, L1+L2, L1)
            # เก็บข้อมูลเพื่อส่งไปวาดกราฟ ('point', value, position)
            vis_loads.append(('point', P, x_pos))

    # 2. Calculation & Process
    if st.button("Run Analysis", type="primary"):
        
        # --- เรียกฟังก์ชันคำนวณ (หรือ Library ของคุณ) ---
        x, v, m, reacts, df_res = calculate_beam(L1, L2, load_type, w, P, x_pos, start_loc, end_loc)
        
        # 3. Display Outputs
        
        # เรียกใช้กราฟตัวใหม่ (Professional Style)
        # Support อยู่ที่ 0, L1, และ L1+L2
        supports_locs = [0, L1, L1+L2] 
        
        try:
            fig = design_view.plot_professional_diagrams(
                L_total=L1+L2,
                loads=vis_loads,          # ส่งข้อมูล Load ที่เตรียมไว้
                reactions_locs=supports_locs,
                shear_x=x, shear_y=v,
                moment_x=x, moment_y=m
            )
            st.pyplot(fig)
            
            # เรียกใช้ตารางผลลัพธ์
            design_view.render_result_tables(df_res, reacts, [L1, L2])
            
        except Exception as e:
            st.error(f"An error occurred during rendering: {e}")

if __name__ == "__main__":
    main()
