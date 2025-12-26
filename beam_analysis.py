import numpy as np
import pandas as pd

def run_beam_analysis(spans, supports, loads):
    """
    Direct Stiffness Method with ULTRA-ROBUST Data Handling.
    """
    n_spans = len(spans)
    n_nodes = n_spans + 1
    
    # ==========================================
    # 0. DATA SANITIZER (The Fix)
    # ==========================================
    # แปลง Supports ทั้งหมดให้เป็น List of Strings ที่สะอาดก่อนเริ่มงาน
    clean_supports = []
    
    for i in range(n_nodes):
        sup_type = "None" # Default
        
        # ตรวจสอบว่า i อยู่ในขอบเขตข้อมูล input หรือไม่
        if i < len(supports):
            # 1. ดึงข้อมูลดิบออกมา (แยกกรณี List vs DataFrame)
            if isinstance(supports, list):
                raw = supports[i]
            elif isinstance(supports, (pd.DataFrame, pd.Series)):
                raw = supports.iloc[i]
            else:
                raw = "None"

            # 2. แปลงข้อมูลดิบให้เป็น String (Type Name)
            if isinstance(raw, str):
                sup_type = raw # ถ้าเป็น Text อยู่แล้ว ใช้เลย
            elif isinstance(raw, dict):
                sup_type = raw.get('type', 'None') # ถ้าเป็น Dict ดึง key
            elif hasattr(raw, 'type'): 
                sup_type = raw.type # ถ้าเป็น Object/Series ดึง attribute
            elif isinstance(raw, pd.Series):
                sup_type = raw['type'] if 'type' in raw else 'None'
        
        clean_supports.append(sup_type)

    # ==========================================
    # 1. SYSTEM SETUP
    # ==========================================
    dof_per_node = 2
    total_dof = n_nodes * dof_per_node
    K_global = np.zeros((total_dof, total_dof))
    F_global = np.zeros(total_dof)
    E = 2e6  
    I = 0.001 

    # ==========================================
    # 2. MATRIX ASSEMBLY & LOADING
    # ==========================================
    for i in range(n_spans):
        L = spans[i]
        L2 = L * L; L3 = L2 * L
        
        # Element Stiffness
        k_local = (E * I / L3) * np.array([
            [12,      6*L,    -12,     6*L],
            [6*L,     4*L2,   -6*L,    2*L2],
            [-12,    -6*L,     12,    -6*L],
            [6*L,     2*L2,   -6*L,    4*L2]
        ])

        idx = [2*i, 2*i+1, 2*(i+1), 2*(i+1)+1]
        for r in range(4):
            for c in range(4):
                K_global[idx[r], idx[c]] += k_local[r, c]

        # FEM Calculation
        fem_vec = np.zeros(4) 
        span_loads = [l for l in loads if l.get('span_idx') == i]
        
        for load in span_loads:
            # Handle standard keys with defaults to prevent KeyErrors
            l_type = load.get('type', 'None')
            w = -load.get('w', 0.0)
            P = -load.get('P', 0.0)
            x_loc = load.get('x', 0.0)

            if l_type == 'U' and abs(w) > 0:
                fem_vec[0] += w * L / 2       
                fem_vec[1] += w * L2 / 12     
                fem_vec[2] += w * L / 2       
                fem_vec[3] += -w * L2 / 12    

            elif l_type == 'P' and abs(P) > 0:
                a = x_loc
                b = L - a
                if 0 < a < L:
                    fem_vec[0] += P * (b**2 * (3*a + b)) / L3  
                    fem_vec[1] += P * a * (b**2) / L2          
                    fem_vec[2] += P * (a**2 * (a + 3*b)) / L3  
                    fem_vec[3] += -P * (a**2) * b / L2         

        for r in range(4):
            F_global[idx[r]] -= fem_vec[r]

    # ==========================================
    # 3. BOUNDARY CONDITIONS (Using Clean Data)
    # ==========================================
    free_dof = np.ones(total_dof, dtype=bool)
    vis_supports_data = [] 

    for i in range(n_nodes):
        # ใช้ clean_supports ที่ล้างมาแล้ว ไม่ต้องไปยุ่งกับ input เดิมอีก
        stype = clean_supports[i] 
        vis_supports_data.append({'x': sum(spans[:i]), 'type': stype})

        dof_y = 2*i
        dof_m = 2*i + 1
        
        if stype == "Pin":
            free_dof[dof_y] = False 
        elif stype == "Roller":
            free_dof[dof_y] = False 
        elif stype == "Fixed":
            free_dof[dof_y] = False 
            free_dof[dof_m] = False 

    # ==========================================
    # 4. SOLVER
    # ==========================================
    K_reduced = K_global[np.ix_(free_dof, free_dof)]
    F_reduced = F_global[free_dof]
    
    try:
        D_reduced = np.linalg.solve(K_reduced, F_reduced)
    except np.linalg.LinAlgError:
        D_reduced = np.zeros_like(F_reduced) 

    D_full = np.zeros(total_dof)
    D_full[free_dof] = D_reduced

    # ==========================================
    # 5. POST-PROCESSING
    # ==========================================
    x_coords = []
    shear_vals = []
    moment_vals = []
    span_ids = []
    global_x = 0.0
    
    for i in range(n_spans):
        L = spans[i]
        idx = [2*i, 2*i+1, 2*(i+1), 2*(i+1)+1]
        u = D_full[idx] 
        
        # Local Stiffness again for internal forces
        L2 = L*L; L3 = L2*L
        k_local = (E * I / L3) * np.array([
            [12,      6*L,    -12,     6*L],
            [6*L,     4*L2,   -6*L,    2*L2],
            [-12,    -6*L,     12,    -6*L],
            [6*L,     2*L2,   -6*L,    4*L2]
        ])
        
        # FEM Re-calculation
        fem_vec = np.zeros(4)
        span_loads = [l for l in loads if l.get('span_idx') == i]
        
        for load in span_loads:
            l_type = load.get('type', 'None')
            w = -load.get('w', 0.0)
            P = -load.get('P', 0.0)
            x_loc = load.get('x', 0.0)

            if l_type == 'U':
                fem_vec[0] += w * L / 2
                fem_vec[1] += w * L2 / 12
                fem_vec[2] += w * L / 2
                fem_vec[3] += -w * L2 / 12
            elif l_type == 'P':
                a = x_loc
                b = L - a
                fem_vec[0] += P * (b**2 * (3*a + b)) / L3
                fem_vec[1] += P * a * (b**2) / L2
                fem_vec[2] += P * (a**2 * (a + 3*b)) / L3
                fem_vec[3] += -P * (a**2) * b / L2

        f_end = np.dot(k_local, u) + fem_vec
        V_start = f_end[0]
        M_start = f_end[1] 
        
        # Plotting Points
        num_pts = 100
        x_span = np.linspace(0, L, num_pts)
        for load in span_loads:
            if load.get('type') == 'P':
                lx = load.get('x', 0)
                x_span = np.append(x_span, [lx - 1e-5, lx + 1e-5])
        
        x_span = np.sort(np.unique(x_span))
        x_span = x_span[(x_span >= 0) & (x_span <= L)]

        for x in x_span:
            V_x = V_start
            M_x = M_start + V_start * x
            
            for load in span_loads:
                l_type = load.get('type')
                if l_type == 'U':
                    w_mag = load.get('w', 0.0)
                    V_x -= w_mag * x
                    M_x -= w_mag * x**2 / 2
                elif l_type == 'P':
                    P_mag = load.get('P', 0.0)
                    px = load.get('x', 0.0)
                    if x > px:
                        V_x -= P_mag
                        M_x -= P_mag * (x - px)
            
            x_coords.append(global_x + x)
            shear_vals.append(V_x)
            moment_vals.append(M_x) 
            span_ids.append(i)
            
        global_x += L

    res_df = pd.DataFrame({
        'x': x_coords,
        'shear': shear_vals,
        'moment': moment_vals,
        'span_id': span_ids
    })
    
    return res_df, pd.DataFrame(vis_supports_data)
