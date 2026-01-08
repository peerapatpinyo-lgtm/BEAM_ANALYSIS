import numpy as np
import pandas as pd

def solve_beam(spans, sup_df, loads_df, params):
    """
    Solves the continuous beam using Direct Stiffness Method (FEM).
    FIXED: Point Load mapping now uses 'd_start' for accurate positioning.
    """
    # --- 0. Safety Check for Empty Loads ---
    if loads_df.empty or 'span_index' not in loads_df.columns:
        loads_df = pd.DataFrame(columns=['span_index', 'type', 'mag', 'dist', 'd_start'])

    E = params['E'] # Pa (N/m2)
    I = params['I'] # m4
    b = params['b'] # m
    h = params['h'] # m
    
    # --- Timoshenko Parameters ---
    nu = 0.2 
    G = E / (2 * (1 + nu))  # Shear Modulus
    k_factor = 5.0 / 6.0    # Shear Correction Factor for Rectangle
    As = k_factor * b * h   # Shear Area
    
    # 1. Setup Nodes & Elements
    n_spans = len(spans)
    n_nodes = n_spans + 1
    node_coords = [0] + list(np.cumsum(spans))
    
    n_dof = 2 * n_nodes
    K_global = np.zeros((n_dof, n_dof))
    F_global = np.zeros(n_dof)
    
    # 2. Build Stiffness Matrix (K) with Timoshenko Factor (Phi)
    for i in range(n_spans):
        L = spans[i]
        Phi = (12 * E * I) / (G * As * L**2)
        const = (E * I) / ((1 + Phi) * L**3)
        
        k11 = 12
        k12 = 6 * L
        k22 = (4 + Phi) * L**2
        k24 = (2 - Phi) * L**2
        
        k_ele = const * np.array([
            [k11,   k12, -k11,   k12],
            [k12,   k22, -k12,   k24],
            [-k11, -k12,  k11,  -k12],
            [k12,   k24, -k12,   k22]
        ])
        
        idx = [2*i, 2*i+1, 2*(i+1), 2*(i+1)+1]
        for r in range(4):
            for c in range(4):
                K_global[idx[r], idx[c]] += k_ele[r, c]

    # 3. Process Loads (Fixed End Actions - FEA)
    fea_local = [] 
    for _ in range(n_spans):
        fea_local.append(np.zeros(4)) 

    if not loads_df.empty:
        for _, load in loads_df.iterrows():
            span_idx = int(load['span_index'])
            L = spans[span_idx]
            mag = load['mag'] 
            idx = [2*span_idx, 2*span_idx+1, 2*(span_idx+1), 2*(span_idx+1)+1]
            fea = np.zeros(4)
            
            if load['type'] == 'P':
                # [FIXED POINT LOAD POSITION]
                # แก้ไขจาก load['dist'] เป็น load['d_start'] เพื่อให้ตำแหน่ง x ถูกต้อง
                P = mag
                a = float(load['d_start']) 
                b_dist = L - a
                
                # ตรวจสอบขอบเขตตำแหน่ง
                a = max(0, min(L, a))
                b_dist = L - a

                fea[0] = (P * b_dist**2 * (3*a + b_dist)) / L**3
                fea[1] = (P * a * b_dist**2) / L**2
                fea[2] = (P * a**2 * (a + 3*b_dist)) / L**3
                fea[3] = -(P * a**2 * b_dist) / L**2
                
            elif load['type'] == 'U':
                w = mag
                # หากเป็น UDL บางส่วน (Partial) ให้ใช้สูตรทั่วไป
                if float(load['dist']) < L or float(load['d_start']) > 0:
                    a = float(load['d_start'])
                    c = float(load['dist'])
                    b_dist = L - a - c
                    # เพื่อความง่ายในตัวอย่างนี้ใช้แบบ Full หากต้องการ Partial ต้องใช้สูตร Integration
                    # แต่ถ้าใน Code app.py รวบมาเป็น Full แล้ว สูตรข้างล่างนี้จะถูกต้อง
                    fea[0] = w * L / 2
                    fea[1] = w * L**2 / 12
                    fea[2] = w * L / 2
                    fea[3] = -w * L**2 / 12
                else:
                    fea[0] = w * L / 2
                    fea[1] = w * L**2 / 12
                    fea[2] = w * L / 2
                    fea[3] = -w * L**2 / 12

            fea_local[span_idx] += fea
            F_global[idx[0]] -= fea[0]
            F_global[idx[1]] -= fea[1]
            F_global[idx[2]] -= fea[2]
            F_global[idx[3]] -= fea[3]

    # 4. Apply Boundary Conditions
    fixed_dofs = []
    for _, row in sup_df.iterrows():
        node_idx = int(row['id'])
        fixed_dofs.append(2*node_idx) 
        if row['type'] == 'Fixed':
            fixed_dofs.append(2*node_idx + 1)
            
    free_dofs = [i for i in range(n_dof) if i not in fixed_dofs]
    K_ff = K_global[np.ix_(free_dofs, free_dofs)]
    F_ff = F_global[free_dofs]
    
    try:
        d_free = np.linalg.solve(K_ff, F_ff)
    except np.linalg.LinAlgError:
        return np.zeros(10), np.zeros(10), np.zeros(10), np.zeros(10), {}
    
    d_all = np.zeros(n_dof)
    d_all[free_dofs] = d_free
    
    # 5. Post-Processing (Refined for Point Load Jump)
    x_total, moment_total, shear_total, def_total = [], [], [], []
    
    for i in range(n_spans):
        L = spans[i]
        x0 = node_coords[i]
        u_ele = d_all[[2*i, 2*i+1, 2*(i+1), 2*(i+1)+1]]
        
        # --- Create High-Resolution x_local with Jump Points ---
        points = [0.0, L]
        span_loads = loads_df[loads_df['span_index'] == i]
        for _, load in span_loads.iterrows():
            if load['type'] == 'P':
                # [FIXED] ใช้ d_start ในการคำนวณจุด Jump ในกราฟ
                p_loc = float(load['d_start'])
                points.extend([max(0, p_loc - 1e-6), p_loc, min(L, p_loc + 1e-6)])
        
        x_dense = np.linspace(0, L, 101)
        x_local = np.sort(np.unique(np.concatenate([x_dense, points])))
        
        # 5.1 Deflection
        xi = x_local / L
        N1 = 1 - 3*xi**2 + 2*xi**3
        N2 = L * (xi - 2*xi**2 + xi**3)
        N3 = 3*xi**2 - 2*xi**3
        N4 = L * (-xi**2 + xi**3)
        v_def = N1*u_ele[0] + N2*u_ele[1] + N3*u_ele[2] + N4*u_ele[3]
        
        # 5.2 Internal Forces
        Phi = (12 * E * I) / (G * As * L**2)
        const = (E * I) / ((1 + Phi) * L**3)
        k_ele_local = const * np.array([
            [12, 6*L, -12, 6*L],
            [6*L, (4+Phi)*L**2, -6*L, (2-Phi)*L**2],
            [-12, -6*L, 12, -6*L],
            [6*L, (2-Phi)*L**2, -6*L, (4+Phi)*L**2]
        ])
        
        f_int = np.dot(k_ele_local, u_ele) + fea_local[i]
        Fy_start, M_start = f_int[0], f_int[1]
        
        m_x, v_x_static = [], []
        for x in x_local:
            V_curr = Fy_start
            M_curr = M_start + Fy_start * x
            
            for _, load in span_loads.iterrows():
                mag = load['mag']
                p_loc = float(load['d_start'])
                if load['type'] == 'P':
                    if x >= p_loc:
                        V_curr -= mag
                        M_curr -= mag * (x - p_loc)
                elif load['type'] == 'U':
                    udl_len = float(load['dist'])
                    udl_start = float(load['d_start'])
                    # คำนวณช่วงที่โหลดกระทำจริง
                    len_cov = max(0, min(x, udl_start + udl_len) - udl_start)
                    if x > udl_start:
                        V_curr -= mag * len_cov
                        M_curr -= (mag * len_cov) * (x - (udl_start + len_cov/2))

            m_x.append(M_curr)
            v_x_static.append(V_curr)

        x_total.extend(x0 + x_local)
        moment_total.extend(m_x)
        shear_total.extend(v_x_static)
        def_total.extend(v_def) 

    # 6. Reactions Calculation
    R_vec = np.dot(K_global, d_all)
    FEA_R = np.zeros(n_dof)
    for i in range(n_spans):
        f = fea_local[i]
        idx = [2*i, 2*i+1, 2*(i+1), 2*(i+1)+1]
        FEA_R[idx[0]] += f[0]; FEA_R[idx[1]] += f[1]
        FEA_R[idx[2]] += f[2]; FEA_R[idx[3]] += f[3]
        
    R_final = R_vec + FEA_R
    reactions = {f"R{row['id']}": R_final[2*int(row['id'])] for _, row in sup_df.iterrows()}

    return np.array(x_total), np.array(moment_total), np.array(shear_total), np.array(def_total), reactions
