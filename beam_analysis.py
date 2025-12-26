import numpy as np
import pandas as pd

def run_beam_analysis(spans, supports, loads):
    """
    Direct Stiffness Method (Robust Version).
    Fixes: 'string indices must be integers' definitively by forcing data conversion first.
    """

    # ==============================================================================
    # 1. THE FIREWALL: SANITIZE ALL INPUTS (Absolute Safety Layer)
    # ==============================================================================
    
    # --- 1.1 Convert LOADS to List of Dictionaries ---
    # ไม่ว่าจะมาเป็น DataFrame, List หรือ None แปลงเป็น List of Dicts ทั้งหมด
    safe_loads = []
    if isinstance(loads, pd.DataFrame):
        # .to_dict('records') คือคำสั่งศักดิ์สิทธิ์ที่แปลงตารางเป็น List of Dicts
        safe_loads = loads.to_dict('records') 
    elif isinstance(loads, list):
        safe_loads = loads
    
    # --- 1.2 Convert SUPPORTS to List of Strings ---
    safe_supports = []
    n_nodes = len(spans) + 1
    
    for i in range(n_nodes):
        raw_val = None
        # ดึงค่าแบบปลอดภัยตาม Data Type
        if isinstance(supports, list):
            raw_val = supports[i] if i < len(supports) else None
        elif isinstance(supports, (pd.DataFrame, pd.Series)):
            try: raw_val = supports.iloc[i]
            except: raw_val = None
        
        # แกะค่า String ออกมา
        sup_str = "None"
        if isinstance(raw_val, str):
            sup_str = raw_val
        elif isinstance(raw_val, dict):
            sup_str = raw_val.get('type', 'None')
        elif hasattr(raw_val, 'get'): # Pandas Series/Object
            sup_str = raw_val.get('type', 'None')
        elif hasattr(raw_val, 'type'):
            sup_str = raw_val.type
            
        safe_supports.append(sup_str)

    # ==============================================================================
    # 2. CALCULATION CORE (Uses ONLY safe_loads and safe_supports)
    # ==============================================================================
    
    n_spans = len(spans)
    dof_per_node = 2
    total_dof = n_nodes * dof_per_node
    
    K_global = np.zeros((total_dof, total_dof))
    F_global = np.zeros(total_dof)
    E = 2e6  
    I = 0.001 

    for i in range(n_spans):
        L = spans[i]
        L2 = L*L; L3 = L2*L
        
        # Local Stiffness
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
        
        # Filter loads for this span from safe_loads
        for load in safe_loads:
            # ใช้ .get() และ float() ดักทุกจุด เพื่อป้องกันข้อมูลขยะ
            try:
                s_idx = int(load.get('span_idx', -999))
            except:
                continue

            if s_idx != i:
                continue

            l_type = load.get('type', 'None')
            try: w = -float(load.get('w', 0.0))
            except: w = 0.0
            try: P = -float(load.get('P', 0.0))
            except: P = 0.0
            try: x_loc = float(load.get('x', 0.0))
            except: x_loc = 0.0

            if l_type == 'U':
                fem_vec[0] += w * L / 2
                fem_vec[1] += w * L2 / 12
                fem_vec[2] += w * L / 2
                fem_vec[3] += -w * L2 / 12
            elif l_type == 'P':
                a = x_loc
                b = L - a
                if 0 <= a <= L:
                    fem_vec[0] += P * (b**2 * (3*a + b)) / L3
                    fem_vec[1] += P * a * (b**2) / L2
                    fem_vec[2] += P * (a**2 * (a + 3*b)) / L3
                    fem_vec[3] += -P * (a**2) * b / L2

        for r in range(4):
            F_global[idx[r]] -= fem_vec[r]

    # --- Boundary Conditions ---
    free_dof = np.ones(total_dof, dtype=bool)
    vis_supports_data = []

    for i in range(n_nodes):
        stype = safe_supports[i] # ใช้ตัวที่ Clean แล้วเท่านั้น
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

    # --- Solver ---
    K_reduced = K_global[np.ix_(free_dof, free_dof)]
    F_reduced = F_global[free_dof]
    
    try:
        D_reduced = np.linalg.solve(K_reduced, F_reduced)
    except np.linalg.LinAlgError:
        D_reduced = np.zeros_like(F_reduced)

    D_full = np.zeros(total_dof)
    D_full[free_dof] = D_reduced

    # --- Post-Processing ---
    x_coords = []
    shear_vals = []
    moment_vals = []
    span_ids = []
    global_x = 0.0
    
    for i in range(n_spans):
        L = spans[i]
        idx = [2*i, 2*i+1, 2*(i+1), 2*(i+1)+1]
        u = D_full[idx]
        
        L2 = L*L; L3 = L2*L
        k_local = (E * I / L3) * np.array([
            [12,      6*L,    -12,     6*L],
            [6*L,     4*L2,   -6*L,    2*L2],
            [-12,    -6*L,     12,    -6*L],
            [6*L,     2*L2,   -6*L,    4*L2]
        ])
        
        # FEM Recalc for internal forces
        fem_vec = np.zeros(4)
        current_loads = []
        
        # Filter again from safe_loads
        for load in safe_loads:
            try: s_idx = int(load.get('span_idx', -999))
            except: continue
            if s_idx == i:
                current_loads.append(load)
                
        for load in current_loads:
            l_type = load.get('type')
            try: w = -float(load.get('w', 0.0))
            except: w = 0.0
            try: P = -float(load.get('P', 0.0))
            except: P = 0.0
            try: x_loc = float(load.get('x', 0.0))
            except: x_loc = 0.0

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
        
        # Generate Plot Points
        x_span = np.linspace(0, L, 100)
        for load in current_loads:
            if load.get('type') == 'P':
                try: lx = float(load.get('x', 0.0))
                except: lx = 0.0
                x_span = np.append(x_span, [lx - 1e-5, lx + 1e-5])
        
        x_span = np.sort(np.unique(x_span))
        x_span = x_span[(x_span >= 0) & (x_span <= L)]

        for x in x_span:
            V_x = V_start
            M_x = M_start + V_start * x
            
            for load in current_loads:
                l_type = load.get('type')
                try: w_mag = float(load.get('w', 0.0))
                except: w_mag = 0.0
                try: P_mag = float(load.get('P', 0.0))
                except: P_mag = 0.0
                try: px = float(load.get('x', 0.0))
                except: px = 0.0

                if l_type == 'U':
                    V_x -= w_mag * x
                    M_x -= w_mag * x**2 / 2
                elif l_type == 'P':
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
