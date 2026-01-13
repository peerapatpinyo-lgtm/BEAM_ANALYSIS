# load_processor.py

def get_load_factors(code_std="ACI 318-19"):
    """
    คืนค่า Load Factors ตามมาตรฐานที่เลือก
    """
    if code_std == "ACI 318-19":
        return 1.2, 1.6  # 1.2DL + 1.6LL
    elif code_std == "EIT (Old)":
        return 1.4, 1.7  # 1.4DL + 1.7LL
    else:
        return 1.4, 1.7  # Default fallback

def calculate_factored_load(dl, ll, f_dl, f_ll):
    """
    คำนวณ Uniform Distributed Load แบบ Factored (Wu)
    """
    return (f_dl * dl) + (f_ll * ll)

def prepare_analysis_data(spans_length, dl, ll, f_dl, f_ll, E, I):
    """
    เตรียมข้อมูล Dictionary สำหรับส่งเข้า analysis engine (model_struct.py)
    
    Parameters:
    - spans_length: list ของความยาวคานแต่ละช่วง [4, 5, 4]
    - dl: Dead Load (kN/m)
    - ll: Live Load (kN/m)
    - f_dl: Factor Dead Load
    - f_ll: Factor Live Load
    - E: Young's Modulus
    - I: Moment of Inertia
    
    Returns:
    - List of dictionaries ที่มีข้อมูลครบถ้วนสำหรับแต่ละ Span
    """
    wu = calculate_factored_load(dl, ll, f_dl, f_ll)
    
    # สร้างโครงสร้างข้อมูลที่ model_struct.py ต้องการ
    # สมมติว่า model_struct ต้องการ list ของ dict ที่มี key: 'L', 'w', 'E', 'I'
    analysis_input = []
    
    for length in spans_length:
        span_data = {
            'L': float(length),
            'w': float(wu),      # Load ที่คูณ Factor แล้ว (kN/m)
            'dl': float(dl),     # เก็บค่า raw data ไว้เผื่อใช้เช็ค serviceability
            'll': float(ll),
            'E': float(E),
            'I': float(I)
        }
        analysis_input.append(span_data)
        
    return analysis_input, wu
