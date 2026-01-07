import os
import glob

# ค้นหาไฟล์ .py ทั้งหมดในโฟลเดอร์ปัจจุบัน
files = glob.glob("*.py")

print("🧹 Starting cleanup...")
for filepath in files:
    if filepath == "cleaner.py": continue # ข้ามไฟล์ตัวเอง
    
    try:
        # อ่านไฟล์
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # ตรวจสอบและแทนที่ \xa0 (non-breaking space) เป็น space ปกติ
        if '\xa0' in content:
            clean_content = content.replace('\xa0', ' ')
            
            # เขียนทับไฟล์เดิม
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(clean_content)
            print(f"✅ Fixed: {filepath}")
        else:
            print(f"👌 OK: {filepath}")
            
    except Exception as e:
        print(f"❌ Error reading {filepath}: {e}")

print("✨ Cleanup complete! Try running streamlit again.")
