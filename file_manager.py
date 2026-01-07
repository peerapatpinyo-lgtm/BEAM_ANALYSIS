import json
import pandas as pd

def export_data(params, spans, sup_df, loads_df):
    """Convert Project Data to JSON String"""
    data = {
        "params": params,
        "spans": spans,
        "supports": sup_df.to_dict('records') if not sup_df.empty else [],
        "loads": loads_df.to_dict('records') if (loads_df is not None and not loads_df.empty) else []
    }
    return json.dumps(data, indent=4)

def load_data(uploaded_file):
    """Load JSON file to Session State"""
    try:
        data = json.load(uploaded_file)
        return data
    except Exception as e:
        return None
