import json

def export_data(params, spans, sup_df, load_list):
    data = {
        "params": params,
        "spans": spans,
        "supports": sup_df.to_dict('records') if not sup_df.empty else [],
        "loads": load_list
    }
    return json.dumps(data, indent=4)

def load_data(uploaded_file):
    try:
        return json.load(uploaded_file)
    except: return None
