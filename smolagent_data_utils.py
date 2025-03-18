import json

def load_combined_data(source_path: str = "combined_factualplusnormative.json", nond_num=25, d_num=25):
    """
    Reads JSON data from source_path, splits into non-deontic (nond_num)
    and deontic (d_num), and returns a combined list.
    """
    with open(source_path, "r") as file:
        train_data = json.load(file)
    nondeontic = train_data[:nond_num]
    deontic = train_data[-d_num:]
    return nondeontic + deontic
