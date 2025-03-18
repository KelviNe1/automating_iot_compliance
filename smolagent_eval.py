import csv
import logging

from smolagent_compliancesynthesis import EnhancedComplianceSynthesisModule
from smolagent_data_utils import load_combined_data

logging.basicConfig(level=logging.INFO)

class EvaluationTest:
    def __init__(self, nond_num: int = 25, d_num: int = 25):
        self.nond_num = nond_num
        self.d_num = d_num

    def compile_report_to_csv(self, data, dest: str, compliance_module):
        """
        data: the list returned by load_combined_data (non-deontic + deontic)
        dest: path to write CSV
        compliance_module: instance of EnhancedComplianceSynthesisModule
        """
        with open(dest, mode="w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["original_query", 
                             "sparql_query", 
                             "sparql_query_generation_time",
                             "raw_sparql_endpoint_result", 
                             "processed_kg_result", 
                             "sparql_execution_time (seconds)", 
                             "regulatory_context", 
                             "combined_KG+Reg_context", 
                             "num_candidates", 
                             "candidates_extract", 
                             "selected_response", 
                             "compliance_advice_generation_time (seconds)"])
            
            for idx, entry in enumerate(data):
                nl_query_desc = entry["descriptions"][0]
                compliance_report = compliance_module.synthesize_compliance_advice(nl_query_desc)
                writer.writerow(compliance_report.values())
                logging.info(f"Writing row {idx+1} into CSV file: \n{compliance_report}")
        logging.info("Compliance advice generation and CSV export completed.")

if __name__ == "__main__":
    source_path = "/Users/Kelchee/Documents/Papers/P3/complaince_app/data/training_data/old/combined_factualplusnormative.json"
    destination = "smol_compliance_advice_output.csv"
    endpoint_url = "http://localhost:3030/Iot-Reg/sparql"

    compliance_module = EnhancedComplianceSynthesisModule(endpoint_url=endpoint_url)

    test1 = EvaluationTest(nond_num=25, d_num=25) #evaluation test instance

    combined_data = load_combined_data(source_path, test1.nond_num, test1.d_num) # load data

    test1.compile_report_to_csv(data=combined_data, dest=destination, compliance_module=compliance_module)
