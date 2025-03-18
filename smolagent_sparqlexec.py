import time
import json
from SPARQLWrapper import SPARQLWrapper, JSON, POST
from smolagent_sparqlgen import SPARQLSelfCorrectionAgent



class SPARQLQueryExecutor:
    def __init__(self, endpoint_url: str):
        self.endpoint_url = endpoint_url
        self.sparql = SPARQLWrapper(endpoint_url)
        
    def execute_sparql_query(self, query: str) -> dict:
        """
        Executes a SPARQL query against a given SPARQL endpoint.
        This function leverages the SPARQLWrapper library to send the generated SPARQL query
        to the specified endpoint and retrieve results in JSON format.
        Args:
            query (str): The SPARQL query to execute.
            endpoint_url (str): The URL of the SPARQL endpoint.
        Returns:
            dict: A dictionary representing the JSON results returned by the SPARQL endpoint.
        """
        self.sparql.setQuery(query)
        self.sparql.setMethod(POST)
        self.sparql.setReturnFormat(JSON)
        try:
            start_time = time.time()
            results = self.sparql.query().convert()
            execution_time = time.time() - start_time
            print(f"SPARQL query: '{query}' executed in {execution_time:.3f} seconds.")
            return results, execution_time
        except Exception as e:
            raise Exception (f"Error executing SPARQL query: {e}")
        

# Example integration with the self-correction agent from previous code.
# (Assuming the SPARQLSelfCorrectionAgent and its custom tools have been defined as shown earlier.)

if __name__ == "__main__":
    # test 1
    # nl_query = "What data retention policies apply to Garmin Smartwatch 2?"
    # agent = SPARQLSelfCorrectionAgent(max_iterations=10)
    # result = agent.forward(nl_query) # Generate a valid SPARQL query from the natural language input.
    # final_query = result["final_query"]
    # iterations_used = result["iterations_used"]
    # print(f"Number of iterations used: {iterations_used}")
    
    # test 2
    # for testing
    nl_query = "Which personal data is collected lawfully by all devices under GDPR Article 5?"
    final_query = "PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>\nPREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\nPREFIX iot-reg: <http://knacc.umbc.edu/kelvinechenim/ontologies/iot-reg#>\nSELECT DISTINCT \n(SUBSTR(STR(?data), STRLEN(STR(iot-reg:)) + 1) AS ?dataValue)\n(SUBSTR(STR(?GdprControl), STRLEN(STR(iot-reg:)) + 1) AS ?GdprControlValue)\n?obligationLabel\n?articleLabel\nWHERE {\n  ?collection rdf:type iot-reg:DataCollection ;\n              iot-reg:collects ?data ;\n              iot-reg:hasRegulatoryControls ?GdprControl .\n  ?GdprControl iot-reg:hasObligation ?obligation .\n  ?obligation rdfs:label ?obligationLabel .\n  ?GdprControl rdfs:label ?articleLabel .\n  FILTER(CONTAINS(STR(?obligationLabel), \"lawfully, fairly, and transparently\") \n      || CONTAINS(STR(?obligationLabel), \"specific information when collecting\")\n      || CONTAINS(STR(?obligationLabel), \"lawful basis for processing\"))\n}"
    # nl_query2 = "Which personal data is shared with healthcare providers according to GDPR?"
    # final_query2 = "PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>\nPREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\nPREFIX iot-reg: <http://knacc.umbc.edu/kelvinechenim/ontologies/iot-reg#>\nSELECT DISTINCT \n(SUBSTR(STR(?data), STRLEN(STR(iot-reg:)) + 1) AS ?sharedData) \n(SUBSTR(STR(?GdprControl), STRLEN(STR(iot-reg:)) + 1) AS ?GdprControlValue) \n?obligationLabel\nWHERE {\n    ?dataSharing rdf:type iot-reg:DataSharing ;\n                    iot-reg:shares ?data ;\n                    iot-reg:hasRecipient iot-reg:HealthcareProvider .\n    ?dataSharing iot-reg:hasRegulatoryControls ?GdprControl .\n    ?GdprControl iot-reg:hasObligation ?obligation .\n    ?obligation rdfs:label ?obligationLabel .\n    FILTER(CONTAINS(STR(?obligationLabel), \"sharing of data\")\n            || CONTAINS(STR(?obligationLabel), \"based on consent\"))\n}"
    
    print("Final generated SPARQL query:")
    print(final_query)
    sparql_endpoint = "http://localhost:3030/Iot-Reg/sparql"
    try:
        query_executor = SPARQLQueryExecutor(sparql_endpoint)
        query_results, _ = query_executor.execute_sparql_query(final_query)
        print(query_results)
        print("Query results (JSON):")
        print(json.dumps(query_results, indent=2))
    except Exception as e:
        print(str(e))
