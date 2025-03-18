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
        
