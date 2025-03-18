# KG result preprocessor

class SPARQLResultProcessor:
    def __init__(self, original_query: str, sparql_result: dict) -> str:
        self.original_query = original_query
        self.sparql_result = sparql_result
        
    def preprocess_sparql_results_dynamic(self) -> str:
        """
        Dynamically preprocesses SPARQL query results into a readable format.
        Returns:
            str: A flexible, human-readable response based on the SPARQL results.
        """
        bindings = self.sparql_result.get('results', {}).get('bindings', [])
        if not bindings:
            return self.handle_no_results_case()
        response = "SPARQL Query Results:\n"
        for binding in bindings:
            for key, value_info in binding.items():
                value = value_info.get('value', 'Unknown')
                response += f"- {key.replace('_', ' ')}: {value}\n"
            response += "\n"
        return response

    def handle_no_results_case(self) -> str:
        """
        Handles cases where no results are returned from the SPARQL query execution dynamically.
        Returns:
            str: A human-readable explanation based on the original query context.
        """
        # Check if the query seems negative (using basic regex for "without", "not", etc.)
        negative_words = ["without", "not", "missing", "absent", "no"]
        if any(word in self.original_query.lower() for word in negative_words):
            return f"Query: '{self.original_query}' resulted in no findings. This suggests no violations or missing data."
        else:
            # General response for affirmative or neutral queries
            return f"No relevant data found for the query: '{self.original_query}'. This may indicate a lack of relevant information in the knowledge graph."



# if __name__ == "__main__":
    # original_query_str = ""
    # example_sparql_result = {
    #     "results": {
    #         "bindings": []
    #     }
    # }

    # processor = SPARQLResultProcessor(original_query_str, example_sparql_result)
    # result_text = processor.preprocess_sparql_results_dynamic()
    # print(result_text)
