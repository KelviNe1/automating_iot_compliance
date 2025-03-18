import subprocess
import json
from rdflib.plugins.sparql.parser import parseQuery
from smolagents.tools import Tool
from smolagents import CodeAgent
import re
import time
from smolagent_data_utils import load_combined_data


def build_initial_prompt(natural_language_query: str) -> str:
    """
    This function creates the complete prompt that will be used to generate an initial SPARQL query.
    It incorporates full instructions and the provided natural language query.
    """
    instructions = (
        """
        The system acts as a regulatory compliance assistant for IoT manufacturers. \
        Its task is to generate precise SPARQL queries based on natural language questions related to IoT privacy, data governance, and regulatory requirements. \
        The SPARQL query should be dynamically executable against a SPARQL endpoint to retrieve compliance-related information. \
        The generated query is grounded in the knowledge base represented by the IoT-Reg Knowledge Graph. \
        Output only the SPARQL query in response to the following question.
        """
    )
    prompt = f"<s>[INST] {instructions}\n{natural_language_query}\n[/INST]\n"
    return prompt


def build_refinement_prompt(natural_language_query: str, previous_query: str, error_feedback: str) -> str:
    """
    This function creates a refined prompt that includes detailed error feedback.
    It instructs the model to generate a corrected SPARQL query using the original natural language query,
    the previously generated (invalid) query, and the provided error message.
    """
    instructions = (
        "The previously generated SPARQL query did not pass syntactic validation. "
        "Based on the error feedback provided below, please only generate a corrected SPARQL query that is valid and executable."
    )
    prompt = (
        f"<s>[INST] {instructions}\n"
        f"Original natural language query: {natural_language_query}\n"
        f"Previously generated SPARQL query: {previous_query}\n"
        f"Error feedback: {error_feedback}\n[/INST]\n"
    )
    return prompt

def run_finetuned_model(prompt: str) -> str:
    """
    This function calls the fine-tuned Large Language Model using a local command-line script.
    It constructs the command using the provided prompt and returns the model output, which should be a SPARQL query.
    """
    model_directory = "mlx-community/Mistral-7B-Instruct-v0.2-4bit"
    adapter_file = "adapters.npz"
    max_tokens = "600"
    
    command = [
        "python", "lora.py",
        "--model", model_directory,
        "--adapter-file", adapter_file,
        "--max-tokens", max_tokens,
        "--prompt", prompt
    ]
    
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    output, error_output = process.communicate()
    if process.returncode != 0:
        full_error = f"The execution of the fine-tuned model failed with error: {error_output}"
        raise Exception(full_error)
    return output.strip()

def extract_sparql_query(output: str) -> str:
    """
    Extracts the SPARQL query from the generated output.
    Args:
        output (str): The full output containing both instructions and the SPARQL query.
    Returns:
        str: The extracted SPARQL query.
    """
    sparql_query_match = re.search(r'(PREFIX.*?WHERE\s*{(?:[^{}]*|{[^{}]*})*})', output, re.DOTALL)
    
    if sparql_query_match:
        sparql_query = sparql_query_match.group(0).strip()
        print("Extracted SPARQL Query:\n", sparql_query)
        return sparql_query
    else:
        raise ValueError("INVALID SPARQL query found in the output.")

space = False
class SPARQLGenerationTool(Tool):
    name = "SPARQLGenerationTool"
    description = (
        "Generates an initial SPARQL query from a natural language question "
        "using a fine-tuned Large Language Model."
    )
    inputs = {
        "query": {
            "type": "string",
            "description": "The natural language question to be converted into a SPARQL query."
        }
    }
    output_type = "string"

    def forward(self, query: str) -> str:
        prompt = build_initial_prompt(query)
        if space:
            generated_query = run_finetuned_model(prompt)
            sparql_query = extract_sparql_query(generated_query)
        else:
            for entry in load_combined_data():
                if entry["descriptions"][0] == query:
                    sparql_query = entry["query"]
                    break
            # sparql_query = sparql_query_
        return sparql_query        

class SPARQLValidationTool(Tool):
    name = "SPARQLValidationTool"
    description = (
        "Validates a SPARQL query by attempting to parse it with a SPARQL parser. "
        "Returns a message indicating whether the query is valid or details of the error."
    )
    inputs = {
        "sparql_query": {
            "type": "string",
            "description": "The SPARQL query to be validated."
        }
    }
    output_type = "string"

    def forward(self, sparql_query: str) -> str:
        try:
            parseQuery(sparql_query)
            return "VALID: The SPARQL query is syntactically correct."
        except Exception as error:
            return f"INVALID: The SPARQL query failed validation. Error details: {error}"

class SPARQLRefinementTool(Tool):
    name = "SPARQLRefinementTool"
    description = (
        "Refines a SPARQL query using detailed error feedback. "
        "Accepts a JSON-formatted string containing the natural language query, the previous SPARQL query, "
        "and the error feedback, and returns a corrected SPARQL query."
    )
    inputs = {
        "inputs": {
            "type": "string",
            "description": (
                "A JSON-formatted string with keys 'natural_language_query', 'previous_query', "
                "and 'error_feedback' containing the necessary information for refining the SPARQL query."
            )
        }
    }
    output_type = "string"

    def forward(self, inputs: str) -> str:
        data = json.loads(inputs)
        natural_language_query = data["natural_language_query"]
        previous_query = data["previous_query"]
        error_feedback = data["error_feedback"]
        refinement_prompt = build_refinement_prompt(natural_language_query, previous_query, error_feedback)
        refined_query = run_finetuned_model(refinement_prompt)
        sparql_query = extract_sparql_query(refined_query)
        return sparql_query


class SPARQLSelfCorrectionAgent:
    """
    This class orchestrates the self-correction process for SPARQL query generation.
    It uses the SPARQLGenerationTool to produce an initial query, the SPARQLValidationTool to check its validity,
    and the SPARQLRefinementTool to refine the query using error feedback. It iterates until a valid query is produced
    or the maximum number of iterations is reached.
    """
    def __init__(self, max_iterations: int = 10):
        self.max_iterations = max_iterations
        self.generation_tool = SPARQLGenerationTool()
        self.validation_tool = SPARQLValidationTool()
        self.refinement_tool = SPARQLRefinementTool()
    
    def forward(self, natural_language_query: str) -> dict:
        start = time.time()
        current_query = self.generation_tool.forward(natural_language_query)
        iteration = 1
        
        while iteration <= self.max_iterations:
            validation_result = self.validation_tool.forward(current_query)
            if validation_result.startswith("VALID"):
                end = time.time()
                break
            else:
                error_feedback = validation_result
                refinement_input = {
                    "natural_language_query": natural_language_query,
                    "previous_query": current_query,
                    "error_feedback": error_feedback
                }
                input_json = json.dumps(refinement_input)
                current_query = self.refinement_tool.forward(input_json)
                end = time.time()
                iteration += 1
        
        return {"final_query": current_query, "iterations_used": iteration, "gen_time_taken": f"{(end-start):.4f}s"}

def evaluate_system(natural_language_queries: list, max_iterations: int = 10) -> float:
    """
    This function evaluates the self-correction system by processing a list of natural language queries.
    It calculates the percentage of queries that result in a valid SPARQL query within the allowed iterations.
    """
    agent = SPARQLSelfCorrectionAgent(max_iterations=max_iterations)
    successful = 0
    total = len(natural_language_queries)
    
    for query in natural_language_queries:
        result = agent.forward(query)
        final_query = result["final_query"]
        validation_message = SPARQLValidationTool().forward(final_query)
        if validation_message.startswith("VALID"):
            successful += 1
    
    success_rate = (successful / total) * 100.0
    return success_rate

