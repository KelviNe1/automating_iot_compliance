# compliance_synthesis.py

import json
import time
import logging
from typing import List, Dict
import numpy as np
from datetime import datetime
import re

# Import SentenceTransformer for semantic similarity (for post-retrieval reranking)
from sentence_transformers import SentenceTransformer, util

# Import our custom modules for SPARQL generation and execution.
from smolagent_sparqlgen import SPARQLSelfCorrectionAgent
from smolagent_sparqlexec import SPARQLQueryExecutor
from smolagent_kgpreprosessor import SPARQLResultProcessor

# Imports for query refinement using T5
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Imports for LlamaIndex regulatory context retrieval
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor

# Configure logging.
logging.basicConfig(level=logging.INFO)


# Pre-Retrieval Query Refinement for Regulatory Context

class RegulatoryQueryRefiner:
    """
    Refines a natural language query to improve regulatory context retrieval.
    
    Uses a pre-trained T5 model (e.g., 't5-base') to reengineer and expand the query,
    aiming to capture additional regulatory nuances for more effective context retrieval.
    """
    
    def __init__(self, model_name: str = "google/flan-t5-large", temperature: float = 0.8, top_k: int = 50):
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.temperature = temperature
        self.top_k = top_k
    
    def refine_query(self, query: str) -> str:
        try:
            with open("complaince_app/data/training_data/old/combined_factualplusnormative.json", "r") as file:
                train_data = json.load(file)
            for e in train_data:
                if query in e["descriptions"]:
                    refined_query = e["descriptions"]
                    break
            return refined_query
        except Exception as e:
            logging.warning(f"Unable to load source dataset due to {e}")
            return query
            
        
        example_instructions = """
        Example:
        Input: "list device compliance"
        Desired refined: "List IoT devices that comply with GDPR storage limitation article"

        Example:
        Input: "What data is being shared?"
        Desired refined: "Which personal data categories are shared with which third-party under GDPR obligations?"
        """
        input_text = f"{example_instructions}\nNow refine this query: {query}"

        inputs = self.tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
        outputs = self.model.generate(
            inputs,
            max_length=512,
            num_beams=5,
            early_stopping=True,
            do_sample=True,
            temperature=self.temperature,
            top_k=self.top_k
        )
        refined_query = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        logging.info("Original Query: %s", query)
        logging.info("Refined Query: %s", refined_query)
        return refined_query
    

def retrieve_regulatory_context(nl_query: str) -> str:
    """
    Retrieves regulatory document context from a corpus (e.g., PDF documents containing official GDPR publications)
    using LlamaIndex. Applies pre-retrieval query refinement for improved context retrieval.
    
    Args:
        nl_query (str): The original natural language query.
    
    Returns:
        str: A human-readable regulatory context extracted from the corpus.
    """
    # Instantiate the regulatory query refiner
    refiner = RegulatoryQueryRefiner(model_name="t5-base") #not in use
    refined_query = refiner.refine_query(nl_query)
    
    # Set up LlamaIndex settings for regulatory documents
    Settings.embed_model = HuggingFaceEmbedding(model_name="nlpaueb/legal-bert-base-uncased")
    Settings.llm = None
    Settings.chunk_size = 256
    Settings.chunk_overlap = 25
    
    # Load regulatory documents from "regulatory_data"
    documents = SimpleDirectoryReader("/Users/Kelchee/Documents/Papers/P3/complaince_app/data/retrieval_data").load_data()
    if not documents:
        logging.warning("No regulatory documents found in 'regulatory_data'.")
        return "No regulatory context available."
    
    index = VectorStoreIndex.from_documents(documents)
    top_k = 3
    retriever = VectorIndexRetriever(index=index, similarity_top_k=top_k)
    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.5)]
    )
    refined_query_concat = f"{','.join(refined_query[:-1])} or {refined_query[-1]}"
    response = query_engine.query(refined_query_concat)
    reg_context = "Regulatory Context:\n"
    for i in range(top_k):
        reg_context += response.source_nodes[i].text + "\n\n"
    logging.info("Retrieved\n%s", reg_context)
    return reg_context


# Enhanced Compliance Synthesis Module

class EnhancedComplianceSynthesisModule:
    """
    Synthesizes compliance advice by integrating:
      - SPARQL query generation via SPARQLSelfCorrectionAgent,
      - SPARQL query execution via SPARQLQueryExecutor,
      - KG result processing via SPARQLResultProcessor,
      - Regulatory context retrieval using refined queries.
      
    Then employs an enhanced agentic RAG approach with post-retrieval reranking.
    """
    def __init__(self, endpoint_url: str, num_candidates: int = 3, rerank_model: str = "all-MiniLM-L6-v2"):
        self.num_candidates = num_candidates
        self.sparql_agent = SPARQLSelfCorrectionAgent(max_iterations=10)
        self.sparql_executor = SPARQLQueryExecutor(endpoint_url)
        self.embedding_model = SentenceTransformer(rerank_model)
    
    def build_compliance_prompt(self,  nl_query: str, processed_kg_result: str, reg_context: str,) -> str:
        """
        Builds the compliance synthesis prompt using the combined context (KG + regulatory),
        the natural language query, and the SPARQL query results.
        """
        instructions = (
            "You are a regulatory compliance assistant for IoT manufacturers. "
            "Your task is to synthesize compliance advice based on the provided natural language query and the SPARQL query result. "
            "Ensure your response includes the following elements:\n"
            "1. Obligations: List any obligations that are met.\n"
            "2. Permissions: List any permissions granted.\n"
            "3. Prohibitions: Flag any violations.\n"
            "4. Final Compliance Verdict: Provide a final verdict: Compliant, Non-compliant, or Conditionally compliant.\n"
            "Output only the compliance advice."
        )
        prompt = (
            f"<s>[INST] \nQuestion:\n{nl_query}\n\n\nRetrieved {reg_context}\n\n"
            f"SPARQL Query 'IoT-Reg' Knowledge Graph Result:\n{processed_kg_result}\n\n{instructions}\n[/INST]\n"
        )
        return prompt

    def generate_candidates(self, prompt: str) -> List[str]:
        """
        Generates multiple candidate compliance advice responses by invoking the fine-tuned LLM repeatedly.
        """
        model_directory = "mlx-community/Mistral-7B-Instruct-v0.2-4bit"
        adapter_file = "adapters.npz"
        max_tokens = 600
        
        candidates = []
        for i in range(self.num_candidates):
            command = [
                "python", "lora.py",
                "--model", model_directory,
                "--adapter-file", adapter_file,
                "--max-tokens", str(max_tokens),
                "--prompt", prompt
            ]
            candidate_output = run_command_with_live_output(command)
            candidate = candidate_output.strip()
            candidates.append(candidate)
            logging.info("Generated candidate %d\n\n\n\n", i + 1)
        return candidates

    def rerank_candidates(self, candidates: List[str], prompt: str) -> str:
        """
        Reranks candidate responses based on semantic similarity with the combined context.
        
        Returns the candidate response with the highest cosine similarity.
        """
        #extract main text from candidates using regex
        candidates_extract = []
        # for candidate in candidates:
        #     trimmed = re.sub(r"^.*?\[INST\]", "", candidate, flags=re.DOTALL)
        #     trimmed = re.sub(r"You are a regulatory compliance assistant.*$", "", trimmed, flags=re.DOTALL)
        #     candidates_extract.append(trimmed.strip())
        for candidate in candidates:
            # match = re.search(
            #     # r"\[INST\](.*?)You are a regulatory compliance assistant", candidate, flags=re.DOTALL
            #     r"(Loading.*?\[INST\].*?\[INST\])", candidate, flags=re.DOTALL
                
            # )
            # if match:
            #     extracted = match.group(1).strip()
            #     candidates_extract.append(extracted)
            # else:
            #     extracted = candidate # Fallback if not found
            #     candidates_extract.append(extracted)
            parts = re.split(r"\[/INST]", candidate, maxsplit=2, flags=re.DOTALL)
            result = (parts[1]).rstrip("=")
            candidates_extract.append(result)
            
        logging.info(f"Extracted Candidates: {candidates_extract}")
        cosine_scores = []
        context_embedding = self.embedding_model.encode(prompt, convert_to_tensor=True)
        for i in range(self.num_candidates):
            candidate_embeddings = self.embedding_model.encode(candidates_extract[i], convert_to_tensor=True)
            cosine_score = util.cos_sim(candidate_embeddings, context_embedding)
            cosine_score_value = cosine_score.item()
            cosine_scores.append(cosine_score_value)
            logging.info(f"Cosine Score for Candidate {i+1}: {cosine_score_value}")
        best_idx = int(np.argmax(cosine_scores)) # 'list' object has no attribute 'cpu'
        logging.info("Selected candidate %d with cosine score %.4f", best_idx + 1, cosine_scores[best_idx])
        return candidates_extract[best_idx], candidates_extract
    

    def synthesize_compliance_advice(self, nl_query: str) -> Dict:
        """
        Synthesizes compliance advice by integrating:
          1. SPARQL query generation (self-correction agent).
          2. SPARQL query execution (query executor).
          3. KG result processing (result processor).
          4. Regulatory context retrieval (with refined query).
          5. Combination of contexts and prompt construction.
          6. Generation of multiple candidate compliance advice responses.
          7. Reranking of candidates to select the best output.
          8. Capture of evaluation metrics.
        
        Returns a report with the selected compliance advice and metrics.
        """
        start_time = time.time()
        
        # Step 1: Generate SPARQL query.
        sparql_query = self.sparql_agent.forward(nl_query)["final_query"]
        sparql_query_gen_time = self.sparql_agent.forward(nl_query)["gen_time_taken"]
        logging.info("Generated SPARQL Query:\n%s", sparql_query)
        
        # Step 2: Execute the SPARQL query.
        sparql_result_dict, exec_time = self.sparql_executor.execute_sparql_query(sparql_query)
        sparql_result_text = json.dumps(sparql_result_dict, indent=2)
        logging.info("SPARQL Query Result:\n%s", sparql_result_text)
        
        # Step 3: Process the SPARQL result into human-readable KG context.
        processor = SPARQLResultProcessor(nl_query, sparql_result_dict)
        processed_kg_result = processor.preprocess_sparql_results_dynamic()
        logging.info("Processed KG Context:\n%s", processed_kg_result)
        
        # Step 4: Retrieve regulatory document context with refined query.
        reg_context = retrieve_regulatory_context(nl_query)
        logging.info("Regulatory Context:\n%s", reg_context)
        
        # Combine the KG context and regulatory context.
        combined_context = processed_kg_result + "\n\n" + reg_context # necessary?
        logging.info("Combined KG+Reg Context:\n%s", combined_context)
        
        # Step 5: Build compliance synthesis prompt.
        prompt = self.build_compliance_prompt(nl_query, processed_kg_result, reg_context)
        logging.info("Compliance Synthesis Prompt:\n%s", prompt)
        
        # Step 6: Generate candidate compliance advice responses.
        candidates = self.generate_candidates(prompt)
        
        # Step 7: Rerank candidates based on semantic similarity with the combined context.
        selected_response, candidates_extract = self.rerank_candidates(candidates, combined_context)
        
        total_time = time.time() - start_time
        
        report = {
            "original_query": nl_query,
            "sparql_query": sparql_query,
            "sparql_query_generation_time": sparql_query_gen_time,
            "raw_sparql_endpoint_result": sparql_result_text,
            "processed_kg_result": processed_kg_result,
            "sparql_execution_time (seconds)": float(f"{exec_time:.4f}"),
            "regulatory_context": reg_context,
            "combined_KG+Reg_context": combined_context,
            "num_candidates": self.num_candidates,
            "candidates_extract": candidates_extract,
            "selected_response": selected_response,
            "compliance_advice_generation_time (seconds)": float(f"{total_time:.4f}")
        }
        return report

def run_command_with_live_output(command: List[str]) -> str:
    """
    Runs a command and returns its full output as a string.
    """
    import subprocess
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    output_lines = []
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            logging.info(output.strip())
            output_lines.append(output.strip())
    err_output = process.stderr.read()
    if err_output:
        logging.error(err_output)
    return "\n".join(output_lines)


# Example Usage
if __name__ == "__main__":
    sample_nl_query = "Show me the devices owned by users who have consented to data sharing."
    endpoint_url = "http://localhost:3030/Iot-Reg/sparql"  
    
    compliance_module = EnhancedComplianceSynthesisModule(endpoint_url=endpoint_url, num_candidates=3)
    compliance_report = compliance_module.synthesize_compliance_advice(sample_nl_query)
    
    print("Enhanced Compliance Synthesis Report:")
    print(json.dumps(compliance_report, indent=2))
