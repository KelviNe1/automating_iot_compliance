import re
import json
import os
from openai import OpenAI
import datetime

from dotenv import load_dotenv
load_dotenv()

client = OpenAI(
  api_key=os.environ['OPENAI_API_KEY'], 
)

def parse_txt_file(file_path):
    """
    Parses the input .txt file containing query pairs.
    
    Expected format for each query pair is:
    (Description: "Main description",
    Query: 
    "
    ... SPARQL query ...
    ")
    ---
    
    Returns a list of dictionaries with keys:
      - "Description": the main query (first element)
      - "query": the associated SPARQL query
    """
    
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    # Split the content into blocks separated by lines of dashes
    blocks = re.split(r'\n\s*---+\n', content)
    entries = []
    for block in blocks:
        if not block.strip():
            continue
        desc_match = re.search(r'\(Description:\s*"(.*?)",', block, re.DOTALL)
        query_match = re.search(r'Query:\s*"\n([\s\S]*?)\n"', blocks[0], re.DOTALL)
        if desc_match and query_match:
            description = desc_match.group(1).strip()
            query = query_match.group(1).strip()
            entries.append({
                "description": description,
                "query": query
            })
    return entries

def augment_query_variations(main_query, model="gpt-4o-mini", max_tokens=100, num_variations=4):
    """
    Uses OpenAI's gpt-4o-mini model to generate additional natural language variations
    for the given main query.
    
    Parameters:
      main_query (str): The original query description.
      model (str): The model to use.
      max_tokens (int): Maximum tokens for the generated text.
      num_variations (int): Number of variations to generate.
    
    Returns:
      list: A list of generated query variations.
    """
    prompt = (
        f"You are an IoT data privacy compliance expert. Generate {num_variations} additional natural language variations of the following query that have the same meaning:\n"
        f"\"{main_query}\"\n"
        "Output each variation on a new line."
    )
    
    try:
        # Create a completion using the new client interface.
        completion = client.completions.create(
            model=model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=0.7,
            n=1,  # one completion containing all variations separated by newlines
        )
    except Exception as e:
        print("Error calling OpenAI API:", e)
        return []
    
    response_dict = completion.model_dump() #convert the response to a dictionary
    variations_text = response_dict["choices"][0]["text"].strip()
    variations = [line.strip() for line in variations_text.split("\n") if line.strip()]
    
    if len(variations) != num_variations:
        print(f"Warning: Expected {num_variations} variations, but received {len(variations)} for query: {main_query}")
    return variations

def main():
    input_file = "nondeontic_extra.txt"   # The .txt file with query pairs
    output_file = "augmented_training_data.json"
    
    parsed_entries = parse_txt_file(input_file) # Parse the .txt file to extract query pairs
    
    augmented_entries = []

    # Process each parsed entry
    for idx, entry in enumerate(parsed_entries, start=1):
        print(f"Entry {idx+1}: {entry}")
        main_desc = entry["description"]
        additional_variations = augment_query_variations(main_desc)
        # Combine the main description with the new variations (total 1+4 = 5 variations)
        descriptions = [main_desc] + additional_variations
        augmented_entry = {
            "id": f"N{idx:03d}",
            "descriptions": descriptions,
            "query": entry["query"],
            "ingestion_timestamp": datetime.datetime.utcnow().isoformat() + "Z"
        }
        print(f"Augmented_Entry {idx+1}: {augmented_entry}")
        augmented_entries.append(augmented_entry)
    

    try:
        with open(output_file, "w", encoding="utf-8") as f_out:
            json.dump(augmented_entries, f_out, indent=2)
        print(f"Augmented training data successfully saved to '{output_file}' with {len(augmented_entries)} entries.")
    except Exception as e:
        print("Error saving augmented training data:", e)

if __name__ == "__main__":
    main()
