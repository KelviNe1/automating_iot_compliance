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
      - "description": the main query (first element)
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
        # Use regex to extract the main description (inside quotes after "Description:")
        desc_match = re.search(r'\(Description:\s*"(.*?)",', block, re.DOTALL)
        # Use regex to extract the SPARQL query (between the first pair of double quotes after "Query:")
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
      model (str): The model to use (default "o3-mini").
      max_tokens (int): Maximum tokens for the generated text.
      num_variations (int): Number of variations to generate.
    
    Returns:
      list: A list of generated query variations.
    """
    prompt = (
        f"You are a compliance expert. Generate {num_variations} additional natural language variations of the following query that have the same meaning:\n"
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
    
    # Use model_dump() to convert the response to a dictionary.
    response_dict = completion.model_dump()
    variations_text = response_dict["choices"][0]["text"].strip()
    variations = [line.strip() for line in variations_text.split("\n") if line.strip()]
    
    if len(variations) != num_variations:
        print(f"Warning: Expected {num_variations} variations, but received {len(variations)} for query: {main_query}")
    return variations
    
    # try:
    #     response = openai.ChatCompletion.create(
    #         model=model,
    #         messages=[
    #             {"role": "system", "content": "You are an expert AI assistant that generates variations of given compliance queries."},
    #             {"role": "user", "content": prompt}
    #         ],
    #         max_tokens=max_tokens,
    #         temperature=0.7
    #     )
    # except Exception as e:
    #     print("Error calling OpenAI API:", e)
    #     return []
    
    # # Split the response into individual lines (each variation)
    # variations_text = response.choices[0].text.strip()
    # variations = [line.strip() for line in variations_text.split("\n") if line.strip()]
    # # Log a warning if we don't get the expected number of variations
    # if len(variations) != num_variations:
    #     print(f"Warning: Expected {num_variations} variations, but received {len(variations)} for query: {main_query}")
    # return variations

def main():
    # Define input and output file names
    input_file = "/Users/Kelchee/Documents/Papers/P3/exp/KG/nondeontic_extra.txt"   # The .txt file with query pairs
    output_file = "/Users/Kelchee/Documents/Papers/P3/exp/KG/augmented_training_data.json"
    
    # Parse the .txt file to extract query pairs
    parsed_entries = parse_txt_file(input_file)
    
    augmented_entries = []

    # Process each parsed entry
    for idx, entry in enumerate(parsed_entries, start=1):
        print(f"Entry {idx+1}: {entry}")
        main_desc = entry["description"]
        # Use the main description to generate additional variations
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
    
    # Write the augmented entries to the output JSON file in pretty-printed format
    try:
        with open(output_file, "w", encoding="utf-8") as f_out:
            json.dump(augmented_entries, f_out, indent=2)
        print(f"Augmented training data successfully saved to '{output_file}' with {len(augmented_entries)} entries.")
    except Exception as e:
        print("Error saving augmented training data:", e)

if __name__ == "__main__":
    main()
