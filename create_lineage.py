import argparse
from Bio import Entrez
import pandas as pd
import csv
from collections import defaultdict

def extract_species_from_csv(csv_file_path):
    """
    Extract species names from CSV file and format them into a clean list.
    """
    bacteria_list = []
    
    with open(csv_file_path, 'r', newline='', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        
        for row in csv_reader:
            for cell in row:
                # Check if the cell contains species information (s__)
                if 's__' in cell:
                    # Split by periods to get individual taxonomic levels
                    parts = cell.split('.')
                    
                    # Find the species part (starts with s__)
                    for part in parts:
                        if part.startswith('s__'):
                            # Remove the 's__' prefix
                            species_name = part[3:]
                            # Replace underscores with spaces
                            species_name = species_name.replace('_', ' ')
                            bacteria_list.append(species_name)
                            break
    
    return bacteria_list

def get_lineage_with_ranks_from_entrez(species_name):
    """
    Returns a dictionary with rank as key and name as value.
    Special handling for "no rank" entries like "cellular organisms".
    """
    try:
        # Search for the species name in NCBI Taxonomy database
        Entrez.email = "noga.zahav@gmail.com"  # Replace with your email
        search_handle = Entrez.esearch(db="taxonomy", term=species_name)
        search_results = Entrez.read(search_handle)
        search_handle.close()

        # Check if any result is found
        if not search_results["IdList"]:
            return {"error": f"Species '{species_name}' not found in NCBI"}

        # Fetch detailed information for the first matching result
        taxid = search_results["IdList"][0]
        fetch_handle = Entrez.efetch(db="taxonomy", id=taxid, retmode="xml")
        fetch_results = Entrez.read(fetch_handle)
        fetch_handle.close()

        # Extract the lineage with ranks
        lineage_data = {}
        lineage = fetch_results[0]["LineageEx"]
        
        for lineage_info in lineage:
            rank = lineage_info["Rank"]
            name = lineage_info["ScientificName"]
            
            # Handle special "no rank" cases
            if rank == "no rank":
                if name == "cellular organisms":
                    rank = "cellular_organisms"
                elif "root" in name.lower():
                    rank = "root"
                else:
                    # For other "no rank" entries, use the name as identifier
                    rank = f"no_rank_{name.replace(' ', '_').lower()}"
            
            lineage_data[rank] = name
        
        # Add the species itself (it's not always in LineageEx)
        species_info = fetch_results[0]
        lineage_data[species_info["Rank"]] = species_info["ScientificName"]
        
        return lineage_data
        
    except Exception as e:
        return {"error": f"An error occurred while processing '{species_name}': {str(e)}"}

def create_formatted_output(df, all_ranks):
    """
    Create a formatted output with specific column structure:
    Original Name, Bacteria Name (species), Kingdom, Phylum, Class, Order, Family, Genus, Other
    """
    formatted_data = []
    
    # Define the target taxonomic levels we want in the formatted output
    target_ranks = ['kingdom', 'phylum', 'class', 'order', 'family', 'genus']
    
    for index, row in df.iterrows():
        # Original species name from input (first column)
        original_species_name = row['Original Species'] 
        
        # Extract genus from the original species name for the original name format
        if ' ' in original_species_name:
            genus_name = original_species_name.split()[0]
        else:
            genus_name = original_species_name
        
        # Get NCBI species name (from the species rank column)
        ncbi_species_name = ""
        for col in df.columns:
            if col.lower() == 'species':
                if not pd.isna(row[col]) and row[col] != '':
                    ncbi_species_name = row[col]
                break
        
        # Use NCBI species name if available, otherwise fall back to original
        if ncbi_species_name:
            bacteria_name = ncbi_species_name.replace(' ', '_')
        else:
            bacteria_name = original_species_name.replace(' ', '_')
        
        # Create original name using the original input species name
        original_name_species = original_species_name.replace(' ', '_')
        original_name = f"g__{genus_name}.s__{original_name_species}"
        
        # Create the formatted row
        formatted_row = [original_name, bacteria_name]
        
        # Add the target taxonomic ranks
        for rank in target_ranks:
            # Find the corresponding column in the original dataframe
            rank_column = None
            for col in df.columns:
                if col.lower() == rank:
                    rank_column = col
                    break
            
            if rank_column and not pd.isna(row[rank_column]) and row[rank_column] != '':
                formatted_row.append(row[rank_column])
            else:
                formatted_row.append('')  # Empty if rank not found
        
        # Add "Other" column (empty for now, can be populated later if needed)
        formatted_row.append('')
        
        formatted_data.append(formatted_row)
    
    # Create headers for the formatted output
    formatted_headers = [
        'Original Name', 
        'Bacteria Name (species)', 
        'Kingdom', 
        'Phylum', 
        'Class', 
        'Order', 
        'Family', 
        'Genus', 
        'Other'
    ]
    
    # Create DataFrame
    formatted_df = pd.DataFrame(formatted_data, columns=formatted_headers)
    
    return formatted_df

def get_all_unique_ranks(all_lineage_data):
    """
    Get all unique taxonomic ranks from all species data.
    """
    all_ranks = set()
    for lineage_data in all_lineage_data:
        if "error" not in lineage_data:
            all_ranks.update(lineage_data.keys())
    
    # Define a preferred order for common taxonomic ranks
    rank_order = [
        "root", "cellular_organisms",  # High-level "no rank" classifications
        "superkingdom", "kingdom", "subkingdom", "superphylum", "phylum", 
        "subphylum", "superclass", "class", "subclass", "infraclass",
        "superorder", "order", "suborder", "infraorder", "parvorder",
        "superfamily", "family", "subfamily", "tribe", "subtribe",
        "genus", "subgenus", "species group", "species subgroup", "species"
    ]
    
    # Sort ranks according to the preferred order, with unknown ranks at the end
    sorted_ranks = []
    for rank in rank_order:
        if rank in all_ranks:
            sorted_ranks.append(rank)
            all_ranks.remove(rank)
    
    # Add any remaining ranks that weren't in our predefined order
    sorted_ranks.extend(sorted(all_ranks))
    
    return sorted_ranks

# Main execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process species data and taxonomic ranks.")
    parser.add_argument(
        '--input_csv', 
        type=str, 
        required=True, 
        help='Path to the CSV file containing the test bacteria names'
    )
    parser.add_argument(
        '--output_csv', 
        type=str, 
        required=True, 
        help='Path to save the formatted CSV output containing the linage of each bacteria'
    )
    args = parser.parse_args()

    # Extract species from CSV file
    csv_file_path = args.input_csv  # Get the input CSV path from arguments

    # # Extract species from CSV file
    # csv_file_path = 'test_bacteria.csv'  

    try:
        bacteria_list = extract_species_from_csv(csv_file_path)
        print(f"Extracted {len(bacteria_list)} species from CSV file")
        
        # Print the extracted species list
        print("\nExtracted bacteria_list:")
        print("bacteria_list = [")
        for i, bacteria in enumerate(bacteria_list):
            if i == len(bacteria_list) - 1:
                print(f'    "{bacteria}"')
            else:
                print(f'    "{bacteria}",')
        print("]\n")
        
    except FileNotFoundError:
        print(f"Error: File '{csv_file_path}' not found.")
        print("Using empty bacteria_list...")
        bacteria_list = []
    except Exception as e:
        print(f"Error processing file: {e}")
        print("Using empty bacteria_list...")
        bacteria_list = []

    # Get lineage data for all species
    all_lineage_data = []
    for species in bacteria_list:
        print(f"Processing: {species}")
        lineage_data = get_lineage_with_ranks_from_entrez(species)
        all_lineage_data.append(lineage_data)

    # Get all unique taxonomic ranks
    all_ranks = get_all_unique_ranks(all_lineage_data)
    print(f"\nFound the following taxonomic ranks: {all_ranks}")

    # Prepare data for DataFrame
    data = []
    for i, species in enumerate(bacteria_list):
        lineage_data = all_lineage_data[i]
        
        if "error" in lineage_data:
            print(f"Error for {species}: {lineage_data['error']}")
            # Create a row with species name and empty values for all ranks
            row = [species] + [""] * len(all_ranks)
        else:
            # Create a row with species name and taxonomic information
            row = [species]
            for rank in all_ranks:
                row.append(lineage_data.get(rank, ""))
        
        data.append(row)

    # Create headers: Original + all taxonomic ranks
    headers = ["Original Species"] + [rank.capitalize() for rank in all_ranks]

    # Create DataFrame
    df = pd.DataFrame(data, columns=headers)

    # Create formatted output file
    formatted_df = create_formatted_output(df, all_ranks)
    formatted_csv_path = args.output_csv  # Get the output path from arguments
    # formatted_csv_path = "bacterial_lineage_formatted.csv"
    formatted_df.to_csv(formatted_csv_path, index=False)

    # Show rank distribution
    print(f"\nTaxonomic ranks found:")
    for rank in all_ranks:
        count = df[rank.capitalize()].ne('').sum()
        print(f"  {rank}: {count}/{len(bacteria_list)} species have this rank")