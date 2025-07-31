import numpy as np
import pandas as pd
import os

def to_csv(npy_file="embedding_labels.npy", csv_file="bacteria_embeddings_labels.csv"):
    try:
        data = np.load(npy_file, allow_pickle=True)
        df = pd.DataFrame(data)
        df.to_csv(csv_file, index=False)
        print(f"Saved embeddings to {csv_file}")
    except Exception as e:
        print(f"Error saving to CSV: {e}")

def format_taxonomy_detailed(taxonomy_string):
    """Format taxonomy string for detailed output"""
    parts = taxonomy_string.split("|")
    
    # Extract the species (the last part) and genus (second to last)
    species = parts[-1].split("__")[-1]
    genus = parts[-2].split("__")[-1] if len(parts) >= 2 else ""
    
    # Create the formatted name (g__Genus.s__species)
    formatted_name = f"g__{genus}.s__{species}"
    
    # Extract taxonomy levels directly from input
    # Expected order: k__Kingdom|p__Phylum|c__Class|o__Order|f__Family|g__Genus|s__Species
    kingdom = ""
    phylum = ""
    class_name = ""
    order = ""
    family = ""
    
    for part in parts:
        if part.startswith("k__"):
            kingdom = part.split("__")[-1]
        elif part.startswith("p__"):
            phylum = part.split("__")[-1]
        elif part.startswith("c__"):
            class_name = part.split("__")[-1]
        elif part.startswith("o__"):
            order = part.split("__")[-1]
        elif part.startswith("f__"):
            family = part.split("__")[-1]
    
    # Construct the formatted result for detailed taxonomy
    # [Original Name, Bacteria Name (species), Kingdom, Phylum, Class, Order, Family, Genus]
    formatted_row = [formatted_name, species, kingdom, phylum, class_name, order, family, genus]
    return formatted_row

def format_taxonomy_species(taxonomy_string):
    """Format taxonomy string for species output"""
    parts = taxonomy_string.split("|")
    species = parts[-1].split("__")[-1]
    genus = parts[-2].split("__")[-1] if len(parts) >= 2 else ""
    
    # Create the formatted name
    formatted_name = f"g__{genus}.s__{species}"
    return [formatted_name]

def process_taxonomy(input_file="bacteria_embeddings_labels.csv",
                    detailed_output_file="full_lineage.csv",
                    species_output_file="species_genus.csv"):
    """Process taxonomy file and create two CSV outputs"""
    try:
        # Check if the input file exists
        if not os.path.exists(input_file):
            print(f"Input file {input_file} does not exist.")
            return

        # skip the first row (which is just "0")
        df = pd.read_csv(input_file, header=None, skiprows=1)
        
        taxonomy_lines = df.iloc[:, 0].dropna().tolist()
        
        # Apply the function to each taxonomy string
        detailed_data = [format_taxonomy_detailed(tax) for tax in taxonomy_lines]
        species_data = [format_taxonomy_species(tax) for tax in taxonomy_lines]
        
        # Create DataFrames
        detailed_df = pd.DataFrame(detailed_data, columns=[
            "Original Name", "Bacteria Name (species)", "Kingdom", 
            "Phylum", "Class", "Order", "Family", "Genus"
        ])
        
        species_df = pd.DataFrame(species_data, columns=["Formatted Name"])
        
        # Write both formatted data to new CSV files
        detailed_df.to_csv(detailed_output_file, index=False)
        species_df.to_csv(species_output_file, index=False, header=False)
        
        print(f"Created detailed taxonomy file: {detailed_output_file}")
        print(f"Created species file: {species_output_file}")
        
    except Exception as e:
        print(f"Error processing taxonomy: {e}")

def main():
    to_csv()
    process_taxonomy()

if __name__ == "__main__":
    main()