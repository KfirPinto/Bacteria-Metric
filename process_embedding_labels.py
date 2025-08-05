import numpy as np
import pandas as pd
import os
import argparse
from pathlib import Path


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Process taxonomy data and convert embeddings from numpy to CSV format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Subcommands for different operations
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Convert numpy to CSV subcommand
    convert_parser = subparsers.add_parser('convert', help='Convert numpy embeddings to CSV')
    convert_parser.add_argument(
        '--input', '-i', 
        type=str, 
        default="embeddings_labels_shani.npy",
        help="Input numpy file path"
    )
    convert_parser.add_argument(
        '--output', '-o', 
        type=str, 
        default="embedding_labels_shani.csv",
        help="Output CSV file path"
    )
    
    # Process taxonomy subcommand
    taxonomy_parser = subparsers.add_parser('taxonomy', help='Process taxonomy data')
    taxonomy_parser.add_argument(
        '--input', '-i', 
        type=str, 
        default="filtered_bacteria_shani.csv",
        help="Input taxonomy CSV file path"
    )
    taxonomy_parser.add_argument(
        '--detailed-output', '-d', 
        type=str, 
        default="full_lineage_shani.csv",
        help="Output file for detailed taxonomy information"
    )
    taxonomy_parser.add_argument(
        '--species-output', '-s', 
        type=str, 
        default="species_genus_shani.csv",
        help="Output file for species/genus information"
    )
    taxonomy_parser.add_argument(
        '--skip-header', 
        action='store_true',
        help="Skip the first row when reading input file"
    )
    
    # Combined operation subcommand
    all_parser = subparsers.add_parser('all', help='Run both convert and taxonomy operations')
    all_parser.add_argument(
        '--npy-input', 
        type=str, 
        default="embeddings_labels_shani.npy",
        help="Input numpy file path for conversion"
    )
    all_parser.add_argument(
        '--csv-output', 
        type=str, 
        default="embedding_labels_shani.csv",
        help="Output CSV file path for conversion"
    )
    all_parser.add_argument(
        '--taxonomy-input', 
        type=str, 
        default="filtered_bacteria_shani.csv",
        help="Input taxonomy CSV file path"
    )
    all_parser.add_argument(
        '--detailed-output', 
        type=str, 
        default="full_lineage_shani.csv",
        help="Output file for detailed taxonomy information"
    )
    all_parser.add_argument(
        '--species-output', 
        type=str, 
        default="species_genus_shani.csv",
        help="Output file for species/genus information"
    )
    all_parser.add_argument(
        '--skip-header', 
        action='store_true',
        help="Skip the first row when reading taxonomy input file"
    )
    
    return parser.parse_args()


def to_csv(npy_file, csv_file):
    """
    Convert numpy file to CSV format
    
    Args:
        npy_file (str): Path to input numpy file
        csv_file (str): Path to output CSV file
    """
    try:
        # Check if input file exists
        if not os.path.exists(npy_file):
            print(f"Error: Input file '{npy_file}' does not exist.")
            return False
            
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(csv_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")
        
        # Load and convert data
        data = np.load(npy_file, allow_pickle=True)
        df = pd.DataFrame(data)
        df.to_csv(csv_file, index=False)
        
        print(f"Successfully converted '{npy_file}' to '{csv_file}'")
        print(f"Data shape: {df.shape}")
        return True
        
    except Exception as e:
        print(f"Error converting numpy to CSV: {e}")
        return False


def format_taxonomy_detailed(taxonomy_string):
    """
    Format taxonomy string for detailed output
    
    Args:
        taxonomy_string (str): Raw taxonomy string in format k__Kingdom|p__Phylum|...
        
    Returns:
        list: Formatted taxonomy information
    """
    parts = taxonomy_string.split("|")
    
    # Extract the species (the last part) and genus (second to last)
    species = parts[-1].split("__")[-1] if parts else ""
    genus = parts[-2].split("__")[-1] if len(parts) >= 2 else ""
    
    # Create the formatted name (g__Genus.s__species)
    formatted_name = f"g__{genus}.s__{species}"
    
    # Initialize taxonomy levels
    kingdom = ""
    phylum = ""
    class_name = ""
    order = ""
    family = ""
    
    # Extract taxonomy levels from input
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
    
    # Return formatted row
    return [formatted_name, species, kingdom, phylum, class_name, order, family, genus]

def format_taxonomy_species(taxonomy_string):
    """
    Format taxonomy string for species output
    
    Args:
        taxonomy_string (str): Raw taxonomy string
        
    Returns:
        list: Formatted species name
    """
    parts = taxonomy_string.split("|")
    species = parts[-1].split("__")[-1] if parts else ""
    genus = parts[-2].split("__")[-1] if len(parts) >= 2 else ""
    
    # Create the formatted name
    formatted_name = f"g__{genus}.s__{species}"
    return [formatted_name]

def process_taxonomy(input_file, detailed_output_file, species_output_file, skip_header=False):
    """
    Process taxonomy file and create two CSV outputs
    
    Args:
        input_file (str): Path to input taxonomy file
        detailed_output_file (str): Path to detailed output file
        species_output_file (str): Path to species output file
        skip_header (bool): Whether to skip the first row
        
    Returns:
        bool: Success status
    """
    try:
        # Check if the input file exists
        if not os.path.exists(input_file):
            print(f"Error: Input file '{input_file}' does not exist.")
            return False
        
        # Create output directories if they don't exist
        for output_file in [detailed_output_file, species_output_file]:
            output_dir = os.path.dirname(output_file)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
                print(f"Created output directory: {output_dir}")
        
        # Read the CSV file
        skiprows = 1 if skip_header else None
        df = pd.read_csv(input_file, header=None, skiprows=skiprows)
        taxonomy_lines = df.iloc[:, 0].dropna().tolist()
        
        if not taxonomy_lines:
            print("Warning: No taxonomy data found in input file.")
            return False
        
        print(f"Processing {len(taxonomy_lines)} taxonomy entries...")
        
        # Apply formatting functions to each taxonomy string
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
        
        print(f"Successfully created detailed taxonomy file: '{detailed_output_file}'")
        print(f"Successfully created species file: '{species_output_file}'")
        print(f"Processed {len(detailed_data)} entries")
        
        return True
        
    except Exception as e:
        print(f"Error processing taxonomy: {e}")
        return False

def main():
    """Main function to handle command line arguments and execute operations"""
    args = parse_arguments()
    
    if not args.command:
        print("Error: No command specified. Use --help for usage information.")
        return
    
    success = True
    
    if args.command == 'convert':
        success = to_csv(args.input, args.output)
        
    elif args.command == 'taxonomy':
        success = process_taxonomy(
            args.input, 
            args.detailed_output, 
            args.species_output, 
            args.skip_header
        )
        
    elif args.command == 'all':
        print("=== Converting numpy to CSV ===")
        success1 = to_csv(args.npy_input, args.csv_output)
        
        print("\n=== Processing taxonomy ===")
        success2 = process_taxonomy(
            args.taxonomy_input, 
            args.detailed_output, 
            args.species_output, 
            args.skip_header
        )
        
        success = success1 and success2
    
    if success:
        print("\n✓ All operations completed successfully!")
    else:
        print("\n✗ Some operations failed. Please check the error messages above.")

if __name__ == "__main__":
    main()