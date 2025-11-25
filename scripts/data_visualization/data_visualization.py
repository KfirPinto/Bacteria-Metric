import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from matplotlib.colors import LinearSegmentedColormap
import sys
import os

# --- הגדרת יצירת תיקיות אוטומטית ---
# זה מוודא שהתיקייה קיימת. אם יצרת אותה ידנית - זה לא יפריע.
try:
    os.makedirs('data_visualization_plots/Union', exist_ok=True)
except OSError:
    pass
# -----------------------------------

def data_distribution(gene_tensor, pathway_tensor, bacteria_list):
    # Load data
    gene_tensor = np.load(gene_tensor)       # shape: (samples, bacteria, gene_families)
    pathway_tensor = np.load(pathway_tensor) # shape: (samples, bacteria, pathways)
    bacteria_list = np.load(bacteria_list, allow_pickle=True)   # shape: (bacteria,)

    # === Gene family counts per bacterium ===
    gene_presence = (gene_tensor != 0).any(axis=0)  # shape: (bacteria, gene_families)
    gene_counts = gene_presence.sum(axis=1)         # shape: (bacteria,)

    # === Pathway counts per bacterium ===
    pathway_presence = (pathway_tensor != 0).any(axis=0)  # shape: (bacteria, pathways)
    pathway_counts = pathway_presence.sum(axis=1)         # shape: (bacteria,)

    # === Sort once, by gene counts ===
    sort_indices = np.argsort(-gene_counts)

    # Apply the same sort to all vectors
    sorted_bacteria = bacteria_list[sort_indices]
    sorted_gene_counts = gene_counts[sort_indices]
    sorted_pathway_counts = pathway_counts[sort_indices]

    # === Plot Gene Families ===
    plt.figure(figsize=(20, 10))
    plt.bar(range(len(sorted_gene_counts)), sorted_gene_counts)
    plt.title("Number of Gene Families per Bacterium")
    plt.ylabel("Unique Gene Families")
    plt.xlabel("Bacterium Index")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('data_visualization_plots/genes_per_bacterium')

    # === Plot Pathways (same bacterium order) ===
    plt.figure(figsize=(20, 10))
    plt.bar(range(len(sorted_gene_counts)), sorted_pathway_counts, color='orange')
    plt.title("Number of Pathways per Bacterium (same order as gene family plot)")
    plt.ylabel("Unique Pathways")
    plt.xlabel("Bacterium Index")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('data_visualization_plots/pathways_per_bacterium')

    threshold = 150
    for i, count in enumerate(sorted_pathway_counts):
        if count > threshold:
            print(f"{sorted_bacteria[i]}: {count} pathways")

    threshold = 3000
    for i, count in enumerate(sorted_gene_counts):
        if count > threshold:
            print(f"{sorted_bacteria[i]}: {count} genes")
    
def bacteria_count_across_samples():

    """
    This function returns the numbers of bacteria appears at each sample
    """

    tensor = np.load("/home/dsi/pintokf/Projects/Microbium/Bacteria-Metric/processed_data/gene_abundance_bacteria_regroup_normalized.csv/after_intersection/tensor.npy")  
    sample_list = np.load("/home/dsi/pintokf/Projects/Microbium/Bacteria-Metric/processed_data/gene_abundance_bacteria_regroup_normalized.csv/after_intersection/sample_list.npy", allow_pickle=True)
    tensor = torch.tensor(tensor)

    flattened_tensor = torch.sum(tensor, dim=2) # tensor shape: (samples, bacteria)
    counts = torch.sum(flattened_tensor != 0, dim=1) # tensor shape: (samples)
    bacteria_counts = list(zip(sample_list, counts.numpy()))
    bacteria_counts_sorted = sorted(bacteria_counts, key=lambda x: x[1])
    sorted_samples, sorted_counts = zip(*bacteria_counts_sorted)

    plt.figure(figsize=(40, 100))
    plt.barh(sorted_samples, sorted_counts)  
    plt.xlabel('Count of microbes', fontsize=100)
    plt.title('Count of microbes per sample', fontsize=100)
    plt.xticks(fontsize=50)
    plt.tight_layout()  # avoid overlapping
    plt.savefig('data_visualization_plots/bacteria_count')
    plt.close()

     # Pie chart for samples with at least one microbe
    non_zero_indices = (counts > 0).nonzero(as_tuple=True)[0]
    zero_indices = len(counts) - len(non_zero_indices)

    labels = ['Samples with microbes', 'Samples without microbes']
    sizes = [len(non_zero_indices), zero_indices]

    plt.figure(figsize=(7, 7))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=['lightgreen', 'lightcoral'])
    plt.title('Proportion of samples with at least one microbe', fontsize=20)
    plt.savefig('data_visualization_plots/pie_chart_bacteria')
    plt.close()

def create_heatmap():
        
    tensor = np.load("/home/dsi/pintokf/Projects/Microbium/Bacteria-Metric/processed_data/pathways_processed/after_intersection/tensor.npy") 
    tensor = torch.tensor(tensor)

    flattened_pathways = torch.sum(tensor, dim=2).detach().numpy()
    flattened_pathways_log = np.log1p(flattened_pathways) 
    white_to_red = LinearSegmentedColormap.from_list("white_to_red", ["white", "red"])
    plt.figure(figsize=(20, 20))
    ax = sns.heatmap(flattened_pathways_log, cmap=white_to_red, annot=False, vmin=np.min(flattened_pathways_log), vmax=0.001)
    ax.set_xlabel("bacteria", fontsize=30)  
    ax.set_ylabel("samples", fontsize=30) 
    ax.set_title("Pathways Flattened", fontsize=30)
    plt.tight_layout()
    plt.savefig('data_visualization_plots/heatmap_pathways_flattened.png')
    plt.close()


    flattened_bacteria = torch.sum(tensor, dim=1).detach().numpy()
    flattened_sample_log = np.log1p(flattened_bacteria) 
    plt.figure(figsize=(20, 20))
    ax = sns.heatmap(flattened_sample_log, cmap=white_to_red, annot=False, cbar=True, vmin=np.min(flattened_sample_log), vmax=0.001)
    ax.set_xlabel("pathways", fontsize=30)  
    ax.set_ylabel("samples", fontsize=30) 
    ax.set_title("Bacterium Flattened", fontsize=30)
    plt.tight_layout()
    plt.savefig('data_visualization_plots/heatmap_bacterium_flattened.png')
    plt.close()

"""
    flattened_pathway = torch.sum(tensor, dim=2).detach().numpy()
    flattened_pathway_log = np.log1p(flattened_pathway) 
    print(f"Max: {np.max(flattened_pathway_log)}")
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(flattened_pathway_log, cmap="viridis", annot=False, cbar=True, vmin=np.min(flattened_pathway_log), vmax=0.00175)
    ax.set_xlabel("bacteria")  
    ax.set_ylabel("samples") 
    ax.set_title("Pathway Flattened")
    plt.tight_layout()
    plt.savefig('pathway_flattened.png')
    plt.close()
"""
def visualize_set_intersection(file1, file2):

     # Load sets from .npy files
    set1 = set(np.load(file1, allow_pickle=True))
    set2 = set(np.load(file2, allow_pickle=True))

    # Calculate intersections and unique elements
    intersection = set1 & set2
    only_in_set1 = set1 - set2
    only_in_set2 = set2 - set1

    # Prepare data for visualization
    labels = ['Intersection', '', '']
    sizes = [len(intersection), len(only_in_set1), len(only_in_set2)]
    colors = ['lightblue', 'lightgreen', 'salmon']

    # Create pie chart
    plt.figure(figsize=(8, 8))
    plt.pie(
        sizes, 
        labels=labels, 
        autopct='%1.1f%%', 
        startangle=140, 
        colors=colors, 
        wedgeprops={'edgecolor': 'black'}
    )
    plt.title('samples intersection')
    plt.savefig('data_visualization_plots/samples_intersection')
    plt.close()

def common_pathways():

    tensor = np.load("/home/dsi/pintokf/Projects/Microbium/Bacteria-Metric/processed_data/pathways_processed/tensor.npy") 
    pathway_list = np.load("/home/dsi/pintokf/Projects/Microbium/Bacteria-Metric/processed_data/pathways_processed/pathway_list.npy", allow_pickle=True)  

    tensor = torch.tensor(tensor)

    flattened_tensor = torch.sum(tensor, dim=1)

    binary_tensor = (flattened_tensor > 0).int()

    counts = torch.sum(binary_tensor, dim=0)

    pathway_counts = list(zip(pathway_list, counts.numpy()))
    pathway_counts_sorted = sorted(pathway_counts, key=lambda x: x[1], reverse=True)[:20]
    sorted_pathway, sorted_counts = zip(*pathway_counts_sorted)

    plt.figure(figsize=(20, 20))
    plt.barh(sorted_pathway, sorted_counts)  
    plt.xlabel('Count', fontsize=14)
    plt.ylabel('Pathway', fontsize=14)
    plt.title('Top 20 Most Common Pathways', fontsize=16)
    plt.gca().invert_yaxis()
    plt.tight_layout()  # avoid overlapping
    plt.savefig('data_visualization_plots/Union/common_pathways')
    plt.close()

def pathway_association():

    tensor = np.load("/home/dsi/pintokf/Projects/Microbium/Bacteria-Metric/processed_data/pathways_processed/tensor.npy")  
    bacteria_list = np.load("/home/dsi/pintokf/Projects/Microbium/Bacteria-Metric/processed_data/pathways_processed/bacteria_list.npy", allow_pickle=True)
    tensor = torch.tensor(tensor)

    flattened_tensor = torch.sum(tensor, dim=0) # tensor shape: (bacteria, pathway)
    counts = torch.sum(flattened_tensor != 0, dim=1) # tensor shape: (bacteria)
    bacteria_counts = list(zip(bacteria_list, counts.numpy()))
    bacteria_counts_sorted = sorted(bacteria_counts, key=lambda x: x[1])
    sorted_bacteria, sorted_counts = zip(*bacteria_counts_sorted)

    plt.figure(figsize=(40, 40))
    plt.barh(sorted_bacteria, sorted_counts)  
    plt.xlabel('Count of Associated Pathways', fontsize=50)
    plt.title('Association Between Microbes And Biological Pathways', fontsize=50)
    plt.xticks(fontsize=50)
    plt.tight_layout()  # avoid overlapping
    plt.savefig('data_visualization_plots/Union/pathway_association')
    plt.close()

def bacteria_count():

    tensor = np.load("/home/dsi/pintokf/Projects/Microbium/Bacteria-Metric/processed_data/pathways_processed/tensor.npy")  

    bacteria_list = np.load("/home/dsi/pintokf/Projects/Microbium/Bacteria-Metric/processed_data/pathways_processed/bacteria_list.npy", allow_pickle=True)

    tensor = torch.tensor(tensor)

    # Count the number of instances of sample and pathway for each bacteria. 
    # 'tensor != 0' returns binary tensor. 
    counts = torch.sum(tensor != 0, dim=(0, 2)) 

    # For each bacteria (include zero count), create pairs of bacteria and count
    bacteria_counts = list(zip(bacteria_list, counts.numpy()))

    # Sort the bacteria by count
    bacteria_counts_sorted = sorted(bacteria_counts, key=lambda x: x[1])

    # Split the bacteria and counts into 2 differnet sets
    sorted_bacteria, sorted_counts = zip(*bacteria_counts_sorted)

    plt.figure(figsize=(40, 40))
    plt.barh(sorted_bacteria, sorted_counts)  
    plt.xlabel('Total Count', fontsize=50)
    plt.title('Full Distribution of Microbe Counts', fontsize=50)
    plt.xticks(fontsize=50)
    plt.tight_layout()  # avoid overlapping
    plt.savefig('data_visualization_plots/Union/bacteria_count')
    plt.close()

def non_zero_microbes():
    
    # For each microbe - how many samples are not 0

    tensor = np.load("/home/dsi/pintokf/Projects/Microbium/Bacteria-Metric/processed_data/pathways_processed/tensor.npy")  

    # Step 1: Check if bacterium participates in at least one biological pathway for a given person. 
    # Perform a boolean operaion and sum over the pathway axis (axis=2)
    # The values are True is the baterium appears in any pathway for a given person; otherwise, False
    participation = np.any(tensor > 0, axis=2) # Array shape: (samples, bacteria)
    
    # Step 2: Sum over the samples axis to determine in how many people each bacterium appears
    bacteria_counts = np.sum(participation, axis=0) # tensor shape: (bacteria)

    # Step 3: Filter out bacteria that do not participate in any sample
    bacteria_counts = bacteria_counts[bacteria_counts > 0]

    # Step 4: Create the histogram
    plt.figure(figsize=(10, 6))
    plt.hist(bacteria_counts, bins=range(1, 26), color='salmon', edgecolor='black', rwidth=0.8)

    # Step 5: Customize the plot
    plt.title("Distribution of Microbes Across non-zero Samples")
    plt.xlabel("Non-zero samples in a microbe")
    plt.ylabel("Number of Microbes")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(range(1, max(bacteria_counts) + 1))  
    plt.tight_layout()

    # Step 6: save the plot
    plt.savefig('data_visualization_plots/non_zero_microbes')
    plt.close()

def non_zero_samples():

    # For each sample - how many bacteria are not 0

    tensor = np.load("/home/dsi/pintokf/Projects/Microbium/Bacteria-Metric/processed_data/pathways_processed/tensor.npy")  
    bacteria_list = np.load("/home/dsi/pintokf/Projects/Microbium/Bacteria-Metric/processed_data/pathways_processed/bacteria_list.npy", allow_pickle=True)
    people_list = np.load("/home/dsi/pintokf/Projects/Microbium/Bacteria-Metric/processed_data/pathways_processed/sample_list.npy", allow_pickle=True)

    # Step 1: Check if bacterium participates in at least one biological pathway for a given person. 
    # Perform a boolean operaion and sum over the pathway axis (axis=2)
    # The values are True is the baterium appears in any pathway for a given person; otherwise, False
    participation = np.any(tensor > 0, axis=2) # Array shape: (samples, bacteria)

    # Step 2: Create a DataFrame
    df = pd.DataFrame(participation.T, columns=people_list)  # Transpose participation -> (bacteria, samples)
    df.insert(0, "Bacteria", bacteria_list)  # Add bacteria names as the first column

    df.to_csv("data_visualization_plots/non_zero_bacteria_participation.csv", index=False)

    # Step 2: Sum over the bacteria axis to determine how many bacteria each samples contains
    samples_counts = np.sum(participation, axis=1) # tensor shape: (samples)

    df_1 = pd.DataFrame({
            "Sample": people_list,
            "Non_Zero_Bacteria_Count": samples_counts
        })    
    df_1.to_csv("data_visualization_plots/samples_count.csv", index = False)

    # Step 4: Create the histogram
    plt.figure(figsize=(10, 6))
    plt.hist(samples_counts, bins=range(min(samples_counts), max(samples_counts) + 2), 
            color='salmon', edgecolor='black', rwidth=0.8)

    plt.title("Distribution of Samples by Non-Zero Microbes")
    plt.xlabel("Non-zero microbes in a sample")
    plt.ylabel("Number of samples")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(range(min(samples_counts), max(samples_counts) + 1, 5))  # ערכים בציר X כל 5 מספרים
    plt.tight_layout()

    # Step 6: save the plot
    plt.savefig('data_visualization_plots/non_zero_samples')
    plt.close()

if __name__ == "__main__":
    # יצירת התיקיות הדרושות לשמירה
    try:
        os.makedirs('data_visualization_plots/Union', exist_ok=True)
    except OSError:
        pass

    print("Starting visualization process...")

    # --- פונקציות ללא ארגומנטים (הנתיבים מוגדרים בפנים) ---
    print("Running: create_heatmap...")
    create_heatmap()
    
    print("Running: bacteria_count_across_samples...")
    bacteria_count_across_samples()

    print("Running: common_pathways...")
    common_pathways()

    print("Running: pathway_association...")
    pathway_association()

    print("Running: bacteria_count (Union)...")
    bacteria_count()

    print("Running: non_zero_microbes...")
    non_zero_microbes()

    print("Running: non_zero_samples...")
    non_zero_samples()

    # --- פונקציות עם ארגומנטים (נתיבים לקבצי after_intersection) ---
    
    print("Running: data_distribution...")
    # חובה להשתמש בנתונים אחרי חיתוך כדי שיהיה אותו מספר חיידקים ואותו סדר
    data_distribution(
        gene_tensor="/home/dsi/pintokf/Projects/Microbium/Bacteria-Metric/processed_data/gene_abundance_bacteria_regroup_normalized.csv/after_intersection/tensor.npy",
        pathway_tensor="/home/dsi/pintokf/Projects/Microbium/Bacteria-Metric/processed_data/pathways_processed/after_intersection/tensor.npy",
        bacteria_list="/home/dsi/pintokf/Projects/Microbium/Bacteria-Metric/processed_data/pathways_processed/after_intersection/bacteria_list.npy"
    )

    print("Running: visualize_set_intersection...")
    # השוואה בין רשימות הדגימות (Samples)
    visualize_set_intersection(
        file1="/home/dsi/pintokf/Projects/Microbium/Bacteria-Metric/processed_data/gene_abundance_bacteria_regroup_normalized.csv/after_intersection/sample_list.npy",
        file2="/home/dsi/pintokf/Projects/Microbium/Bacteria-Metric/processed_data/pathways_processed/after_intersection/sample_list.npy"
    )

    print("All plots generated successfully in 'data_visualization_plots' folder!")