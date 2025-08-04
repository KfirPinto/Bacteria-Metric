# Assign 16S Sequences to Phyla

# Read 16S sequences
seqs <- readLines("seqs_.07_embed.fasta")
seqs <- seqs[seq(2, length(seqs) - 1, by = 2)]

# Load DADA2 (assumes already installed)
library(dada2)

# Path to SILVA database (update if needed)
silva_path <- "C:/Users/LENOVO/Documents/Bioinformatics/MicrobiomeProject/FASTA/silva_nr_v132_train_set.fa.gz"

# Assign taxonomy
set.seed(100)
taxa <- assignTaxonomy(seqs, silva_path, multithread = TRUE)

# Write results to file
write.table(taxa, file = "taxa_results.txt", sep = ",", quote = FALSE, col.names = NA)
