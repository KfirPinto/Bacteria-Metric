from Bio.Blast import NCBIWWW
from Bio import SeqIO

records = list(SeqIO.parse("/home/barsapi1/metric/Bacteria-Metric/data/HMP/raw/seqs_.07_embed.fasta", "fasta"))

# Split into chunks (e.g., 200 sequences per batch)
batch_size = 200

# Open one output file in append mode
with open("blast_results_all_batches.xml", "w") as out_handle:
    for i in range(0, len(records), batch_size):
        batch_records = records[i:i+batch_size]
        fasta_data = "\n".join([f">{rec.id}\n{rec.seq}" for rec in batch_records])
        
        print(f"Running BLAST for batch {i//batch_size+1} with {len(batch_records)} sequences...")
        result_handle = NCBIWWW.qblast(
            program="blastn",
            database="nt",
            sequence=fasta_data,
            entrez_query="Bacteria[Organism]"
        )
        
        # Write this batch result and append to the same file
        out_handle.write(result_handle.read())
        out_handle.flush()  # Ensure it's written to disk
