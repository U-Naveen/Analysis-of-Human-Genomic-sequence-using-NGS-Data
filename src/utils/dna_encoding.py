# ===============================
# K-mer Encoding Utility
# ===============================

def generate_kmers(sequence, k=3):
    """
    Convert DNA sequence into k-mers

    Example:
    ATGCGA with k=3

    → ATG
      TGC
      GCG
      CGA
    """

    kmers = []

    for i in range(len(sequence) - k + 1):

        kmer = sequence[i:i+k]

        kmers.append(kmer)

    return kmers