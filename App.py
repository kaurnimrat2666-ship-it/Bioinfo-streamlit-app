import streamlit as st
from Bio.Seq import Seq
from Bio.SeqUtils import gc_fraction
from Bio.SeqUtils.ProtParam import ProteinAnalysis

st.title("In Silico Gene Prediction and Protein Characterization")

st.write("Paste your nucleotide FASTA sequence below:")

fasta_input = st.text_area("FASTA Sequence")


def get_sequence(fasta):
    lines = fasta.split("\n")
    seq = ""
    for line in lines:
        if not line.startswith(">"):
            seq += line.strip()
    return seq.upper()


def find_orf(seq):
    start = seq.find("ATG")
    stop_codons = ["TAA", "TAG", "TGA"]

    if start == -1:
        return None

    for i in range(start, len(seq), 3):
        codon = seq[i:i+3]
        if codon in stop_codons:
            return seq[start:i+3]
    return None


if st.button("Analyze Sequence"):

    sequence = get_sequence(fasta_input)

    st.subheader("Basic Sequence Analysis")
    st.write("Sequence Length:", len(sequence))

    gc = gc_fraction(sequence) * 100
    st.write("GC Content:", round(gc, 2), "%")

    orf = find_orf(sequence)

    if orf:
        st.subheader("ORF Found")
        st.write("ORF Length:", len(orf))

        protein = Seq(orf).translate(to_stop=True)

        st.subheader("Translated Protein Sequence")
        st.write(protein)

        analysed = ProteinAnalysis(str(protein))

        st.subheader("Protein Properties")
        st.write("Amino Acid Count:", analysed.count_amino_acids())
        st.write("Molecular Weight:", analysed.molecular_weight())
        st.write("Isoelectric Point:", analysed.isoelectric_point())
        st.write("Aromaticity:", analysed.aromaticity())
        st.write("Instability Index:", analysed.instability_index())

        st.subheader("Mutation Analysis")

        position = st.number_input(
            "Enter position to mutate (1-based index):",
            min_value=1,
            max_value=len(orf),
            step=1
        )

        new_base = st.selectbox("Select new base:", ["A", "T", "G", "C"])

        if st.button("Apply Mutation"):
            mutated_orf = list(orf)
            mutated_orf[position - 1] = new_base
            mutated_orf = "".join(mutated_orf)

            mutated_protein = Seq(mutated_orf).translate(to_stop=True)

            st.write("Mutated Protein Sequence:")
            st.write(mutated_protein)

            mut_analysed = ProteinAnalysis(str(mutated_protein))
            st.write("Mutated Molecular Weight:", mut_analysed.molecular_weight())
            st.write("Mutated Isoelectric Point:", mut_analysed.isoelectric_point())

    else:
        st.write("No valid ORF found in the sequence.")
