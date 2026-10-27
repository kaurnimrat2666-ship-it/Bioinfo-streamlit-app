import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from Bio.Seq import Seq
from Bio.SeqUtils import gc_fraction
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from collections import Counter

st.set_page_config(page_title="DNA Analysis Tool", page_icon="🧬", layout="wide")

st.title("🧬 In Silico Gene Prediction & Protein Characterization")
st.write("DNA sequence analysis using Python, Biopython and Streamlit.")


def parse_sequence(text):
    return "".join(
        line.strip() for line in text.splitlines()
        if line.strip() and not line.startswith(">")
    ).upper().replace(" ", "")


def find_orfs(dna):
    results = []
    stop_codons = {"TAA", "TAG", "TGA"}
    n = len(dna)

    for strand, seq in [
        ("+", dna),
        ("-", str(Seq(dna).reverse_complement()))
    ]:
        for frame in range(3):
            i = frame

            while i + 3 <= len(seq):
                if seq[i:i+3] == "ATG":
                    j = i + 3

                    while j + 3 <= len(seq):
                        if seq[j:j+3] in stop_codons:
                            start = i + 1
                            end = j + 3

                            if strand == "-":
                                original_start = n - end + 1
                                original_end = n - start + 1
                            else:
                                original_start = start
                                original_end = end

                            results.append({
                                "Strand": strand,
                                "Frame": frame + 1,
                                "Start": original_start,
                                "End": original_end,
                                "Length (nt)": end - start + 1,
                                "DNA": seq[i:j+3]
                            })
                            break
                        j += 3
                i += 3

    return sorted(results, key=lambda x: (x["Start"], x["End"]))


fasta = st.text_area(
    "Paste DNA FASTA sequence",
    height=150,
    placeholder=">Example\nATGGCTGCTGAATAA"
)

if st.button("Analyze Sequence", type="primary"):
    dna = parse_sequence(fasta)

    if not dna or any(base not in "ATGCN" for base in dna):
        st.error("Enter a valid DNA sequence containing A, T, G, C or N.")
        st.stop()

    if "N" in dna:
        st.warning("Unknown bases (N) may affect ORF detection.")

    gc = gc_fraction(dna) * 100
    orfs = find_orfs(dna)

    st.session_state["dna"] = dna
    st.session_state["orfs"] = orfs

    st.subheader("📊 Sequence Summary")
    a, b, c = st.columns(3)
    a.metric("Sequence Length", f"{len(dna)} nt")
    b.metric("GC Content", f"{gc:.2f}%")
    c.metric("Candidate ORFs", len(orfs))

    st.subheader("🧬 Six-Frame ORFs")

    if not orfs:
        st.info("No complete ATG-to-stop ORFs detected.")
    else:
        table = pd.DataFrame([
            {
                "Strand": x["Strand"],
                "Frame": f'{x["Strand"]}{x["Frame"]}',
                "Start": x["Start"],
                "End": x["End"],
                "Length (nt)": x["Length (nt)"]
            }
            for x in orfs
        ])
        st.dataframe(table, use_container_width=True)

        st.caption(
            "Start and end positions use 1-based coordinates on the original "
            "DNA sequence. Frame numbers refer to the scanned strand."
        )

        st.subheader("🎨 ORF Visualization")
        fig, ax = plt.subplots(figsize=(10, 3))

        ys = {("+", 1): 5, ("+", 2): 4, ("+", 3): 3,
              ("-", 1): 2, ("-", 2): 1, ("-", 3): 0}

        for key, y in ys.items():
            ax.hlines(y, 1, max(len(dna), 2), linewidth=2)

        for x in orfs:
            y = ys[(x["Strand"], x["Frame"])]
            ax.barh(
                y, x["Length (nt)"],
                left=x["Start"], height=0.35
            )

        ax.set_yticks(list(ys.values()))
        ax.set_yticklabels(["+1", "+2", "+3", "-1", "-2", "-3"])
        ax.set_xlim(1, max(len(dna), 2))
        ax.set_xlabel("Position on original DNA (nt)")
        ax.set_title("Candidate ORFs in Six Reading Frames")
        ax.grid(axis="x", alpha=0.2)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        labels = [
            f'ORF {i+1}: {x["Strand"]}{x["Frame"]}, '
            f'{x["Start"]}-{x["End"]}'
            for i, x in enumerate(orfs)
        ]

        chosen = st.selectbox(
            "Select an ORF for protein analysis",
            range(len(orfs)),
            format_func=lambda i: labels[i]
        )

        selected = orfs[chosen]
        protein = str(Seq(selected["DNA"]).translate(to_stop=True))

        st.subheader("🧪 Protein Characterization")
        st.write("**Translated protein sequence**")
        st.code(protein or "No amino acids translated.")

        if protein:
            pa = ProteinAnalysis(protein)
            props = {
                "Protein Length (aa)": len(protein),
                "Molecular Weight (Da)": round(pa.molecular_weight(), 2),
                "Isoelectric Point": round(pa.isoelectric_point(), 2),
                "Instability Index": round(pa.instability_index(), 2),
                "Aliphatic Index": round(pa.aliphatic_index(), 2),
                "GRAVY": round(pa.gravy(), 3)
            }
            st.dataframe(
                pd.DataFrame(props.items(), columns=["Property", "Value"]),
                use_container_width=True
            )

            st.write("**Amino Acid Composition**")
            aa = pa.count_amino_acids()
            st.bar_chart(pd.Series(aa, name="Count"))

        st.subheader("🧬 Codon Usage")
        codons = [
            selected["DNA"][i:i+3]
            for i in range(0, len(selected["DNA"])-3, 3)
        ]
        counts = Counter(codons)

        codon_table = pd.DataFrame([
            {
                "Codon": codon,
                "Amino Acid": str(Seq(codon).translate()),
                "Count": count
            }
            for codon, count in sorted(counts.items())
        ])

        st.dataframe(codon_table, use_container_width=True)

        st.subheader("🧫 Basic Mutation Comparison")
        st.caption(
            "This compares sequence translations; it does not confirm "
            "biological effects."
        )

        pos = st.number_input(
            "Nucleotide position within selected ORF",
            min_value=1,
            max_value=len(selected["DNA"]),
            value=1
        )
        base = st.selectbox("Replacement base", ["A", "T", "G", "C"])

        if st.button("Compare Mutation"):
            original = selected["DNA"]
            mutated = list(original)
            mutated[int(pos)-1] = base
            mutated = "".join(mutated)

            p1 = str(Seq(original).translate())
            p2 = str(Seq(mutated).translate())

            st.write("Original protein:")
            st.code(p1)
            st.write("Mutated protein:")
            st.code(p2)

            if original[int(pos)-1] == base:
                st.info("The selected base is unchanged.")
            elif p1 == p2:
                st.success("The translated amino-acid sequence is unchanged.")
            else:
                st.warning("The translated amino-acid sequence has changed.")

    report = (
        "BIOINFORMATICS SEQUENCE ANALYSIS REPORT\n"
        f"Sequence length: {len(dna)} nt\n"
        f"GC content: {gc:.2f}%\n"
        f"Candidate ORFs: {len(orfs)}\n\n"
    )

    for i, x in enumerate(orfs, 1):
        report += (
            f"ORF {i}: strand {x['Strand']}, frame {x['Frame']}, "
            f"start {x['Start']}, end {x['End']}, "
            f"length {x['Length (nt)']} nt\n"
        )

    st.subheader("📄 Download Report")
    st.download_button(
        "Download Analysis Report",
        data=report,
        file_name="sequence_analysis_report.txt",
        mime="text/plain"
    )

    st.success("Analysis completed!")
