import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from Bio.Seq import Seq
from Bio.SeqUtils import gc_fraction
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from collections import Counter

st.set_page_config(
    page_title="Bioinformatics Analysis",
    page_icon="",
    layout="wide"
)

st.title("In Silico Gene Prediction and Protein Characterization")
st.write(
    "DNA sequence analysis using Python, Biopython and Streamlit."
)

st.divider()


def parse_sequence(text):
    return "".join(
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.startswith(">")
    ).upper().replace(" ", "")


def find_orfs(dna):
    results = []
    stop_codons = {"TAA", "TAG", "TGA"}
    n = len(dna)

    strands = [
        ("+", dna),
        ("-", str(Seq(dna).reverse_complement()))
    ]

    for strand, sequence in strands:
        for frame in range(3):
            i = frame

            while i + 3 <= len(sequence):
                if sequence[i:i + 3] == "ATG":
                    j = i + 3

                    while j + 3 <= len(sequence):
                        if sequence[j:j + 3] in stop_codons:
                            start = i + 1
                            end = j + 3

                            if strand == "+":
                                original_start = start
                                original_end = end
                            else:
                                original_start = n - end + 1
                                original_end = n - start + 1

                            results.append({
                                "Strand": strand,
                                "Frame": frame + 1,
                                "Start": original_start,
                                "End": original_end,
                                "Length (nt)": end - start + 1,
                                "DNA": sequence[i:j + 3]
                            })
                            break

                        j += 3

                i += 3

    return sorted(
        results,
        key=lambda x: (x["Start"], x["End"])
    )


def protein_properties(protein):
    analysis = ProteinAnalysis(protein)
    aa = analysis.count_amino_acids()
    length = len(protein)

    aliphatic_index = 100 * (
        (aa.get("A", 0) / length)
        + 2.9 * (aa.get("V", 0) / length)
        + 3.9 * (
            (aa.get("I", 0) + aa.get("L", 0)) / length
        )
    )

    return {
        "Protein Length (aa)": length,
        "Molecular Weight (Da)": round(
            analysis.molecular_weight(), 2
        ),
        "Isoelectric Point": round(
            analysis.isoelectric_point(), 2
        ),
        "Aromaticity": round(analysis.aromaticity(), 4),
        "Instability Index": round(
            analysis.instability_index(), 2
        ),
        "Aliphatic Index": round(aliphatic_index, 2),
        "GRAVY": round(analysis.gravy(), 4)
    }


fasta = st.text_area(
    "Enter nucleotide FASTA sequence",
    height=150,
    placeholder=">Example_sequence\nATGGCTGCTGAATAA"
)

analyze = st.button("Analyze Sequence", type="primary")

if analyze or "dna" in st.session_state:
    dna = parse_sequence(fasta)

    if not dna:
        st.error("Please enter a DNA sequence.")
        st.stop()

    if any(base not in "ATGCN" for base in dna):
        st.error("Use only A, T, G, C and N in the DNA sequence.")
        st.stop()

    if "N" in dna:
        st.warning(
            "The sequence contains unknown bases (N); "
            "some ORFs may not be detected."
        )

    gc = gc_fraction(dna) * 100
    orfs = find_orfs(dna)

    st.session_state["dna"] = dna
    st.session_state["orfs"] = orfs
    st.session_state["gc"] = gc

    st.subheader("Sequence Summary")

    col1, col2, col3 = st.columns(3)
    col1.metric("Sequence Length", f"{len(dna)} nt")
    col2.metric("GC Content", f"{gc:.2f}%")
    col3.metric("Candidate ORFs", len(orfs))

    st.divider()
    st.subheader("Six-Frame ORF Analysis")

    if not orfs:
        st.info("No complete ATG-to-stop-codon ORFs were detected.")
    else:
        orf_table = pd.DataFrame([
            {
                "Strand": item["Strand"],
                "Frame": f'{item["Strand"]}{item["Frame"]}',
                "Start Position": item["Start"],
                "End Position": item["End"],
                "Length (nt)": item["Length (nt)"]
            }
            for item in orfs
        ])

        st.dataframe(orf_table, use_container_width=True)

        st.caption(
            "Coordinates are 1-based on the original DNA sequence. "
            "Frame labels refer to the strand being scanned."
        )

        st.subheader("ORF Visualization")

        fig, ax = plt.subplots(figsize=(10, 3))

        frame_positions = {
            ("+", 1): 5,
            ("+", 2): 4,
            ("+", 3): 3,
            ("-", 1): 2,
            ("-", 2): 1,
            ("-", 3): 0
        }

        for key, y in frame_positions.items():
            ax.hlines(y, 1, max(len(dna), 2), linewidth=1.5)

        for item in orfs:
            y = frame_positions[
                (item["Strand"], item["Frame"])
            ]
            ax.barh(
                y,
                item["Length (nt)"],
                left=item["Start"],
                height=0.3
            )

        ax.set_yticks(list(frame_positions.values()))
        ax.set_yticklabels(
            ["+1", "+2", "+3", "-1", "-2", "-3"]
        )
        ax.set_xlim(1, max(len(dna), 2))
        ax.set_xlabel("Position on original DNA (nt)")
        ax.set_title("Candidate ORFs in Six Reading Frames")
        ax.grid(axis="x", alpha=0.2)
        plt.tight_layout()

        st.pyplot(fig)
        plt.close(fig)

        labels = [
            f'ORF {i + 1} | {item["Strand"]}{item["Frame"]} | '
            f'{item["Start"]}-{item["End"]}'
            for i, item in enumerate(orfs)
        ]

        selected_index = st.selectbox(
            "Select an ORF for detailed analysis",
            range(len(orfs)),
            format_func=lambda i: labels[i]
        )

        selected_orf = orfs[selected_index]
        orf_dna = selected_orf["DNA"]
        protein = str(Seq(orf_dna).translate(to_stop=True))

        st.divider()
        st.subheader("Protein Characterization")

        st.write("Selected ORF DNA sequence")
        st.code(orf_dna)

        st.write("Translated protein sequence")
        st.code(protein if protein else "No amino acids translated.")

        if protein:
            properties = protein_properties(protein)

            p1, p2, p3 = st.columns(3)
            p1.metric(
                "Protein Length",
                f'{properties["Protein Length (aa)"]} aa'
            )
            p2.metric(
                "Molecular Weight",
                f'{properties["Molecular Weight (Da)"]} Da'
            )
            p3.metric(
                "Isoelectric Point",
                properties["Isoelectric Point"]
            )

            st.write("Additional protein properties")
            st.dataframe(
                pd.DataFrame(
                    properties.items(),
                    columns=["Property", "Value"]
                ),
                use_container_width=True
            )

            # SIMPLE AMINO ACID COMPOSITION TABLE

            amino_acids = ProteinAnalysis(protein).count_amino_acids()

            amino_acid_names = {
                "A": "Alanine",
                "R": "Arginine",
                "N": "Asparagine",
                "D": "Aspartic acid",
                "C": "Cysteine",
                "Q": "Glutamine",
                "E": "Glutamic acid",
                "G": "Glycine",
                "H": "Histidine",
                "I": "Isoleucine",
                "L": "Leucine",
                "K": "Lysine",
                "M": "Methionine",
                "F": "Phenylalanine",
                "P": "Proline",
                "S": "Serine",
                "T": "Threonine",
                "W": "Tryptophan",
                "Y": "Tyrosine",
                "V": "Valine"
            }

            st.write("Amino Acid Composition")
            st.caption(
                "This table shows the name, code, count and percentage "
                "of each amino acid in the selected protein."
            )

            aa_table = pd.DataFrame([
                {
                    "Amino Acid": amino_acid_names[code],
                    "Code": code,
                    "Count": count,
                    "Percentage (%)": round(
                        count / len(protein) * 100, 2
                    )
                }
                for code, count in amino_acids.items()
                if count > 0
            ])

            aa_table = aa_table.sort_values(
                by="Count",
                ascending=False
            ).reset_index(drop=True)

            st.dataframe(
                aa_table,
                use_container_width=True,
                hide_index=True
            )

        st.divider()
        st.subheader("Codon Usage Analysis")

        codons = [
            orf_dna[i:i + 3]
            for i in range(0, len(orf_dna) - 3, 3)
        ]

        codon_counts = Counter(codons)

        codon_table = pd.DataFrame([
            {
                "Codon": codon,
                "Amino Acid": str(Seq(codon).translate()),
                "Count": count
            }
            for codon, count in sorted(codon_counts.items())
        ])

        if not codon_table.empty:
            st.dataframe(codon_table, use_container_width=True)
            st.bar_chart(codon_table.set_index("Codon")["Count"])
        else:
            st.info("No codons available for display.")


    st.divider()
    st.subheader("Download Complete Analysis Report")

    if orfs and protein:

        report_lines = [
            "IN SILICO GENE PREDICTION AND PROTEIN CHARACTERIZATION",
            "=" * 60,
            "",
            "1. INPUT DNA SEQUENCE",
            "-" * 35,
            dna,
            "",
            "2. SEQUENCE SUMMARY",
            "-" * 35,
            f"Sequence Length: {len(dna)} nucleotides",
            f"GC Content: {gc:.2f}%",
            f"Candidate ORFs Detected: {len(orfs)}",
            "",
            "3. SELECTED ORF",
            "-" * 35,
            f"Selected ORF: ORF {selected_index + 1}",
            f"Start Position: {selected_orf['Start']}",
            f"End Position: {selected_orf['End']}",
            "",
            "4. TRANSLATED PROTEIN SEQUENCE",
            "-" * 35,
            protein,
            "",
            "5. PROTEIN CHARACTERIZATION",
            "-" * 35,
            f"Protein Length: {len(protein)} amino acids",
            f"Molecular Weight: {properties['Molecular Weight (Da)']} Da",
            f"Isoelectric Point: {properties['Isoelectric Point']}",
            f"GRAVY: {properties['GRAVY']}",
            f"Instability Index: {properties['Instability Index']}",
            f"Aromaticity: {properties['Aromaticity']}",
            f"Aliphatic Index: {properties['Aliphatic Index']}",
            "",
            "6. AMINO ACID COMPOSITION",
            "-" * 35,
            aa_table.to_string(index=False),
            "",
            "7. CODON USAGE ANALYSIS",
            "-" * 35,
            codon_table.to_string(index=False)
        ]

        report = "\n".join(report_lines)

        st.download_button(
            label="Download Complete Analysis Report",
            data=report,
            file_name="Bioinformatics_Analysis_Report.txt",
            mime="text/plain",
            key="complete_analysis_report"
        )

    else:
        st.info("Report will be available after selecting an ORF with a translated protein.")

    st.success("Analysis completed.")
