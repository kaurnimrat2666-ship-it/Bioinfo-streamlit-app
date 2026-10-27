import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from collections import Counter
from Bio.Seq import Seq
from Bio.SeqUtils import gc_fraction
from Bio.SeqUtils.ProtParam import ProteinAnalysis


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Gene Prediction and Protein Characterization",
    page_icon=None,
    layout="wide"
)


# --------------------------------------------------
# TITLE AND INTRODUCTION
# --------------------------------------------------

st.title("In Silico Gene Prediction and Protein Characterization")

st.write(
    "This application analyzes a DNA sequence using basic "
    "bioinformatics approaches. It identifies candidate open "
    "reading frames (ORFs), translates selected ORFs into proteins, "
    "and calculates basic sequence and protein properties."
)

st.info(
    "Note: ORFs identified by this application are candidate regions "
    "and are not experimentally validated genes."
)


# --------------------------------------------------
# AMINO ACID NAMES
# --------------------------------------------------

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


# --------------------------------------------------
# FUNCTION: CLEAN DNA SEQUENCE
# --------------------------------------------------

def clean_dna(sequence):

    lines = sequence.strip().splitlines()

    sequence_lines = [
        line.strip()
        for line in lines
        if not line.strip().startswith(">")
    ]

    dna = "".join(sequence_lines)
    dna = "".join(dna.split()).upper()

    return dna


# --------------------------------------------------
# FUNCTION: FIND ORFS IN ALL SIX FRAMES
# --------------------------------------------------

def find_orfs(dna, min_protein_length):

    stop_codons = {"TAA", "TAG", "TGA"}

    orfs = []

    dna_length = len(dna)

    strands = [
        ("Forward", dna),
        ("Reverse", str(Seq(dna).reverse_complement()))
    ]

    for strand_name, strand_sequence in strands:

        for frame in range(3):

            position = frame

            while position <= len(strand_sequence) - 3:

                codon = strand_sequence[position:position + 3]

                if codon == "ATG":

                    start_position = position
                    current_position = position + 3

                    while current_position <= len(strand_sequence) - 3:

                        current_codon = strand_sequence[
                            current_position:current_position + 3
                        ]

                        if current_codon in stop_codons:

                            end_position = current_position + 3

                            orf_dna = strand_sequence[
                                start_position:end_position
                            ]

                            protein = str(
                                Seq(orf_dna).translate(to_stop=True)
                            )

                            if len(protein) >= min_protein_length:

                                if strand_name == "Forward":

                                    original_start = start_position + 1
                                    original_end = end_position

                                    frame_label = f"+{frame + 1}"

                                else:

                                    original_start = (
                                        dna_length - end_position + 1
                                    )

                                    original_end = (
                                        dna_length - start_position
                                    )

                                    frame_label = f"-{frame + 1}"

                                orfs.append({
                                    "Strand": strand_name,
                                    "Frame": frame_label,
                                    "Start": original_start,
                                    "End": original_end,
                                    "Length (nt)": len(orf_dna),
                                    "Protein Length (aa)": len(protein),
                                    "DNA Sequence": orf_dna,
                                    "Protein Sequence": protein
                                })

                            break

                        current_position += 3

                position += 3

    return orfs


# --------------------------------------------------
# FUNCTION: PROTEIN CHARACTERIZATION
# --------------------------------------------------

def protein_properties(protein):

    analysis = ProteinAnalysis(protein)

    amino_acid_counts = analysis.count_amino_acids()

    total_amino_acids = len(protein)

    if total_amino_acids == 0:
        return {}

    alanine = amino_acid_counts.get("A", 0)
    valine = amino_acid_counts.get("V", 0)
    isoleucine = amino_acid_counts.get("I", 0)
    leucine = amino_acid_counts.get("L", 0)

    alanine_percentage = (alanine / total_amino_acids) * 100
    valine_percentage = (valine / total_amino_acids) * 100
    isoleucine_percentage = (isoleucine / total_amino_acids) * 100
    leucine_percentage = (leucine / total_amino_acids) * 100

    aliphatic_index = (
        alanine_percentage
        + 2.9 * valine_percentage
        + 3.9 * (isoleucine_percentage + leucine_percentage)
    )

    properties = {
        "Protein Length": total_amino_acids,
        "Molecular Weight (Da)": round(
            analysis.molecular_weight(), 2
        ),
        "Theoretical Isoelectric Point (pI)": round(
            analysis.isoelectric_point(), 2
        ),
        "Instability Index": round(
            analysis.instability_index(), 2
        ),
        "Aliphatic Index": round(
            aliphatic_index, 2
        ),
        "GRAVY": round(
            analysis.gravy(), 3
        )
    }

    return properties


# --------------------------------------------------
# FUNCTION: CREATE REPORT
# --------------------------------------------------

def create_report(
    dna,
    gc,
    selected_orf,
    properties,
    amino_acid_table,
    codon_table
):

    report = []

    report.append("GENE PREDICTION AND PROTEIN CHARACTERIZATION REPORT")
    report.append("=" * 55)

    report.append("")
    report.append("DNA SEQUENCE ANALYSIS")
    report.append("-" * 30)
    report.append(f"DNA Length: {len(dna)} nucleotides")
    report.append(f"GC Content: {gc:.2f}%")

    report.append("")
    report.append("SELECTED ORF")
    report.append("-" * 30)
    report.append(f"Strand: {selected_orf['Strand']}")
    report.append(f"Reading Frame: {selected_orf['Frame']}")
    report.append(f"Start Position: {selected_orf['Start']}")
    report.append(f"End Position: {selected_orf['End']}")
    report.append(
        f"ORF Length: {selected_orf['Length (nt)']} nucleotides"
    )
    report.append(
        f"Protein Length: {selected_orf['Protein Length (aa)']} amino acids"
    )

    report.append("")
    report.append("ORF DNA SEQUENCE")
    report.append("-" * 30)
    report.append(selected_orf["DNA Sequence"])

    report.append("")
    report.append("TRANSLATED PROTEIN SEQUENCE")
    report.append("-" * 30)
    report.append(selected_orf["Protein Sequence"])

    report.append("")
    report.append("PROTEIN CHARACTERIZATION")
    report.append("-" * 30)

    for name, value in properties.items():
        report.append(f"{name}: {value}")

    report.append("")
    report.append("AMINO ACID COMPOSITION")
    report.append("-" * 30)
    report.append(amino_acid_table.to_string(index=False))

    report.append("")
    report.append("CODON USAGE")
    report.append("-" * 30)
    report.append(codon_table.to_string(index=False))

    report.append("")
    report.append("LIMITATION")
    report.append(
        "ORFs are computationally identified candidate regions. "
        "Their biological function and validity as genes have not "
        "been experimentally confirmed."
    )

    return "\n".join(report)


# --------------------------------------------------
# INPUT SECTION
# --------------------------------------------------

st.subheader("1. Enter DNA Sequence")

dna_input = st.text_area(
    "Paste your DNA sequence in FASTA format or as a plain sequence:",
    height=180,
    placeholder=">Example_DNA\nATGAAAGCTGCTGCT..."
)

min_protein_length = st.number_input(
    "Minimum protein length for ORF detection (amino acids)",
    min_value=1,
    max_value=1000,
    value=10,
    step=1
)

analyze_button = st.button(
    "Analyze Sequence",
    type="primary"
)


# --------------------------------------------------
# ANALYSIS SECTION
# --------------------------------------------------

if analyze_button:

    dna = clean_dna(dna_input)

    if not dna:

        st.error("Please enter a DNA sequence.")

    elif any(base not in "ATGC" for base in dna):

        st.error(
            "The sequence contains invalid characters. "
            "Please enter a DNA sequence containing only A, T, G and C."
        )

    elif len(dna) < 3:

        st.error("The DNA sequence is too short for ORF analysis.")

    else:

        # DNA ANALYSIS

        gc = gc_fraction(dna) * 100

        st.subheader("2. DNA Sequence Summary")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Sequence Length",
            f"{len(dna)} nt"
        )

        col2.metric(
            "GC Content",
            f"{gc:.2f}%"
        )

        col3.metric(
            "Amino Acid Minimum Length",
            int(min_protein_length)
        )

        st.write("### Nucleotide Composition")

        nucleotide_counts = Counter(dna)

        nucleotide_table = pd.DataFrame([
            {
                "Nucleotide": nucleotide,
                "Count": nucleotide_counts.get(nucleotide, 0),
                "Percentage (%)": round(
                    nucleotide_counts.get(nucleotide, 0)
                    / len(dna) * 100,
                    2
                )
            }
            for nucleotide in ["A", "T", "G", "C"]
        ])

        st.dataframe(
            nucleotide_table,
            use_container_width=True,
            hide_index=True
        )

        # ORF DETECTION

        st.subheader("3. Open Reading Frame (ORF) Detection")

        orfs = find_orfs(
            dna,
            int(min_protein_length)
        )

        if not orfs:

            st.warning(
                "No complete ORFs meeting the selected minimum "
                "protein length were found. Try a shorter minimum "
                "length or use another DNA sequence."
            )

        else:

            st.success(
                f"{len(orfs)} candidate ORF(s) found across the six reading frames."
            )

            orf_table = pd.DataFrame([
                {
                    "ORF": f"ORF {index + 1}",
                    "Strand": orf["Strand"],
                    "Frame": orf["Frame"],
                    "Start Position": orf["Start"],
                    "End Position": orf["End"],
                    "Length (nt)": orf["Length (nt)"],
                    "Protein Length (aa)": orf["Protein Length (aa)"]
                }
                for index, orf in enumerate(orfs)
            ])

            st.dataframe(
                orf_table,
                use_container_width=True,
                hide_index=True
            )

            st.caption(
                "Coordinates refer to the original DNA sequence. "
                "The six reading frames include three forward and "
                "three reverse-complement frames."
            )

            # ORF VISUALIZATION

            st.subheader("4. ORF Location Visualization")

            fig, ax = plt.subplots(figsize=(10, max(3, len(orfs) * 0.35)))

            for index, orf in enumerate(orfs):

                start = orf["Start"]
                end = orf["End"]

                if orf["Strand"] == "Forward":
                    ax.plot(
                        [start, end],
                        [index, index],
                        linewidth=5,
                        solid_capstyle="butt"
                    )
                else:
                    ax.plot(
                        [start, end],
                        [index, index],
                        linewidth=5,
                        linestyle="--",
                        solid_capstyle="butt"
                    )

            ax.set_yticks(range(len(orfs)))
            ax.set_yticklabels([
                f"ORF {index + 1} ({orf['Frame']})"
                for index, orf in enumerate(orfs)
            ])

            ax.set_xlabel("Position on Original DNA Sequence")
            ax.set_ylabel("Candidate ORF")
            ax.set_title("Candidate ORF Locations")
            ax.grid(axis="x", linestyle=":", alpha=0.5)

            plt.tight_layout()

            st.pyplot(fig)

            plt.close(fig)

            st.caption(
                "Solid lines represent forward-strand ORFs; "
                "dashed lines represent reverse-strand ORFs. "
                "This visualization shows candidate ORF locations, "
                "not experimentally confirmed genes."
            )

            # SELECT ORF

            st.subheader("5. Select an ORF for Protein Analysis")

            selected_index = st.selectbox(
                "Choose an ORF:",
                options=list(range(len(orfs))),
                format_func=lambda i: (
                    f"ORF {i + 1} | "
                    f"Frame {orfs[i]['Frame']} | "
                    f"{orfs[i]['Protein Length (aa)']} amino acids"
                )
            )

            selected_orf = orfs[selected_index]

            protein = selected_orf["Protein Sequence"]

            # DISPLAY DNA AND PROTEIN

            st.write("### Selected ORF DNA Sequence")

            st.code(
                selected_orf["DNA Sequence"],
                language=None
            )

            st.write("### Translated Protein Sequence")

            st.code(
                protein,
                language=None
            )

            # PROTEIN CHARACTERIZATION

            st.subheader("6. Protein Characterization")

            properties = protein_properties(protein)

            property_col1, property_col2 = st.columns(2)

            property_col1.metric(
                "Protein Length",
                f"{properties['Protein Length']} amino acids"
            )

            property_col2.metric(
                "Molecular Weight",
                f"{properties['Molecular Weight (Da)']} Da"
            )

            property_table = pd.DataFrame([
                {
                    "Property": name,
                    "Value": value
                }
                for name, value in properties.items()
            ])

            st.dataframe(
                property_table,
                use_container_width=True,
                hide_index=True
            )

            st.caption(
                "These are sequence-derived estimates. They do not "
                "confirm the protein's biological function or activity."
            )

            # AMINO ACID COMPOSITION

            st.subheader("7. Amino Acid Composition")

            amino_acids = ProteinAnalysis(
                protein
            ).count_amino_acids()

            total_amino_acids = len(protein)

            aa_table = pd.DataFrame([
                {
                    "Amino Acid": amino_acid_names[code],
                    "Code": code,
                    "Count": count,
                    "Percentage (%)": round(
                        count / total_amino_acids * 100,
                        2
                    )
                }
                for code, count in amino_acids.items()
                if count > 0
            ])

            aa_table = aa_table.sort_values(
                by="Count",
                ascending=False
            ).reset_index(drop=True)

            st.write(
                "This table shows how many times each amino acid "
                "occurs in the selected protein."
            )

            st.dataframe(
                aa_table,
                use_container_width=True,
                hide_index=True
            )

            # CODON USAGE

            st.subheader("8. Codon Usage Analysis")

            orf_dna = selected_orf["DNA Sequence"]

            # Exclude the terminal stop codon from codon counts.
            coding_sequence = orf_dna[:-3]

            codons = [
                coding_sequence[i:i + 3]
                for i in range(0, len(coding_sequence), 3)
                if len(coding_sequence[i:i + 3]) == 3
            ]

            codon_counts = Counter(codons)

            codon_table = pd.DataFrame([
                {
                    "Codon": codon,
                    "Count": count
                }
                for codon, count in sorted(codon_counts.items())
            ])

            if not codon_table.empty:

                st.write(
                    "This table shows the number of times each codon "
                    "appears in the selected ORF, excluding its terminal "
                    "stop codon."
                )

                st.dataframe(
                    codon_table,
                    use_container_width=True,
                    hide_index=True
                )

            else:

                st.write(
                    "Codon usage information is not available for this ORF."
                )

            # DOWNLOAD REPORT

            st.subheader("9. Download Analysis Report")

            report_text = create_report(
                dna,
                gc,
                selected_orf,
                properties,
                aa_table,
                codon_table
            )

            st.download_button(
                label="Download Report (TXT)",
                data=report_text,
                file_name="bioinformatics_analysis_report.txt",
                mime="text/plain"
            )

            st.success(
                "Analysis completed. You can download the report above."
    )
