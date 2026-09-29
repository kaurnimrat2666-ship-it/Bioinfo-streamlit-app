import streamlit as st
from Bio.Seq import Seq
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import io
import re
import datetime
from PIL import Image, ImageDraw, ImageFont

st.title("In Silico Gene Prediction and Protein Characterization")

VALID_BASES = set("ACGTN")
STOP_CODONS = ["TAA", "TAG", "TGA"]

# session_state keeps the results saved so that clicking "Apply Mutation"
# (a button inside the analysis block) does not make the whole analysis
# disappear when the page reruns
if "results" not in st.session_state:
    st.session_state.results = None
if "mutation" not in st.session_state:
    st.session_state.mutation = None


def get_sequence(fasta):
    lines = fasta.strip().split("\n")
    seq = ""
    for line in lines:
        if not line.startswith(">"):
            seq += line.strip()
    return seq.upper()


def is_valid_sequence(seq):
    if len(seq) == 0:
        return False, "Please paste a sequence first."
    extra_chars = set(seq) - VALID_BASES
    if extra_chars:
        return False, "Sequence has letters other than A, T, G, C, N: " + ", ".join(extra_chars)
    if len(seq) < 6:
        return False, "Sequence is too short to have a start and stop codon."
    return True, ""


def orfs_in_frame(seq, frame):
    orfs = []
    i = frame
    while i < len(seq) - 2:
        if seq[i:i + 3] == "ATG":
            for j in range(i, len(seq) - 2, 3):
                if seq[j:j + 3] in STOP_CODONS:
                    orfs.append(seq[i:j + 3])
                    break
        i += 3
    return orfs


def find_best_orf(seq):
    # check all 3 forward frames and 3 frames on the reverse complement,
    # so we don't miss the real gene just because it's on the other strand
    candidates = []
    for frame in range(3):
        candidates += orfs_in_frame(seq, frame)

    rev_seq = str(Seq(seq).reverse_complement())
    for frame in range(3):
        candidates += orfs_in_frame(rev_seq, frame)

    if not candidates:
        return None
    return max(candidates, key=len)


def identify_protein(protein_str):
    # tries to find the protein name and organism using NCBI BLAST
    # (needs internet - if it fails for any reason we just say so)
    try:
        from Bio.Blast import NCBIWWW, NCBIXML
        result_handle = NCBIWWW.qblast("blastp", "nr", protein_str, hitlist_size=1)
        record = NCBIXML.read(result_handle)
        if not record.alignments:
            return "Not found", "Not found"

        title = record.alignments[0].title
        organism_match = re.search(r"\[(.*?)\]", title)
        organism = organism_match.group(1) if organism_match else "Unknown"
        name = title.split("|")[-1].split("[")[0].strip()
        return name, organism
    except Exception:
        return "Could not identify (no internet or no match found)", "Could not identify"


def build_text_report(data):
    lines = ["Gene Prediction and Protein Characterization - Results",
             "Generated: " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
             ""]
    for key, value in data.items():
        lines.append(f"{key}: {value}")
    return "\n".join(lines)


def build_image_report(data):
    lines = ["Gene Prediction and Protein Characterization - Results",
             "Generated: " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), ""]
    for key, value in data.items():
        lines.append(f"{key}: {value}")

    font = ImageFont.load_default()
    line_height = 18
    width = 900
    height = line_height * (len(lines) + 2)

    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    y = 10
    for line in lines:
        draw.text((10, y), line, fill="black", font=font)
        y += line_height

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


st.write("Paste your nucleotide FASTA sequence below:")
fasta_input = st.text_area("FASTA Sequence")

if st.button("Analyze Sequence"):
    sequence = get_sequence(fasta_input)
    valid, msg = is_valid_sequence(sequence)

    if not valid:
        st.error(msg)
        st.session_state.results = None
    else:
        orf = find_best_orf(sequence)
        results = {
            "Sequence Length (bp)": len(sequence),
            "A count": sequence.count("A"),
            "T count": sequence.count("T"),
            "G count": sequence.count("G"),
            "C count": sequence.count("C"),
        }

        if orf:
            protein = str(Seq(orf).translate(to_stop=True))
            analysed = ProteinAnalysis(protein)

            with st.spinner("Identifying protein..."):
                protein_name, organism = identify_protein(protein)

            results["ORF Length (bp)"] = len(orf)
            results["Protein Sequence"] = protein
            results["Molecular Weight"] = round(analysed.molecular_weight(), 2)
            results["Isoelectric Point"] = round(analysed.isoelectric_point(), 2)
            results["Aromaticity"] = round(analysed.aromaticity(), 3)
            results["Instability Index"] = round(analysed.instability_index(), 2)
            results["Protein Name"] = protein_name
            results["Organism"] = organism

            st.session_state.results = results
            st.session_state.orf = orf
        else:
            results["ORF"] = "No valid ORF found"
            st.session_state.results = results
            st.session_state.orf = None

        st.session_state.mutation = None

if st.session_state.results:
    r = st.session_state.results

    st.subheader("Sequence Analysis")
    st.write("Sequence Length:", r["Sequence Length (bp)"], "bp")
    st.write("A:", r["A count"], "  T:", r["T count"], "  G:", r["G count"], "  C:", r["C count"])

    if st.session_state.orf:
        st.subheader("Predicted Protein")
        st.write("ORF Length:", r["ORF Length (bp)"], "bp")
        st.code(r["Protein Sequence"])

        st.subheader("Protein Properties")
        st.write("Molecular Weight:", r["Molecular Weight"])
        st.write("Isoelectric Point:", r["Isoelectric Point"])
        st.write("Aromaticity:", r["Aromaticity"])
        st.write("Instability Index:", r["Instability Index"])

        st.subheader("Protein Identification")
        st.write("Protein Name:", r["Protein Name"])
        st.write("Organism:", r["Organism"])

        st.subheader("Check the Protein Structure")
        st.write("Copy the sequence below and paste it into any of these free tools:")
        st.code(">predicted_protein\n" + r["Protein Sequence"])
        st.write("Secondary structure - PSIPRED: http://bioinf.cs.ucl.ac.uk/psipred/")
        st.write("Tertiary structure - AlphaFold: https://alphafold.ebi.ac.uk/")
        st.write("Tertiary structure - SWISS-MODEL: https://swissmodel.expasy.org/")

        st.subheader("Mutation Analysis")
        position = st.number_input("Position to mutate (1-based index):",
                                    min_value=1, max_value=len(st.session_state.orf), step=1)
        new_base = st.selectbox("New base:", ["A", "T", "G", "C"])

        if st.button("Apply Mutation"):
            mutated_orf = list(st.session_state.orf)
            mutated_orf[position - 1] = new_base
            mutated_orf = "".join(mutated_orf)
            mutated_protein = str(Seq(mutated_orf).translate(to_stop=True))
            mut_analysed = ProteinAnalysis(mutated_protein)

            st.session_state.mutation = {
                "Mutated Protein Sequence": mutated_protein,
                "Mutated Molecular Weight": round(mut_analysed.molecular_weight(), 2),
                "Mutated Isoelectric Point": round(mut_analysed.isoelectric_point(), 2),
            }

        if st.session_state.mutation:
            m = st.session_state.mutation
            st.write("Mutated Protein Sequence:")
            st.code(m["Mutated Protein Sequence"])
            st.write("Mutated Molecular Weight:", m["Mutated Molecular Weight"])
            st.write("Mutated Isoelectric Point:", m["Mutated Isoelectric Point"])
    else:
        st.warning("No valid ORF found in the sequence.")

    st.subheader("Save Results")
    report_data = dict(r)
    if st.session_state.mutation:
        report_data.update(st.session_state.mutation)

    file_format = st.radio("Download as:", ["Text (.txt)", "Image (.png)"])

    if file_format == "Text (.txt)":
        st.download_button(
            "Download Results",
            data=build_text_report(report_data),
            file_name="sequence_analysis.txt",
            mime="text/plain",
        )
    else:
        st.download_button(
            "Download Results",
            data=build_image_report(report_data),
            file_name="sequence_analysis.png",
            mime="image/png",
        )
