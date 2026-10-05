import streamlit as st
from Bio.Seq import Seq
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import io
import re
import math
import datetime
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D projection)

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

# --- simplified Chou-Fasman propensity table (classic published values) ---
# used to predict secondary structure locally, with no internet needed
HELIX_PROP = {"A": 1.42, "R": 0.98, "N": 0.67, "D": 1.01, "C": 0.70, "Q": 1.11,
              "E": 1.51, "G": 0.57, "H": 1.00, "I": 1.08, "L": 1.21, "K": 1.16,
              "M": 1.45, "F": 1.13, "P": 0.57, "S": 0.77, "T": 0.83, "W": 1.08,
              "Y": 0.69, "V": 1.06}
SHEET_PROP = {"A": 0.83, "R": 0.93, "N": 0.89, "D": 0.54, "C": 1.19, "Q": 1.10,
              "E": 0.37, "G": 0.75, "H": 0.87, "I": 1.60, "L": 1.30, "K": 0.74,
              "M": 1.05, "F": 1.38, "P": 0.55, "S": 0.75, "T": 1.19, "W": 1.37,
              "Y": 1.47, "V": 1.70}
TURN_PROP = {"A": 0.66, "R": 0.95, "N": 1.56, "D": 1.46, "C": 1.19, "Q": 0.98,
             "E": 0.74, "G": 1.56, "H": 0.95, "I": 0.47, "L": 0.59, "K": 1.01,
             "M": 0.60, "F": 0.60, "P": 1.52, "S": 1.43, "T": 0.96, "W": 0.96,
             "Y": 1.14, "V": 0.50}


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


def predict_secondary_structure(protein_seq):
    # simple per-residue prediction: pick whichever of helix/sheet/turn
    # has the highest propensity for that amino acid (Chou-Fasman values).
    # This is a basic approximation, not a state-of-the-art predictor,
    # but it runs instantly and needs no internet connection.
    ss = ""
    for aa in protein_seq:
        h = HELIX_PROP.get(aa, 1.0)
        e = SHEET_PROP.get(aa, 1.0)
        t = TURN_PROP.get(aa, 1.0)
        best = max([("H", h), ("E", e), ("C", t)], key=lambda pair: pair[1])
        ss += best[0]
    return ss


def structure_fractions(ss_string):
    n = len(ss_string)
    return {
        "Helix (H) %": round(100 * ss_string.count("H") / n, 1),
        "Sheet (E) %": round(100 * ss_string.count("E") / n, 1),
        "Coil/Turn (C) %": round(100 * ss_string.count("C") / n, 1),
    }


def draw_3d_backbone(ss_string):
    # Builds a rough 3D backbone trace from the secondary structure string:
    # helix residues twist like a real alpha helix, sheet residues stay
    # mostly straight, coil residues wander gently. This is only a
    # simplified illustration of the fold shape, not a real atomic model -
    # getting the true 3D structure needs dedicated modelling software.
    x, y, z = 0.0, 0.0, 0.0
    angle = 0.0
    xs, ys, zs = [x], [y], [z]

    for i, s in enumerate(ss_string):
        if s == "H":
            angle += 100
            rise = 1.5
        elif s == "E":
            angle += 10
            rise = 3.4
        else:
            angle += 40 + 15 * math.sin(i)
            rise = 3.0

        rad = math.radians(angle)
        x += math.cos(rad) * 1.5
        y += math.sin(rad) * 1.5
        z += rise
        xs.append(x)
        ys.append(y)
        zs.append(z)

    fig = plt.figure(figsize=(4.5, 4.5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(xs, ys, zs, color="steelblue", linewidth=2)
    ax.set_axis_off()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def _run_blast(protein_str):
    from Bio.Blast import NCBIWWW, NCBIXML
    result_handle = NCBIWWW.qblast("blastp", "swissprot", protein_str, hitlist_size=1)
    record = NCBIXML.read(result_handle)
    if not record.alignments:
        return "Unidentified", "Unknown"

    title = record.alignments[0].title
    organism_match = re.search(r"\[(.*?)\]", title)
    organism = organism_match.group(1) if organism_match else "Unknown"
    name = title.split("|")[-1].split("[")[0].strip()
    return name, organism


def identify_protein(protein_str, wait_seconds=15):
    # Runs in a background thread with a strict time limit, so this can
    # never hang the app - if NCBI doesn't answer in time, it just reports
    # that instead of leaving the app stuck.
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_run_blast, protein_str)
            return future.result(timeout=wait_seconds)
    except FutureTimeoutError:
        return "Unidentified (lookup took too long)", "Unknown"
    except Exception:
        return "Unidentified (no internet access)", "Unknown"


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
            ss_string = predict_secondary_structure(protein)

            with st.spinner("Analyzing protein (up to 15 seconds)..."):
                protein_name, organism = identify_protein(protein)

            results["ORF Length (bp)"] = len(orf)
            results["Protein Sequence"] = protein
            results["Molecular Weight"] = round(analysed.molecular_weight(), 2)
            results["Isoelectric Point"] = round(analysed.isoelectric_point(), 2)
            results["Aromaticity"] = round(analysed.aromaticity(), 3)
            results["Instability Index"] = round(analysed.instability_index(), 2)
            results["Protein Name"] = protein_name
            results["Organism"] = organism
            results["Secondary Structure"] = ss_string
            results.update(structure_fractions(ss_string))

            st.session_state.results = results
            st.session_state.orf = orf
            st.session_state.ss_string = ss_string
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
        st.write("Protein Name:", r["Protein Name"])
        st.write("Organism:", r["Organism"])
        st.write("ORF Length:", r["ORF Length (bp)"], "bp")

        st.subheader("Protein Properties")
        st.write("Molecular Weight:", r["Molecular Weight"])
        st.write("Isoelectric Point:", r["Isoelectric Point"])
        st.write("Aromaticity:", r["Aromaticity"])
        st.write("Instability Index:", r["Instability Index"])

        st.subheader("Primary Structure")
        st.code(r["Protein Sequence"])

        st.subheader("Secondary Structure")
        st.write("H = helix, E = sheet, C = coil/turn (predicted per residue)")
        st.code(r["Secondary Structure"])
        st.write("Helix:", r["Helix (H) %"], "%  Sheet:", r["Sheet (E) %"],
                  "%  Coil/Turn:", r["Coil/Turn (C) %"], "%")

        st.subheader("Tertiary Structure")
        st.write("Approximate 3D backbone shape based on the predicted secondary structure "
                  "(illustration only, not an exact atomic model):")
        st.image(draw_3d_backbone(st.session_state.ss_string))

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
