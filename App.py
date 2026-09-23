"""
In Silico Gene Prediction and Protein Characterization
--------------------------------------------------------
A Streamlit app that:
  1. Validates and parses a pasted nucleotide FASTA sequence
  2. Reports polished, biologically meaningful nucleotide statistics
  3. Scans all 6 reading frames (3 forward + 3 reverse-complement) for ORFs
  4. Translates the best ORF and reports protein physicochemical properties
  5. Optionally attempts a putative identification of the protein via NCBI BLAST
     (to suggest likely function / biological source)
  6. Links out to open, free structure-prediction tools (ExPASy, SWISS-MODEL,
     AlphaFold, Phyre2, PSIPRED) with the sequence ready to copy/paste
  7. Lets the user apply a point mutation and see its effect
  8. Lets the user download a full report of each search (TXT / JSON / CSV)
     to their own device

Requires: streamlit, biopython
    pip install streamlit biopython
Run with: streamlit run gene_prediction_app.py
"""

import json
import io
import csv
import datetime

import streamlit as st
from Bio.Seq import Seq
from Bio.SeqUtils import gc_fraction, molecular_weight
from Bio.SeqUtils.ProtParam import ProteinAnalysis

try:
    from Bio.SeqUtils import MeltingTemp as mt
    HAVE_MT = True
except ImportError:
    HAVE_MT = False

st.set_page_config(page_title="Gene Prediction & Protein Characterization", layout="wide")
st.title("In Silico Gene Prediction and Protein Characterization")

VALID_BASES = set("ACGTN")
STOP_CODONS = {"TAA", "TAG", "TGA"}

# ---------------------------------------------------------------------------
# Session state initialisation (fixes the "Apply Mutation" button losing the
# analysis view on rerun, since Streamlit reruns the whole script on every
# widget interaction and a nested `if st.button(...)` block was being reset)
# ---------------------------------------------------------------------------
for key in ["analyzed", "sequence", "orfs", "best_orf", "protein_str",
            "analysed", "blast_result", "mutated_protein_str", "mut_analysed"]:
    if key not in st.session_state:
        st.session_state[key] = None


# ---------------------------------------------------------------------------
# Core helper functions
# ---------------------------------------------------------------------------

def parse_fasta(raw_text):
    """Extract the sequence from pasted FASTA text (first record only)."""
    lines = raw_text.strip().split("\n")
    seq = "".join(line.strip() for line in lines if not line.startswith(">"))
    return seq.upper().replace(" ", "")


def validate_sequence(seq):
    """Return (is_valid, message)."""
    if not seq:
        return False, "No sequence detected. Please paste a FASTA sequence."
    bad_chars = set(seq) - VALID_BASES
    if bad_chars:
        return False, (f"Sequence contains invalid character(s): "
                        f"{', '.join(sorted(bad_chars))}. Only A, T, G, C, N are allowed.")
    if len(seq) < 6:
        return False, "Sequence is too short to contain a codon-based ORF."
    return True, "Sequence looks valid."


def find_orfs_in_frame(seq, frame, strand):
    """Find all ATG...stop ORFs in a single reading frame of a given strand."""
    orfs = []
    i = frame
    n = len(seq)
    while i < n - 2:
        codon = seq[i:i + 3]
        if codon == "ATG":
            for j in range(i, n - 2, 3):
                stop_codon = seq[j:j + 3]
                if stop_codon in STOP_CODONS:
                    orf_seq = seq[i:j + 3]
                    orfs.append({
                        "seq": orf_seq,
                        "frame": frame + 1,
                        "strand": strand,
                        "start": i + 1,
                        "end": j + 3,
                        "length": len(orf_seq),
                    })
                    break
        i += 3
    return orfs


def find_all_orfs(seq, min_length=30):
    """Scan all 6 reading frames (forward + reverse complement) for ORFs."""
    all_orfs = []
    for frame in range(3):
        all_orfs.extend(find_orfs_in_frame(seq, frame, "+"))

    rev_seq = str(Seq(seq).reverse_complement())
    for frame in range(3):
        all_orfs.extend(find_orfs_in_frame(rev_seq, frame, "-"))

    all_orfs = [o for o in all_orfs if o["length"] >= min_length]
    all_orfs.sort(key=lambda o: o["length"], reverse=True)
    return all_orfs


def nucleotide_report(seq):
    """Build a polished nucleotide-level summary using real Biopython metrics."""
    length = len(seq)
    counts = {b: seq.count(b) for b in "ATGC"}
    gc = gc_fraction(seq) * 100
    at = 100 - gc

    report = {
        "Length (bp)": length,
        "A count": counts["A"],
        "T count": counts["T"],
        "G count": counts["G"],
        "C count": counts["C"],
        "GC content (%)": round(gc, 2),
        "AT content (%)": round(at, 2),
    }

    try:
        report["Molecular weight (Da, ssDNA)"] = round(molecular_weight(seq, seq_type="DNA"), 2)
    except Exception:
        pass

    if HAVE_MT and length <= 10000:
        try:
            report["Estimated melting temp, Tm (°C, Wallace rule)"] = round(mt.Tm_Wallace(seq), 2)
        except Exception:
            pass

    return report


def protein_report(protein_str):
    analysed = ProteinAnalysis(protein_str)
    helix, turn, sheet = analysed.secondary_structure_fraction()
    instability = analysed.instability_index()

    report = {
        "Length (aa)": len(protein_str),
        "Amino acid composition": analysed.count_amino_acids(),
        "Molecular weight (Da)": round(analysed.molecular_weight(), 2),
        "Theoretical isoelectric point (pI)": round(analysed.isoelectric_point(), 2),
        "Aromaticity": round(analysed.aromaticity(), 3),
        "Instability index": round(instability, 2),
        "Stability prediction": "Unstable" if instability > 40 else "Stable",
        "GRAVY (hydropathy)": round(analysed.gravy(), 3),
        "Estimated secondary structure fraction": {
            "Helix": round(helix, 3),
            "Turn": round(turn, 3),
            "Sheet": round(sheet, 3),
        },
    }
    return analysed, report


def try_blast_identification(protein_str, hitlist_size=3):
    """Attempt a putative identification of the protein via NCBI BLASTp.
    Requires internet access; can take a couple of minutes."""
    try:
        from Bio.Blast import NCBIWWW, NCBIXML
    except ImportError:
        return {"error": "Biopython BLAST module not available."}

    try:
        result_handle = NCBIWWW.qblast("blastp", "nr", protein_str, hitlist_size=hitlist_size)
        blast_record = NCBIXML.read(result_handle)
        hits = []
        for alignment in blast_record.alignments[:hitlist_size]:
            hsp = alignment.hsps[0]
            hits.append({
                "title": alignment.title,
                "e_value": hsp.expect,
                "identity": f"{hsp.identities}/{hsp.align_length} "
                            f"({100 * hsp.identities / hsp.align_length:.1f}%)",
            })
        return {"hits": hits}
    except Exception as e:
        return {"error": f"BLAST search failed or timed out: {e}"}


def build_report_text(data):
    lines = [f"Gene Prediction & Protein Characterization Report",
             f"Generated: {datetime.datetime.now().isoformat(timespec='seconds')}",
             "=" * 60, ""]
    for section, content in data.items():
        lines.append(f"--- {section} ---")
        if isinstance(content, dict):
            for k, v in content.items():
                lines.append(f"{k}: {v}")
        elif isinstance(content, list):
            for item in content:
                lines.append(str(item))
        else:
            lines.append(str(content))
        lines.append("")
    return "\n".join(lines)


def build_report_csv(data):
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["Section", "Key", "Value"])
    for section, content in data.items():
        if isinstance(content, dict):
            for k, v in content.items():
                writer.writerow([section, k, v])
        elif isinstance(content, list):
            for i, item in enumerate(content):
                writer.writerow([section, f"item_{i}", item])
        else:
            writer.writerow([section, "", content])
    return buf.getvalue()


# ---------------------------------------------------------------------------
# UI - Input
# ---------------------------------------------------------------------------
st.write("Paste your nucleotide FASTA sequence below:")
fasta_input = st.text_area("FASTA Sequence", height=150)

col_a, col_b = st.columns([1, 2])
with col_a:
    run_blast = st.checkbox(
        "Attempt protein identification via NCBI BLAST",
        value=False,
        help="Requires an internet connection and may take 1-3 minutes."
    )
with col_b:
    min_orf_len = st.slider("Minimum ORF length to report (bp)", 30, 300, 30, step=3)

if st.button("Analyze Sequence"):
    sequence = parse_fasta(fasta_input)
    valid, message = validate_sequence(sequence)

    if not valid:
        st.error(message)
        st.session_state["analyzed"] = False
    else:
        orfs = find_all_orfs(sequence, min_length=min_orf_len)
        st.session_state["sequence"] = sequence
        st.session_state["nuc_report"] = nucleotide_report(sequence)
        st.session_state["orfs"] = orfs

        if orfs:
            best_orf = orfs[0]
            protein_str = str(Seq(best_orf["seq"]).translate(to_stop=True))
            analysed, prot_report = protein_report(protein_str)

            st.session_state["best_orf"] = best_orf
            st.session_state["protein_str"] = protein_str
            st.session_state["analysed"] = analysed
            st.session_state["prot_report"] = prot_report
            st.session_state["blast_result"] = (
                try_blast_identification(protein_str) if run_blast else None
            )
        else:
            st.session_state["best_orf"] = None
            st.session_state["protein_str"] = None

        st.session_state["analyzed"] = True

# ---------------------------------------------------------------------------
# UI - Results (read from session_state so it survives reruns triggered by
# the mutation / download buttons below)
# ---------------------------------------------------------------------------
if st.session_state.get("analyzed"):

    st.subheader("Nucleotide Sequence Analysis")
    st.json(st.session_state["nuc_report"])

    orfs = st.session_state["orfs"]
    if not orfs:
        st.warning("No valid ORF found in the sequence.")
    else:
        st.subheader(f"ORFs Found ({len(orfs)} across 6 reading frames)")
        st.dataframe([
            {"Frame": o["frame"], "Strand": o["strand"], "Start": o["start"],
             "End": o["end"], "Length (bp)": o["length"]}
            for o in orfs
        ], use_container_width=True)

        best_orf = st.session_state["best_orf"]
        protein_str = st.session_state["protein_str"]
        st.info(f"Using the longest ORF (frame {best_orf['frame']}, "
                f"{best_orf['strand']} strand, {best_orf['length']} bp) for translation.")

        st.subheader("Translated Protein Sequence")
        st.code(protein_str, language="text")

        st.subheader("Protein Properties")
        st.json(st.session_state["prot_report"])

        st.subheader("Putative Function, Origin & Likely Uses")
        blast_result = st.session_state["blast_result"]
        if blast_result is None:
            st.write(
                "Enable **'Attempt protein identification via NCBI BLAST'** above and "
                "re-run the analysis to get a putative match against known proteins "
                "(source organism, likely function/uses) based on sequence homology. "
                "Without a homology search, function cannot be reliably inferred from "
                "sequence alone."
            )
        elif "error" in blast_result:
            st.warning(blast_result["error"])
        elif not blast_result["hits"]:
            st.write("No significant homologous proteins were found in the database.")
        else:
            for hit in blast_result["hits"]:
                st.markdown(f"**{hit['title']}**")
                st.write(f"- Sequence identity: {hit['identity']}")
                st.write(f"- E-value: {hit['e_value']}")
                st.write(
                    "The hit title typically indicates the protein's likely name, "
                    "the organism it was found in, and can be used to look up its "
                    "known biological role (e.g. via UniProt or NCBI Gene)."
                )
                st.write("---")

        st.subheader("Explore Protein Structure (Primary / Secondary / Tertiary)")
        st.write(
            "Use the sequence below with free, open structure-prediction tools. "
            "Copy it and paste it into the tool of your choice:"
        )
        st.code(f">predicted_protein\n{protein_str}", language="text")

        st.markdown("""
- **Primary structure** – the amino acid sequence itself, shown above.
- **Secondary structure** (helix/sheet/turn prediction):
  [PSIPRED](http://bioinf.cs.ucl.ac.uk/psipred/) ·
  [ExPASy tools](https://www.expasy.org/resources)
- **Tertiary structure** (3D model prediction):
  [AlphaFold DB / Server](https://alphafold.ebi.ac.uk/) ·
  [SWISS-MODEL](https://swissmodel.expasy.org/) ·
  [Phyre2](http://www.sbg.bio.ic.ac.uk/phyre2/html/page.cgi?id=index)

Paste the protein sequence above (in FASTA format) directly into any of these free tools
to generate a full structural prediction.
""")

        # -----------------------------------------------------------------
        # Mutation analysis
        # -----------------------------------------------------------------
        st.subheader("Mutation Analysis")
        position = st.number_input(
            "Enter position to mutate (1-based index, within ORF):",
            min_value=1, max_value=best_orf["length"], step=1, key="mut_pos"
        )
        new_base = st.selectbox("Select new base:", ["A", "T", "G", "C"], key="mut_base")

        if st.button("Apply Mutation"):
            mutated_orf = list(best_orf["seq"])
            mutated_orf[position - 1] = new_base
            mutated_orf = "".join(mutated_orf)

            mutated_protein_str = str(Seq(mutated_orf).translate(to_stop=True))
            mut_analysed, mut_report = protein_report(mutated_protein_str)

            st.session_state["mutated_protein_str"] = mutated_protein_str
            st.session_state["mut_report"] = mut_report

        if st.session_state.get("mutated_protein_str"):
            st.write("**Mutated Protein Sequence:**")
            st.code(st.session_state["mutated_protein_str"], language="text")
            st.write("**Mutated Protein Properties:**")
            st.json(st.session_state["mut_report"])

        # -----------------------------------------------------------------
        # Save / download results
        # -----------------------------------------------------------------
        st.subheader("Save Results")
        st.write("Download the full results of this search to your device:")

        report_data = {
            "Input sequence (length)": len(st.session_state["sequence"]),
            "Nucleotide Analysis": st.session_state["nuc_report"],
            "Best ORF": {k: v for k, v in best_orf.items() if k != "seq"},
            "Translated Protein": protein_str,
            "Protein Properties": st.session_state["prot_report"],
        }
        if blast_result:
            report_data["Putative Identification (BLAST)"] = blast_result
        if st.session_state.get("mutated_protein_str"):
            report_data["Mutated Protein"] = st.session_state["mutated_protein_str"]
            report_data["Mutated Protein Properties"] = st.session_state["mut_report"]

        file_format = st.radio("File format:", ["TXT", "JSON", "CSV"], horizontal=True)

        if file_format == "TXT":
            content = build_report_text(report_data)
            mime, ext = "text/plain", "txt"
        elif file_format == "JSON":
            content = json.dumps(report_data, indent=2, default=str)
            mime, ext = "application/json", "json"
        else:
            content = build_report_csv(report_data)
            mime, ext = "text/csv", "csv"

        st.download_button(
            label=f"Download results (.{ext})",
            data=content,
            file_name=f"sequence_analysis_{datetime.date.today().isoformat()}.{ext}",
            mime=mime,
        )
