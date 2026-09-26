import csv
import datetime
from pathlib import Path


class SubmissionError(ValueError):
    pass

def write_submission(
    test_s1_ids: set[str],
    valid_target_ids: set[str],
    candidates: dict[str, list[str]],
    matches: dict[str, list[str]],
    output_dir: Path,
    run_id: str | None = None
) -> Path:
    """
    Write matching_results.tsv and candidate_pairs.tsv
    """
    if run_id is None:
        run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
    run_dir = output_dir / run_id
    
    # Validation before writing
    for s1 in matches:
        if s1 not in test_s1_ids:
            raise SubmissionError(f"Match for unknown S1 ID: {s1}")
        m_list = matches[s1]
        c_list = candidates.get(s1, [])
        
        m_set = set(m_list)
        c_set = set(c_list)
        
        if not m_set.issubset(c_set):
            raise SubmissionError(f"Final matches for {s1} are not a subset of candidates")
            
        for tid in m_list:
            if tid not in valid_target_ids:
                raise SubmissionError(f"Invalid or unknown target ID {tid}")
                
        if len(m_set) != len(m_list):
            raise SubmissionError(f"Duplicate matches for S1 {s1}")
            
    for s1 in candidates:
        if s1 not in test_s1_ids:
            raise SubmissionError(f"Candidates for unknown S1 ID: {s1}")
        c_list = candidates[s1]
        for tid in c_list:
            if tid not in valid_target_ids:
                raise SubmissionError(f"Invalid or unknown target ID {tid}")
        if len(set(c_list)) != len(c_list):
            raise SubmissionError(f"Duplicate candidates for S1 {s1}")

    if run_dir.exists():
        raise SubmissionError(f"Output directory {run_dir} already exists.")
        
    run_dir.mkdir(parents=True)
    
    matches_file = run_dir / "matching_results.tsv"
    candidates_file = run_dir / "candidate_pairs.tsv"
    
    # Writing
    sorted_s1_ids = sorted(list(test_s1_ids))
    
    with matches_file.open("w", encoding="utf-8", newline="\n") as fm:
        writer_m = csv.writer(fm, delimiter="\t", lineterminator="\n")
        writer_m.writerow(["source1_entity_id", "matched_entity_ids"])
        
        for s1 in sorted_s1_ids:
            # matches are already duplicate-free as per validation
            m = sorted(matches.get(s1, []))
            writer_m.writerow([s1, ",".join(m)])
            
    with candidates_file.open("w", encoding="utf-8", newline="\n") as fc:
        writer_c = csv.writer(fc, delimiter="\t", lineterminator="\n")
        writer_c.writerow(["source1_entity_id", "candidate_entity_ids"])
        
        for s1 in sorted_s1_ids:
            c = sorted(candidates.get(s1, []))
            writer_c.writerow([s1, ",".join(c)])
            
    return run_dir
