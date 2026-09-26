import csv
from pathlib import Path


class GroundTruthParserError(ValueError):
    pass

def load_ground_truth(path: Path) -> dict[str, list[str]]:
    """
    Parse the training ground-truth TSV.
    
    Returns a dictionary mapping source1_entity_id to a sorted list of matched_entity_ids.
    Preserves every S1 entity, including rows with an empty match list.
    """
    if not path.is_file():
        raise GroundTruthParserError(f"File not found: {path}")

    ground_truth = {}
    
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        try:
            header = next(reader)
        except StopIteration as exc:
            raise GroundTruthParserError(f"File is empty: {path}") from exc
            
        if header != ["source1_entity_id", "matched_entity_ids"]:
            raise GroundTruthParserError(f"Unexpected header: {header}")
            
        for line_num, row in enumerate(reader, start=2):
            if len(row) != 2:
                raise GroundTruthParserError(f"Line {line_num}: Expected 2 columns, got {len(row)}")
                
            s1_id, raw_matches = row
            
            if not s1_id.startswith("S1-"):
                raise GroundTruthParserError(f"Line {line_num}: Invalid S1 prefix for {s1_id}")
                
            if s1_id in ground_truth:
                raise GroundTruthParserError(f"Line {line_num}: Duplicate S1 row for {s1_id}")
                
            if not raw_matches.strip():
                ground_truth[s1_id] = []
                continue
                
            matches = raw_matches.split(",")
            seen_matches = set()
            for match in matches:
                if not match.startswith(("S2-", "S3-")):
                    raise GroundTruthParserError(f"Line {line_num}: Invalid target for {match}")
                if match in seen_matches:
                    raise GroundTruthParserError(f"Line {line_num}: Duplicate ID {match}")
                seen_matches.add(match)
                
            ground_truth[s1_id] = sorted(list(seen_matches))
            
    return ground_truth
