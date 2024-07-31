from pathlib import Path

from hloc import (
    extract_features,
    match_features,
    pairs_from_retrieval,
    reconstruction
)

if __name__ == "__main__": 
    images = Path("data/orth/")

    outputs = Path("outputs/orth/")
    sfm_pairs = outputs / "pairs-netvlad.txt"
    sfm_dir = outputs / "sfm_superpoint+superglue"

    retrieval_conf = extract_features.confs["netvlad"]
    feature_conf = extract_features.confs["superpoint_aachen"]
    matcher_conf = match_features.confs["superglue"]

    retrieval_path = extract_features.main(retrieval_conf, images, outputs)
    pairs_from_retrieval.main(retrieval_path, sfm_pairs, num_matched=5)

    feature_path = extract_features.main(feature_conf, images, outputs)
    match_path = match_features.main(
        matcher_conf, sfm_pairs, feature_conf["output"], outputs
    )

    model = reconstruction.main(sfm_dir, images, sfm_pairs, feature_path, match_path)