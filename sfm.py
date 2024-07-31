from pathlib import Path

from hloc import (
    extract_features,
    match_features,
    pairs_from_retrieval,
    reconstruction
)
import shutil
from argparse import ArgumentParser

if __name__ == "__main__": 
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, default="room0")
    args = parser.parse_args()

    dataset_name = args.dataset
    
    destination_folder = Path("data/" + dataset_name)
    images = destination_folder / "images"
    if not images.exists():
        images_path = sorted(list(Path("Replica/" + dataset_name + "/results/").glob("frame*")))
        images_path = images_path[::20]
        images.mkdir(parents=True, exist_ok=True)
        for image_path in images_path:
            shutil.copy(image_path, images)

    outputs = Path("outputs/" + dataset_name)
    sfm_pairs = outputs / "pairs-netvlad.txt"
    sfm_dir = outputs / "sfm_superpoint+superglue"

    output_dir = Path(str(destination_folder) + "/sparse/0")
    if not output_dir.exists():
        retrieval_conf = extract_features.confs["netvlad"]
        feature_conf = extract_features.confs["superpoint_inloc"]
        matcher_conf = match_features.confs["superglue"]

        retrieval_path = extract_features.main(retrieval_conf, images, outputs)
        pairs_from_retrieval.main(retrieval_path, sfm_pairs, num_matched=5)

        feature_path = extract_features.main(feature_conf, images, outputs)
        match_path = match_features.main(
            matcher_conf, sfm_pairs, feature_conf["output"], outputs
        )

        model = reconstruction.main(sfm_dir, images, sfm_pairs, feature_path, match_path, image_options=dict(camera_model='PINHOLE'))

        sfm_files = list(Path(sfm_dir).glob("*.bin"))
        
        for sfm_file in sfm_files:
            output_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(str(sfm_file), output_dir)

