import argparse

def extract_features():
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=Path, required=True)
    parser.add_argument("--export_dir", type=Path, required=True)
    parser.add_argument(
        "--conf", type=str, default="superpoint_aachen", choices=list(confs.keys())
    )
    parser.add_argument("--as_half", action="store_true")
    parser.add_argument("--image_list", type=Path)
    parser.add_argument("--feature_path", type=Path)
    args = parser.parse_args()
    main(confs[args.conf], args.image_dir, args.export_dir, args.as_half)
