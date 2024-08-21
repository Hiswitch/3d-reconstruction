from pathlib import Path
from argparse import ArgumentParser
from PIL import Image

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--src_dir', type=str, default='')
    parser.add_argument('--dst_dir', type=str, default='')
    args = parser.parse_args()

    src_dir = Path(args.src_dir)
    dst_dir = Path(args.dst_dir)

    for path in src_dir.glob('*'):
        img = Image.open(path)

        if img.height > img.width:
            crop = (img.height - img.width) // 2
            top = crop
            bottom = crop + img.width
            img = img.crop((0, top, img.width, bottom))
            img = img.resize((384, 384))
        else:
            crop = (img.width - img.height) // 2
            left = crop
            right = crop + img.height
            img = img.crop((left, 0, right, img.height))
            img = img.resize((384, 384))

        img.save(dst_dir / path.name)
