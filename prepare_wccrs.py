from pathlib import Path

MAP_LABELS = {
    "__label__z_zero": 0,
    "__label__z_amb": 1,
    "__label__z_plus_s": 2,
    "__label__z_plus_m": 2,
    "__label__z_minus_s": 3,
    "__label__z_minus_m": 3,
}


def prepare(input_dir: Path, prefix: str, output_dir: Path):
    prepare_file(input_dir / (prefix + ".train.txt"), output_dir / "train.txt")
    prepare_file(input_dir / (prefix + ".dev.txt"), output_dir / "dev.txt")
    prepare_file(input_dir / (prefix + ".test.txt"), output_dir / "test.txt")


def prepare_file(input_path: Path, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    replace = {"a m ": "am ", "y m ": "ym ", " śmy ": "śmy ", "A M ": "AM ", "Y M ": "YM "}
    with input_path.open("r", encoding="utf-8") as input_file, output_path.open("w", encoding="utf-8") as output_file:
        for line in input_file:
            tokens = line.strip()
            for key, val in replace.items():
                tokens = tokens.replace(key, val)
            tokens = tokens.split()
            words = tokens[:len(tokens)-1]
            label = MAP_LABELS[(tokens[len(tokens)-1])]
            output_file.write(str(label) + " " + " ".join(words) + "\n")


if __name__ == '__main__':
    path: Path = Path("/home/sdadas/Downloads/dataset_clarin/dataset/")
    prepare(path, "hotels.sentence", Path("resources/downstream/WCCRS_HOTELS"))
    prepare(path, "medicine.sentence", Path("resources/downstream/WCCRS_MEDICINE"))