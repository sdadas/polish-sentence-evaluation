from utils.analyzer import PolishAnalyzer

analyzer = PolishAnalyzer()

def write_output(path: str, records):
    header = ("pair_ID", "sentence_A", "sentence_B", "relatedness_score", "entailment_judgment")
    with open(path, "w", encoding="utf-8") as output_file:
        output_file.write("\t".join(header))
        output_file.write("\n")
        for record in records:
            output_file.write("\t".join(record))
            output_file.write("\n")

def analyze(sent: str):
    tokens, _ = analyzer.analyze(sent)
    return " ".join(tokens)

out = {"TRAIN": [], "TEST": [], "TRIAL": []}
with open("resources/SICK_PL.txt", "r") as input_file:
    for idx, line in enumerate(input_file):
        if idx == 0: continue
        values = line.strip().split("\t")
        pair_id = values[0]
        senta = analyze(values[1])
        sentb = analyze(values[2])
        relatedness = values[4]
        entailment = values[3]
        part = values[11]
        record = (pair_id, senta, sentb, relatedness, entailment)
        out[part].append(record)
        assert len(values) == 12

write_output("resources/SICK_train.txt", out["TRAIN"])
write_output("resources/SICK_trial.txt", out["TRIAL"])
write_output("resources/SICK_test_annotated.txt", out["TEST"])
