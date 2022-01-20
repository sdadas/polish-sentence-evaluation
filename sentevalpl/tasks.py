

TASKS = [
    {
        "dir": "PPC",
        "name": "PPC",
        "type": "ppc"
    },
    {
        "dir": "WCCRS_HOTELS",
        "name": "WCCRS_HOTELS",
        "type": "classification",
        "num_classes": 4
    },
    {
        "dir": "WCCRS_MEDICINE",
        "name": "WCCRS_MEDICINE",
        "type": "classification",
        "num_classes": 4
    },
    {
        "dir": "CDS",
        "name": "CDSC-E",
        "type": "entailment"
    },
    {
        "dir": "SICK",
        "name": "SICK-E",
        "type": "entailment"
    },
    {
        "dir": "CDS",
        "name": "CDSC-R",
        "type": "relatedness"
    },
    {
        "dir": "SICK",
        "name": "SICK-R",
        "type": "relatedness"
    },
    {
        "dir": "8TAGS",
        "name": "8TAGS",
        "type": "classification",
        "num_classes": 8
    }
]


def get_task_names():
    return [task["name"] for task in TASKS]

def get_task_by_name(name: str):
    return next(filter(lambda v: v["name"] == name, TASKS))