from collections import OrderedDict

from preprocess.parse_csv import EHRParser


def encode_concept(patient_admission, admission_concepts):
    concept_map = OrderedDict()
    for pid, admissions in patient_admission.items():
        for admission in admissions:
            adm_id = admission[EHRParser.adm_id_col]
            if adm_id in admission_concepts:
                concepts = admission_concepts[adm_id]
                for concept in concepts:
                    if concept not in concept_map:
                        concept_map[concept] = len(concept_map)

    admission_concept_encoded = {
        admission_id: list(set(concept_map[concept] for concept in concept))
        for admission_id, concept in admission_concepts.items()
    }
    return admission_concept_encoded, concept_map
