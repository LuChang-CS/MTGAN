import os
from datetime import datetime
from collections import OrderedDict

import pandas
import pandas as pd
import numpy as np


class EHRParser:
    pid_col = 'pid'
    adm_id_col = 'adm_id'
    adm_time_col = 'adm_time'
    cid_col = 'cid'

    def __init__(self, path):
        self.path = path

        self.skip_pid_check = False

        self.patient_admission = None
        self.admission_codes = None
        self.admission_procedures = None
        self.admission_medications = None

        self.parse_fn = {'d': self.set_diagnosis, 'p': self.set_procedure, 'm': self.set_medication}

    def set_admission(self):
        raise NotImplementedError

    def set_diagnosis(self):
        raise NotImplementedError

    def set_procedure(self):
        raise NotImplementedError

    def set_medication(self):
        raise NotImplementedError

    def parse_admission(self):
        print('parsing the csv file of admission ...')
        filename, cols, converters = self.set_admission()
        admissions = pd.read_csv(os.path.join(self.path, filename), usecols=list(cols.values()), converters=converters)
        admissions = self._after_read_admission(admissions, cols)
        all_patients = OrderedDict()
        for i, row in admissions.iterrows():
            if i % 100 == 0:
                print('\r\t%d in %d rows' % (i + 1, len(admissions)), end='')
            pid, adm_id, adm_time = row[cols[self.pid_col]], row[cols[self.adm_id_col]], row[cols[self.adm_time_col]]
            if pid not in all_patients:
                all_patients[pid] = []
            admission = all_patients[pid]
            admission.append({self.adm_id_col: adm_id, self.adm_time_col: adm_time})
        print('\r\t%d in %d rows' % (len(admissions), len(admissions)))

        patient_admission = OrderedDict()
        for pid, admissions in all_patients.items():
            if len(admissions) >= 2:
                patient_admission[pid] = sorted(admissions, key=lambda admission: admission[self.adm_time_col])

        self.patient_admission = patient_admission

    def _after_read_admission(self, admissions, cols):
        return admissions

    def _parse_concept(self, concept_type):
        assert concept_type in self.parse_fn.keys()
        filename, cols, converters = self.parse_fn[concept_type]()
        concepts = pd.read_csv(os.path.join(self.path, filename), usecols=list(cols.values()), converters=converters)
        concepts = self._after_read_concepts(concepts, concept_type, cols)
        result = OrderedDict()
        for i, row in concepts.iterrows():
            if i % 100 == 0:
                print('\r\t%d in %d rows' % (i + 1, len(concepts)), end='')
            pid = row[cols[self.pid_col]]
            if self.skip_pid_check or pid in self.patient_admission:
                adm_id, code = row[cols[self.adm_id_col]], row[cols[self.cid_col]]
                if code == '':
                    continue
                if adm_id not in result:
                    result[adm_id] = []
                codes = result[adm_id]
                codes.append(code)
        print('\r\t%d in %d rows' % (len(concepts), len(concepts)))
        return result

    def _after_read_concepts(self, concepts, concept_type, cols):
        return concepts

    def parse_diagnoses(self):
        print('parsing csv file of diagnosis ...')
        self.admission_codes = self._parse_concept('d')

    def parse_procedures(self):
        print('parsing csv file of procedures ...')
        self.admission_procedures = self._parse_concept('p')

    def parse_medications(self):
        print('parsing csv file of medications ...')
        self.admission_medications = self._parse_concept('m')

    def calibrate_patient_by_admission(self):
        print('calibrating patients by admission ...')
        del_pids = []
        for pid, admissions in self.patient_admission.items():
            for admission in admissions:
                adm_id = admission[self.adm_id_col]
                if adm_id not in self.admission_codes:
                    del_pids.append(pid)
                    break
        for pid in del_pids:
            admissions = self.patient_admission[pid]
            for admission in admissions:
                adm_id = admission[self.adm_id_col]
                for concepts in [self.admission_codes]:
                    if adm_id in concepts:
                        del concepts[adm_id]
            del self.patient_admission[pid]

    def calibrate_admission_by_patient(self):
        print('calibrating admission by patients ...')
        concepts_set = [self.admission_codes]
        adm_id_set = set()
        for admissions in self.patient_admission.values():
            for admission in admissions:
                adm_id_set.add(admission[self.adm_id_col])
        del_adm_ids = set()
        for concepts in concepts_set:
            for adm_id in concepts:
                if adm_id not in adm_id_set:
                    del_adm_ids.add(adm_id)
        for adm_id in del_adm_ids:
            for concepts in concepts_set:
                if adm_id in concepts:
                    del concepts[adm_id]

    def sample_patients(self, sample_num, seed):
        np.random.seed(seed)
        keys = list(self.patient_admission.keys())
        selected_pids = np.random.choice(keys, sample_num, False)
        self.patient_admission = {pid: self.patient_admission[pid] for pid in selected_pids}
        admission_codes = dict()
        for admissions in self.patient_admission.values():
            for admission in admissions:
                adm_id = admission[self.adm_id_col]
                admission_codes[adm_id] = self.admission_codes[adm_id]
        self.admission_codes = admission_codes

    def parse(self, sample_num=None, seed=6669):
        self.parse_admission()
        self.parse_diagnoses()
        self.calibrate_patient_by_admission()
        self.calibrate_admission_by_patient()
        if sample_num is not None:
            self.sample_patients(sample_num, seed)
        return self.patient_admission, self.admission_codes


class Mimic3Parser(EHRParser):
    def set_admission(self):
        filename = 'ADMISSIONS.csv'
        cols = {self.pid_col: 'SUBJECT_ID', self.adm_id_col: 'HADM_ID', self.adm_time_col: 'ADMITTIME'}
        converter = {
            'SUBJECT_ID': int,
            'HADM_ID': int,
            'ADMITTIME': lambda cell: datetime.strptime(str(cell), '%Y-%m-%d %H:%M:%S')
        }
        return filename, cols, converter

    def set_diagnosis(self):
        filename = 'DIAGNOSES_ICD.csv'
        cols = {self.pid_col: 'SUBJECT_ID', self.adm_id_col: 'HADM_ID', self.cid_col: 'ICD9_CODE'}
        converter = {'SUBJECT_ID': int, 'HADM_ID': int, 'ICD9_CODE': str}
        return filename, cols, converter

    def set_procedure(self):
        filename = 'PROCEDURES_ICD.csv'
        cols = {self.pid_col: 'SUBJECT_ID', self.adm_id_col: 'HADM_ID', self.cid_col: 'ICD9_CODE'}
        converter = {'SUBJECT_ID': int, 'HADM_ID': int, 'ICD9_CODE': str}
        return filename, cols, converter

    def set_medication(self):
        filename = 'PRESCRIPTIONS.csv'
        cols = {self.pid_col: 'SUBJECT_ID', self.adm_id_col: 'HADM_ID', self.cid_col: 'NDC'}
        converter = {'SUBJECT_ID': int, 'HADM_ID': int, 'NDC': str}
        return filename, cols, converter


class Mimic4Parser(EHRParser):
    def __init__(self, path):
        super().__init__(path)
        self.icd_ver_col = 'icd_version'
        self.icd_map = self._load_icd_map()
        self.patient_year_map = self._load_patient()

    def _load_icd_map(self):
        print('loading ICD-10 to ICD-9 map ...')
        filename = 'icd10-icd9.csv'
        cols = ['ICD10', 'ICD9']
        converters = {'ICD10': str, 'ICD9': str}
        icd_csv = pandas.read_csv(os.path.join(self.path, filename), usecols=cols, converters=converters)
        icd_map = {row['ICD10']: row['ICD9'] for _, row in icd_csv.iterrows()}
        return icd_map

    def _load_patient(self):
        print('loading patients anchor year ...')
        filename = 'patients.csv'
        cols = ['subject_id', 'anchor_year', 'anchor_year_group']
        converters = {'subject_id': int, 'anchor_year': int, 'anchor_year_group': lambda cell: int(str(cell)[:4])}
        patient_csv = pandas.read_csv(os.path.join(self.path, filename), usecols=cols, converters=converters)
        patient_year_map = {row['subject_id']: row['anchor_year'] - row['anchor_year_group']
                            for i, row in patient_csv.iterrows()}
        return patient_year_map

    def set_admission(self):
        filename = 'admissions.csv'
        cols = {self.pid_col: 'subject_id', self.adm_id_col: 'hadm_id', self.adm_time_col: 'admittime'}
        converter = {
            'subject_id': int,
            'hadm_id': int,
            'admittime': lambda cell: datetime.strptime(str(cell), '%Y-%m-%d %H:%M:%S')
        }
        return filename, cols, converter

    def set_diagnosis(self):
        filename = 'diagnoses_icd.csv'
        cols = {
            self.pid_col: 'subject_id',
            self.adm_id_col: 'hadm_id',
            self.cid_col: 'icd_code',
            self.icd_ver_col: 'icd_version'
        }
        converter = {'subject_id': int, 'hadm_id': int, 'icd_code': str, 'icd_version': int}
        return filename, cols, converter

    def _after_read_admission(self, admissions, cols):
        print('\tselecting valid admission ...')
        valid_admissions = []
        n = len(admissions)
        for i, row in admissions.iterrows():
            if i % 100 == 0:
                print('\r\t\t%d in %d rows' % (i + 1, n), end='')
            pid = row[cols[self.pid_col]]
            year = row[cols[self.adm_time_col]].year - self.patient_year_map[pid]
            if year > 2012:
                valid_admissions.append(i)
        print('\r\t\t%d in %d rows' % (n, n))
        print('\t\tremaining %d rows' % len(valid_admissions))
        return admissions.iloc[valid_admissions]

    def _after_read_concepts(self, concepts, concept_type, cols):
        print('\tmapping ICD-10 to ICD-9 ...')
        n = len(concepts)
        if concept_type == 'd':
            def _10to9(i, row):
                if i % 100 == 0:
                    print('\r\t\t%d in %d rows' % (i + 1, n), end='')
                cid = row[cid_col]
                if row[icd_ver_col] == 10:
                    if cid not in self.icd_map:
                        code = self.icd_map[cid + '1'] if cid + '1' in self.icd_map else ''
                    else:
                        code = self.icd_map[cid]
                    if code == 'NoDx':
                        code = ''
                else:
                    code = cid
                return code

            cid_col, icd_ver_col = cols[self.cid_col], self.icd_ver_col
            col = np.array([_10to9(i, row) for i, row in concepts.iterrows()])
            print('\r\t\t%d in %d rows' % (n, n))
            concepts[cid_col] = col
        return concepts
