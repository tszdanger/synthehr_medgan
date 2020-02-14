import os
import sys
import numpy as np
import pandas as pd


if __name__ == "__main__":

	adm_file = "../raw/ADMISSIONS.csv"
	adm = pd.read_csv(adm_file)
	adm = adm.drop(['ROW_ID'], axis = 1)
	adm['HADM_ID'] = adm['SUBJECT_ID']
	adm.to_csv("../raw/ADMISSIONS_aggregated.csv", index = False)


	diag_file = "../raw/DIAGNOSES_ICD.csv"
	diag = pd.read_csv(diag_file)
	diag = diag.drop(['ROW_ID'], axis = 1)
	diag['HADM_ID'] = diag['SUBJECT_ID']
	diag['HADM_ID'] = diag['SUBJECT_ID']
	adm.to_csv("../raw/DIAGNOSES_ICD_aggregated.csv", index = False)

