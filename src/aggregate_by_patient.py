import os
import sys
import numpy as np
import pandas as pd


if __name__ == "__main__":

	adm_file = "../raw/ADMISSIONS.csv"
	adm = pd.read_csv(adm_file)
	adm['HADM_ID'] = adm['SUBJECT_ID']
	# print(adm.loc[:, ['SUBJECT_ID', 'HADM_ID']].head(100))
	adm.to_csv("../raw/ADMISSIONS_aggregated.csv")


	diag_file = "../raw/DIAGNOSES_ICD.csv"
	diag = pd.read_csv(diag_file)
	print(diag.columns)
	diag['HADM_ID'] = diag['SUBJECT_ID']
	# print(diag.loc[:, ['SUBJECT_ID', 'HADM_ID']].head(100))
	adm.to_csv("../raw/DIAGNOSES_ICD_aggregated.csv")

