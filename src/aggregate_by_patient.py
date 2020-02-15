import os
import sys
import numpy as np
import pandas as pd


if __name__ == "__main__":

	aggregate = int(sys.argv[1])

	adm_file = "../raw/ADMISSIONS.csv"
	adm = pd.read_csv(adm_file)
	adm = adm.drop(['ROW_ID'], axis = 1)
	if aggregate:
		adm['HADM_ID'] = adm['SUBJECT_ID']
		adm.to_csv("../raw/ADMISSIONS_aggregated.csv", index = False)
	else:
		adm.to_csv("../raw/ADMISSIONS_not_aggregated.csv", index = False)

	diag_file = "../raw/DIAGNOSES_ICD.csv"
	diag = pd.read_csv(diag_file)
	diag = diag.drop(['ROW_ID'], axis = 1)
	if aggregate:
		print(aggregate)
		diag['HADM_ID'] = diag['SUBJECT_ID']
		diag.to_csv("../raw/DIAGNOSES_ICD_aggregated.csv", index = False)
	else:
		diag.to_csv("../raw/DIAGNOSES_ICD_not_aggregated.csv", index = False)
