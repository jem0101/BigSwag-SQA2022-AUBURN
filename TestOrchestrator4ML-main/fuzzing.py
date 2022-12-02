import traceback
import pandas as pd
import json

from resources.Code.generation.identify_algo import py_parser


# Fuzzing 5  methods from  original code
# Cross-references original methods with set values and compares

def checkAlgoNamesFuzz():

	try:

		f = open("blns.json", "r")
   		filedata = json.load(f)
    	for i in filedata:
			checkAlgoNames(i)

	except Exception as exc:
		print(f"FUZZ: checkAlgoNames FAILED")
        traceback.print_exc()
	else:
		print(f:"FUZZ: checkAlgoNames Passed" )



def checkForLibraryImportFuzz():

	try:

		f = open("blns.json", "r")
   		filedata = json.load(f)
    		for i in filedata:
				checkForLibraryImport(i)

	except Exception as exc:
		print(f"FUZZ: checkForLibraryImport FAILED")
        traceback.print_exc()
	else:
		print(f:"FUZZ: checkForLibraryImport Passed" )


def getImportFuzz():

	try:

		f = open("blns.json", "r")
   		filedata = json.load(f)
    		for i in filedata:
				getImport(i)

	except Exception as exc:
		print(f"FUZZ: getImport FAILED")
        traceback.print_exc()
	else:
		print(f:"FUZZ: getImport Passed" )



def getFunctionDetailsForClaasesFuzz():

	try:

		f = open("blns.json", "r")
   		filedata = json.load(f)
    		for i in filedata:
				getFunctionDetailsForClaases(i)

	except Exception as exc:
		print(f"FUZZ: getFunctionDetailsForClaases FAILED")
        traceback.print_exc()
	else:
		print(f:"FUZZ: getFunctionDetailsForClaases Passed" )



def getClassificationAlgoNamesFuzz():

	try:
		f = open("blns.json", "r")
   		filedata = json.load(f)
    		for i in filedata:
				getClassificationAlgoNames(i)
	except Exception as exc:
		print(f"FUZZ: getClassificationAlgoNames FAILED")
        traceback.print_exc()
	else:
		print(f:"FUZZ: getClassificationAlgoNames Passed" )


if __name__ == '__main__':

	checkAlgoNamesFuzz()
	checkForLibraryImportFuzz()
	getImportFuzz()
	getFunctionDetailsForClaasesFuzz()
	getClassificationAlgoNamesFuzz()