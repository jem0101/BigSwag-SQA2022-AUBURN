import traceback
import pandas as pd
import json
import sys
sys.path.insert(1, 'TestOrchestrator4ML-main/resources/Code/generation/identify_algo')
import py_parser


# Fuzzing 5  methods from  original code
# Cross-references original methods with set values and compares

def checkAlgoNamesFuzz():
	try:
		f = open("blns.json", "r")
		filedata = json.load(f)
		for i in filedata:
			print("FUZZ: Running on method checkAlgoNames() with parameter \"{}\"".format(i))
			checkAlgoNames(i)
	except Exception as exc:
		print("	FUZZ: checkAlgoNames FAILED")
	else:
		print("FUZZ: checkAlgoNames PASSED" )



def checkForLibraryImportFuzz():
	try:
		f = open("blns.json", "r")
		filedata = json.load(f)
		for i in filedata:
			print("FUZZ: Running on method checkAlgoNames() with parameter \"{}\"".format(i))
			checkForLibraryImport(i)
	except Exception as exc:
		print("	FUZZ: checkForLibraryImport FAILED")
	else:
		print("FUZZ: checkForLibraryImport PASSED" )


def getImportFuzz():
	try:
		f = open("blns.json", "r")
		filedata = json.load(f)
		for i in filedata:
			print("FUZZ: Running on method getImport() with parameter \"{}\"".format(i))
			getImport(i)
	except Exception as exc:
		print("	FUZZ: getImport FAILED")
	else:
		print("FUZZ: getImport PASSED" )



def getFunctionDetailsForClaasesFuzz():
	try:
		f = open("blns.json", "r")
		filedata = json.load(f)
		for i in filedata:
			print("FUZZ: Running on method getFunctionDetailsForClaases() with parameter \"{}\"".format(i))
			getFunctionDetailsForClaases(i)
	except Exception as exc:
		print("	FUZZ: getFunctionDetailsForClaases FAILED")
	else:
		print("FUZZ: getFunctionDetailsForClaases PASSED" )



def getClassificationAlgoNamesFuzz():
	try:
		f = open("blns.json", "r")
		filedata = json.load(f)
		for i in filedata:
			print("FUZZ: Running on method getClassificationAlgoNames() with parameter \"{}\"".format(i))
			getClassificationAlgoNames(i)
	except Exception as exc:
		print("	FUZZ: getClassificationAlgoNames FAILED")
	else:
		print("FUZZ: getClassificationAlgoNames PASSED" )


if __name__ == '__main__':

	checkAlgoNamesFuzz()
	checkForLibraryImportFuzz()
	getImportFuzz()
	getFunctionDetailsForClaasesFuzz()
	getClassificationAlgoNamesFuzz()