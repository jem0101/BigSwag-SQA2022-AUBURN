import traceback
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
			#print("FUZZ: Running on method checkAlgoNames() with parameter \"{}\"".format(i))
			py_parser.checkAlgoNames(i)
			#print("FUZZ: checkAlgoNames() with parameter \"{}\" PASSED".format(i))
	except Exception as exc:
		print("FUZZ: checkAlgoNames() with parameter \"{}\" FAILED".format(i))
		traceback.print_exc()
		print("*"*150)
		print("")



def checkForLibraryImportFuzz():
	try:
		f = open("blns.json", "r")
		filedata = json.load(f)
		for i in filedata:
			#print("FUZZ: Running on method checkForLibraryImport() with parameter \"{}\"".format(i))
			py_parser.checkForLibraryImport(i)
			#print("FUZZ: checkForLibraryImport() with parameter \"{}\" PASSED".format(i))
	except Exception as exc:
		print("FUZZ: checkForLibraryImport() with parameter \"{}\" FAILED".format(i))
		traceback.print_exc()
		print("*"*150)
		print("")

def getImportFuzz():
	try:
		f = open("blns.json", "r")
		filedata = json.load(f)
		for i in filedata:
			#print("FUZZ: Running on method getImport() with parameter \"{}\"".format(i))
			py_parser.getImport(i)
			#print("FUZZ: getImport() with parameter \"{}\" PASSED".format(i))
	except Exception as exc:
		print("FUZZ: getImport() with parameter \"{}\" FAILED".format(i))
		traceback.print_exc()
		print("*"*150)
		print("")


def getFunctionDetailsForClaasesFuzz():
	try:
		f = open("blns.json", "r")
		filedata = json.load(f)
		for i in filedata:
			#print("FUZZ: Running on method getFunctionDetailsForClaases() with parameter \"{}\"".format(i))
			py_parser.getFunctionDetailsForClaases(i)
			#print("FUZZ: getFunctionDetailsForClaases() with parameter \"{}\" PASSED".format(i))
	except Exception as exc:
		print("FUZZ: getFunctionDetailsForClaases() with parameter \"{}\" FAILED".format(i))
		traceback.print_exc()
		print("*"*150)
		print("")


def getClassificationAlgoNamesFuzz():
	try:
		f = open("blns.json", "r")
		filedata = json.load(f)
		for i in filedata:
			#print("FUZZ: Running on method getClassificationAlgoNames() with parameter \"{}\"".format(i))
			py_parser.getClassificationAlgoNames(i)
			#print("FUZZ: getClassificationAlgoNames() with parameter \"{}\" PASSED".format(i))
	except Exception as exc:
		print("FUZZ: getClassificationAlgoNames() with parameter \"{}\" FAILED".format(i))
		traceback.print_exc()
		print("*"*150)
		print("")

if __name__ == '__main__':

	checkAlgoNamesFuzz()
	checkForLibraryImportFuzz()
	getImportFuzz()
	getFunctionDetailsForClaasesFuzz()
	getClassificationAlgoNamesFuzz()