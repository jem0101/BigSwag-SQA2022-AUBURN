import ast 
import os 
import constants 
import astdump

def getPythonParseObject( pyFile ): 
	try:
		full_tree = ast.parse( open( pyFile ).read())    
	except Exception:
		full_tree = ast.parse(constants.EMPTY_STRING) 
	# print(ast.dump(ast.parse(full_tree)))
	return full_tree 
	
def getImport(pyTree): 
    import_list = []
    for stmt_ in pyTree.body:
        for node_ in ast.walk(stmt_):
        	if isinstance(node_, ast.Import):
        		for name in node_.names:
        			import_list.append( (name.name.split('.')[0] ) )
        	elif isinstance(node_, ast.ImportFrom):
        		if(node_.module is not None):
        			import_list.append( ( node_.module.split('.')[0] ) )
        			for name in node_.names:
        			    import_list.append( (name.name.split('.')[0] ) )
#     print("import list: ", import_list)
    return import_list 
    
def getFunctionDetailsForClaases(pyTree):
    func_list = []
    func_list_per_class = []
    for stmt_ in pyTree.body:
        for node_ in ast.walk(stmt_):
        	if isinstance(node_, ast.ClassDef):
        	    # print(node_.__dict__)
        	    classDict = node_.__dict__ 
        	    class_name, class_bases, class_body = classDict[constants.NAME_KW], classDict[constants.BASE_KW], classDict[constants.BODY_KW]
        	    # print(class_bases)
        	    for class_base in class_bases:
        	        if( isinstance(class_base, ast.Attribute) ):  
        	            arg_dic  = class_base.__dict__
        	            arg_class = arg_dic[constants.VALUE_KW]
        	            arg_name = arg_dic[constants.ATTRIB_KW] 
        	            # print(arg_name)
        	            if( isinstance(arg_class, ast.Name ) ):
        	                # print(arg_class.id)
        	                if ('unittest' in arg_class.id):
        	                    # print("body     " , class_body)
        	                    func_list_per_class = getFunctionAssignments(class_body)
        	                    for each_list in func_list_per_class:
        	                        func_list.append(each_list)      
    return func_list
        
    

def getFunctionAssignments(class_body):
    func_list = []
    for stmt_ in class_body:
        for node_ in ast.walk(stmt_):
            if isinstance(node_, ast.Assign):
                assign_dict = node_.__dict__
                targets, value  =  assign_dict[ constants.TARGETS_KW ], assign_dict[ constants.VALUE_KW ]
                if isinstance(value, ast.Call):
                    funcDict = value.__dict__ 
                    funcName, funcArgs, funcLineNo, funcKeys =  funcDict[ constants.FUNC_KW ], funcDict[ constants.ARGS_KW ], funcDict[constants.LINE_NO_KW], funcDict[constants.KEY_WORDS_KW]  
                    if( isinstance(funcName, ast.Name ) ): 
                        func_name = funcName.id
                        func_list.append(func_name)  
                    elif( isinstance( funcName, ast.Attribute ) ):  
                    	func_name_dict  = funcName.__dict__
                    	func_name = func_name_dict[constants.ATTRIB_KW] 
                    	func_list.append(func_name )  
    return func_list

        
def checkForLibraryImport(pyTree):
    import_list = getImport(pyTree)
    if (constants.TENSOR_LIB in import_list or constants.KERAS_LIB in import_list or constants.TORCH_LIB in import_list or constants.SKLEARN_LIB in import_list):
#         print("import list", import_list)
        return True
    return False
  

def checkAlgoNames(func_list):
    algo_list = []
    for item in func_list:
        if item in constants.all_possible_algo:
            algo_list.append(item)
    return algo_list
    
    
def getClassificationAlgoNames(pyTree):
    algo_list = []
    library_import = checkForLibraryImport(pyTree)
    if library_import:
        func_list = getFunctionDetailsForClaases(pyTree) 
        print("pre algo list  ", func_list)  
        algo_list = checkAlgoNames(func_list)
#     print("algo list  ", algo_list)  
    return algo_list
    
