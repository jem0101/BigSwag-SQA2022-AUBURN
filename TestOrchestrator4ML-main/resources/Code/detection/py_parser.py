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
    
def getFunctionDetailsForClaases(pyTree, func, algo, attack):
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
        	                    if (func == 1):
        	                        func_list_per_class = getFunctionDefinitionsWithAssert(class_body)
        	                        for each_list in func_list_per_class:
        	                            func_list.append(each_list)     
        	                    if (algo == 1):
        	                        func_list_per_class = getFunctionAssignments(class_body)
        	                        for each_list in func_list_per_class:
        	                            func_list.append(each_list)    
        	                    if (attack == 1):
        	                        func_list_per_class = getFunctionAssignmentsWithLHS(class_body)
        	                        for each_list in func_list_per_class:
        	                            func_list.append(each_list)       
    return func_list
        

def getFunctionDefinitionsWithAssert(class_body):
    func_list = []
    for stmt_ in class_body:
        for node_ in ast.walk(stmt_):
            if isinstance(node_, ast.FunctionDef):
                funcDict = node_.__dict__ 
                func_name, funcLineNo, func_bodies =  funcDict[ constants.NAME_KW ], funcDict[constants.LINE_NO_KW], funcDict[constants.BODY_KW]
                body_list = []
                check_assert_block = False
                index = 0                
                for x_ in range(len(func_bodies)):
                    index = x_ + 1
                    func_body = func_bodies[x_]    
                    # print(func_body)     
                    if( isinstance(func_body, ast.Expr ) )  :
                        # print(func_body.value)
                        body_value = func_body.value
                        if( isinstance( body_value, ast.Call ) ):
                            func_arg_dict  = body_value.__dict__
                            # print(func_arg_dict)
                            func_call, func_args = func_arg_dict[constants.FUNC_KW], func_arg_dict[constants.ARGS_KW]
                            # print(func_call)
                            # print(func_args)
                            if( isinstance(func_call, ast.Attribute) ): 
                                call_dic  = func_call.__dict__
                                call_name = call_dic[constants.ATTRIB_KW] 
                                # print(call_name)
                                check_assert_block = "assert" in call_name
                            # print("@#@#@#", check_assert_block)
                            if (check_assert_block) :
                                call_arg_list = []
                                index1 = 0                
                                for y_ in range(len(func_args)):
                                    index1 = y_ + 1
                                    func_arg = func_args[y_] 
                                    # print(func_arg)
                                    if( isinstance(func_arg, ast.Name ) )  :
                                        # print("%%%%", func_arg)
                                        call_arg_list.append( (  func_arg.id )  )
                                    elif( isinstance( func_arg, ast.Call ) ):
                                        # print("%%%%####", func_arg)
                                        func_arg_dict  = func_arg.__dict__
                                        # print(func_arg_dict)
                                        func_, funcArgs =  func_arg_dict[ constants.FUNC_KW ], func_arg_dict[constants.ARGS_KW]
                                        if( isinstance(func_, ast.Name ) ) :        
                                            for z_ in range(len(funcArgs)):
                                                func_arg = funcArgs[z_] 
                                               #  print("&&&&&&&&&&&&", func_arg.id)
                                            if isinstance(func_arg, ast.Name):
                                                call_arg_list.append( ( func_.id + "(" + func_arg.id + ")"  ) )
                                    elif isinstance(func_arg, ast.Subscript):
                                        funcArg =  func_arg.value
                                        if isinstance(funcArg, ast.Name):
                                            func_arg = funcArg.id 
                                        elif isinstance(funcArg, ast.Subscript):
                                            func_arg = funcArg.value 
                                        call_arg_list.append( ( func_arg ) )
                    				
                                # print(call_arg_list)
                if (check_assert_block) :
                    func_list.append( ( func_name , funcLineNo, call_name, call_arg_list  ) )    
                else:
                    func_list.append( ( func_name , funcLineNo) )     
    # print("list  ", func_list)         
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
    
    
def getFunctionAssignmentsWithLHS(class_body):
    func_list = []
    lhs = ''
    for stmt_ in class_body:
        for node_ in ast.walk(stmt_):
            if isinstance(node_, ast.Assign):
                assign_dict = node_.__dict__
                targets, value  =  assign_dict[ constants.TARGETS_KW ], assign_dict[ constants.VALUE_KW ]
                for target in targets:
                    if( isinstance(target, ast.Name) ):
                        lhs = target.id
                if isinstance(value, ast.Call):
                    funcDict = value.__dict__ 
                    funcName, funcArgs, funcLineNo, funcKeys =  funcDict[ constants.FUNC_KW ], funcDict[ constants.ARGS_KW ], funcDict[constants.LINE_NO_KW], funcDict[constants.KEY_WORDS_KW]  
                    if( isinstance(funcName, ast.Name ) ): 
                        func_name = funcName.id
                        func_list.append((lhs, func_name)) 
                    elif( isinstance( funcName, ast.Attribute ) ):  
                    	func_name_dict  = funcName.__dict__
                    	func_name = func_name_dict[constants.ATTRIB_KW] 
                    	func_list.append((lhs, func_name))  
    return func_list


def checkForUnitTestImport(pyTree):
    import_list = getImport(pyTree)
    if ('unittest' in import_list):
        return True
    return False
        
def checkForLibraryImport(pyTree):
    import_list = getImport(pyTree)
    if (constants.TENSOR_LIB in import_list or constants.KERAS_LIB in import_list or constants.TORCH_LIB in import_list or constants.SKLEARN_LIB in import_list):
        print("import list", import_list)
        return True
    return False
    

def checkForMetricImport(pyTree):
    import_list = getImport(pyTree)
    if any(item in constants.all_possible_metric for item in import_list):
        return True
    return False


def getTestNames(pyTree):
    func_list = []
    unit_test_import = checkForUnitTestImport(pyTree)
    if unit_test_import:
        func_list = getFunctionDetailsForClaases(pyTree, 1, 0, 0) 
    print("func list  ", func_list)  
    return func_list
    

def checkAlgoNames(func_list):
    algo_list = []
    for item in func_list:
        if item in constants.all_possible_algo:
            algo_list.append(item)
    return algo_list
    

def checkMetricNames(func_list):
    metric_list = []
    for item in func_list:
        if item in constants.all_possible_metric:
            metric_list.append(item)
    return metric_list
    
    
def checkmetricLHSNames(func_list_with_lhs):
    metric_lhs_list = []
    for item in func_list_with_lhs:
        if item[1] in constants.all_possible_metric:
            metric_lhs_list.append(item[0])
    return metric_lhs_list
    
    
def getClassificationAlgoNames(pyTree):
    algo_list = []
    library_import = checkForLibraryImport(pyTree)
    if library_import:
        func_list = getFunctionDetailsForClaases(pyTree, 0, 1, 0) 
        print("pre algo list  ", func_list)  
        algo_list = checkAlgoNames(func_list)
    print("algo list  ", algo_list)  
    return algo_list
    
    
def getMetricNames(pyTree):
    metric_list = []
    metric_import = checkForMetricImport(pyTree)
    if metric_import:
        func_list = getFunctionDetailsForClaases(pyTree, 0, 1, 0) 
        metric_list = checkMetricNames(func_list)
    print("metric list  ", metric_list)  
    return metric_list


def getmetricLHSNames(pyTree):
    metric_lhs_list = []
    metric_import = checkForMetricImport(pyTree)
    if metric_import:
        func_list_with_lhs = getFunctionDetailsForClaases(pyTree, 0, 0, 1) 
        metric_lhs_list = checkmetricLHSNames(func_list_with_lhs)
    print("metric lhs list ", metric_lhs_list)  
    return metric_lhs_list