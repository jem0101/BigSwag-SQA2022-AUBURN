class FunctionDecomposition:
    def __init__( self, name, parameters, text, returnType, returnRValue, static, const, templatePrefix = "",
            virtual = False ):
        self.name = name
        self.parameters = parameters
        self.text = text
        self.returnType = returnType
        self.returnRValue = returnRValue
        self.static = static
        self.templatePrefix = templatePrefix
        self.const = const
        self.virtual = virtual

    def parametersFullSpec( self ):
        return ", ".join( [ p[ 'text' ] for p in self.parameters ] )

    def parametersForwardingList( self ):
        return ", ".join( [ p[ 'name' ] for p in self.parameters ] )

    def returnTypeIsVoid( self ):
        return self.returnType == "void"

    def stringReturnIfNotVoid( self ):
        return "" if self.returnTypeIsVoid() else "return "

    def stringStaticIfStatic( self ):
        return "static " if self.static else ""

    def stringStaticInlineIfStatic( self ):
        return "static inline " if self.static else ""
    def stringVirtualIfVirtual( self ):
        return "virtual" if self.virtual else ""
