import os
from qgis.core import *

from logger import log
from style import css3

class renderers(list):
    """List of data ranges per renderer within a particular layer. 
      May only contain a single renderer, or multiple graduated or categorized renderers"""
      
    """ TODO: At the moment only supports one symbol per renderer data range. However
          Q supports multiple and nested symbols to provide an overall effect. This will be difficult 
          to present on the web in a performant way, but leave the code ready to tackle this later"""
    
    def __init__(self):
        """Constructor. Nothing special here"""
        self.__logger = log(self.__class__.__name__) 
        
    def getAvergageOutlineWidth(self):
        """Retrieve the average outline width
        In theory each symbol could have a different outline width
        Zooming on the client requires scaling of the Svg border to maintain clarity
        However, re-scaling on each polygon/line/point is too costly
        Take the average width and do it to the entire group"""
        items = []
        for r in self:
            items.append(r.symbols[0].outlineWidth)
        average = sum(items) / float(len(items)) 
        return round(average, 4)
    
    def logData(self):
        """Log the symbols in the list"""
        
        template = u"""       {type}
                  geotype = [{geometryType}],  size =  [{size}],  color = [{color}],  colorTrans = [{colorTrans}]
                  outlineWidth = [{outlineWidth}],  outlineColor = [{outlineColor}],  outlineStyle = [{outlineStyle}],  outlineTrans = [{outlineTrans}]
                  brushStyle = [{brushStyle}],  legendWidth = [{legendWidth}],  legendHeight = [{legendHeight}]"""
        
        for r in self:
            self.__logger.info(template.format(
                type = type(r.symbols[0]),
                geometryType = r.symbols[0].geometryType,
                size = r.symbols[0].size,
                color = r.symbols[0].color,
                colorTrans = r.symbols[0].colorTrans,
                outlineWidth = r.symbols[0].outlineWidth,
                outlineColor = r.symbols[0].outlineColor,
                outlineStyle = r.symbols[0].outlineStyle,
                outlineTrans = r.symbols[0].outlineTrans,
                brushStyle = r.symbols[0].brushStyle,
                legendWidth = r.symbols[0].legendWidth,
                legendHeight = r.symbols[0].legendHeight
            ))
    
class symbol(object):
    
    def __init__(self, geoType, parentSymbol, layerTransparency):
        """
        Abstract base class 
        
        :param geoType: Layer.geometryType() GeometryType of the layer.
        :type geoType: GeometryType  e.g. QGis.WKBPolygon
        
        :param parentSymbol: The parent symbol to use with this layer.
        :type parentSymbol: QgsSymbolV2
        
        :param layerTransparency: Transparency setting for the overall layer.
        :type layerTransparency: float
        """
        self.geometryType = geoType
        self.isMain = False
        self.transparency = layerTransparency
        self.cssHelper = css3()
        self.index = 0
        self.css = ""
        self.size = "20"
        self.color = "#000000"
        self.colorTrans = 255
        self.symbolTrans = 1.0
        self.outlineWidth = 0.26
        self.outlineColor = "#000000"
        self.outlineStyle = None
        self.outlineTrans = 1.0
        self.brushStyle = 1
        self.path = u""
        self.legendWidth = "20"
        self.legendHeight = "20"

    def getOpacity(self):
        """Get the opacity for the range"""
        opacity = "0"
        if self.brushStyle > 0: 
            '''Might implement different brush styles later, for now just interested in "No brush"'''
            opacity = self.cssHelper.getOpacity(self.transparency, self.colorTrans, self.symbolTrans)
        return opacity    
                
    def getOutlineOpacity(self):
        """Get the opacity for the range"""
        opacity = "0"
        if self.outlineStyle > 0: 
            '''Might implement different brush styles later, for now just interested in "No brush"'''
            opacity = self.cssHelper.getOpacity(self.transparency, self.outlineTrans, self.symbolTrans)           
        
        return opacity
    
    def getAdditionalScripts(self):
        """Get any additional JavaScript functions for the symbols to render correctly """
        
        return ""
    
    def hasImage(self):
        """Does the symbology include an external image?"""
        return False
           
    def toCss(self):
        """Retrieve the Css for the symbol"""
        return ""
       
    def toLayerScript(self, outputIndex, pattern, safeCentroid):
        """Retrieve the layer render script for d3 to use"""        
        output = pattern.format("""        .attr("d", path)\n""")
                
        return output    
    
    def zoomScalingScript(self, outputIndex, safeCentroid):
        """Retrieve the layer zoom script d3 to use
        
        :param outputIndex: int The layer order for the output

        :param safeCentroid: Check whether the label is clipped on the other side of the globe
                             Othrographic version to  return the JavaScript for creating the SVG text elements
                             Orthographic projections may have labels clipped from view
        :type safeCentroid: bool

        """
        template = "      vector{index}.style(\"stroke-width\", {width} / d3.event.scale);\n"
        
        return template.format(
                               index = outputIndex,
                               width = self.outlineWidth)
        
    def safeSvgNode(self, outputIndex, safeCentroid):
        """Retrieve any special node creation variables
        
        :param outputIndex: int The layer order for the output

        :param safeCentroid: Check whether the label is clipped on the other side of the globe
                             Othrographic version to  return the JavaScript for creating the SVG text elements
                             Orthographic projections may have labels clipped from view
        :type safeCentroid: bool

        """
        return ""
    
    def getShape(self):
        """Retrieve the d3 equivalent shape"""
        return "square"
    
    
class simpleLineSymbol(symbol):
    
    def __init__(self, geoType, parentSymbol, layerTransparency):
        """ 
        
        :param geoType: Layer.geometryType() GeometryType of the layer.
        :type geoType: GeometryType  e.g. QGis.WKBPolygon
        
        :param parentSymbol: The parent symbol to use with this layer.
        :type parentSymbol: QgsLineSymbolV2  
        
        :param layerTransparency: Transparency setting for the overall layer.
        :type layerTransparency: float
        """
        self.__logger = log(self.__class__.__name__)
        self.isMain = False
        self.geometryType = geoType
        self.transparency = layerTransparency
        self.cssHelper = css3()
        self.index = 0
        self.css = "" 
        self.size = ""
        self.color = "#000000"
        self.colorTrans = 255
        self.symbolTrans = 1.0
        self.outlineWidth = 0.26
        self.outlineColor = "#000000"
        self.outlineStyle = None
        self.outlineTrans = 1.0
        self.brushStyle = 1
        self.legendWidth = "20"
        self.legendHeight = "20"
        
        self.readStyles(parentSymbol)

       
    def readStyles(self, parentSymbol):
        """Read the styles properties from the layer and store for later use
        
        :param parentSymbol: The parent symbol to use with this layer.
        :type parentSymbol: QgsSymbolV2        
        """
            
        try:
            self.color = parentSymbol.color().name()
            self.colorTrans = self.cssHelper.convertColorTransToCssOpacity(parentSymbol.color().alpha())
            self.symbolTrans = parentSymbol.alpha()
            self.outlineWidth = parentSymbol.symbolLayer(0).width() * 4
            self.outlineColor = parentSymbol.symbolLayer(0).color().name()
            self.outlineTrans = self.cssHelper.convertColorTransToCssOpacity(parentSymbol.symbolLayer(0).color().alpha())
            self.outlineStyle = parentSymbol.symbolLayer(0).penStyle()
            self.size = str(self.outlineWidth)
            self.legendHeight = self.size
            
        except (AttributeError, TypeError):
            self.__logger.error2()
            pass    
        
    def toCss(self):
        """Retrieve the Css for line symbols"""
        val = ".{c} {{ {m}stroke: {s}; stroke-width: {w}; stroke-opacity: {o}; stroke-dasharray: {d}; fill-opacity: 0.0; }}"
        output = val.format(
            m = "" if self.isMain == True else "pointer-events: none; ",
            c = self.css,
            s = unicode(self.outlineColor),
            w = unicode(self.outlineWidth),
            d = self.cssHelper.getBorderStyle(self.outlineStyle),
            o = self.getOpacity())  
        
        self.__logger.info(output)
        
        return output    
    
    def getShape(self):
        """Retrieve the d3 equivalent shape"""
        return "line"

class simpleMarkerSymbol(symbol):
    
    def __init__(self, geoType, parentSymbol, layerTransparency):
        """ 
        
        :param geoType: Layer.geometryType() GeometryType of the layer.
        :type geoType: GeometryType  e.g. QGis.WKBPolygon
        
        :param parentSymbol: The parent symbol to use with this layer.
        :type parentSymbol: QgsMarkerSymbolV2
        
        :param layerTransparency: Transparency setting for the overall layer.
        :type layerTransparency: float
        """
        self.__logger = log(self.__class__.__name__)
        self.isMain = False
        self.geometryType = geoType
        self.transparency = layerTransparency
        self.cssHelper = css3()
        self.index = 0
        self.css = ""
        self.size = "8"            
        self.name = "circle"
        self.color = "#000000"
        self.colorTrans = 255
        self.symbolTrans = 1.0
        self.outlineWidth = 0.26
        self.outlineColor = "#000000"
        self.outlineStyle = None
        self.outlineTrans = 1.0
        self.brushStyle = 1
        self.legendWidth = "20"
        self.legendHeight = "20"
        
        self.readStyles(parentSymbol)
        
    def readStyles(self, parentSymbol):
        """Read the styles properties from the layer and store for later use
        
        :param parentSymbol: The parent symbol to use with this layer.
        :type parentSymbol: QgsSymbolV2        
        """        
        try:
            self.color = parentSymbol.color().name()
            self.colorTrans = self.cssHelper.convertColorTransToCssOpacity(parentSymbol.color().alpha())
            self.symbolTrans = parentSymbol.alpha()
            '''No width property in a simple marker
            self.outlineWidth = parentSymbol.width()'''
            ''' d3 symbols are sized 64 by default '''
            self.size = str(parentSymbol.size() * 10)           
            self.outlineColor = parentSymbol.symbolLayer(0).outlineColor().name()
            self.outlineTrans = self.cssHelper.convertColorTransToCssOpacity(parentSymbol.symbolLayer(0).outlineColor().alpha())
            self.outlineStyle = parentSymbol.symbolLayer(0).outlineStyle()
            try:
                self.name = parentSymbol.symbolLayer(0).name()
            except (AttributeError, TypeError):
                self.__logger.error2()
                '''No name property on an Ellipse marker, but there is a symbolName, try that
                QgsEllipseSymbolLayerV2 is a subclass of QgsMarkerSymbolLayerV2 so we could end up in this code'''
                self.name =parentSymbol.symbolLayer(0).symbolName()
            self.legendHeight = self.size
            
        except (AttributeError, TypeError):
            self.__logger.error2()
            pass
        
    def toCss(self):
        """Get the style for points"""
        
        val = ".{c} {{ {m}stroke: {s}; stroke-width: {w}; stroke-opacity: {so}; stroke-dasharray: {d}; fill: {f}; fill-opacity: {fo}; }}"
        
        output = val.format(
            m = "" if self.isMain == True else "pointer-events: none; ",
            c = self.css,
            s = unicode(self.outlineColor),
            w = unicode(self.outlineWidth),
            d = self.cssHelper.getBorderStyle(self.outlineStyle),
            f = unicode(self.color),
            fo = self.getOpacity(),
            so = self.getOutlineOpacity()) 
        
        self.__logger.info(output)
        
        return output   

    def toLayerScript(self, outputIndex, pattern, safeCentroid):
        """Retrieve the layer render script for d3 to use
        
        :param outputIndex: int The layer order for the output
        
        :param pattern: The string.format to use as the layer JavaScript.
        :type pattern: string
        
        :param safeCentroid: Check whether the label is clipped on the other side of the globe
                             Othrographic version to  return the JavaScript for creating the SVG text elements
                             Orthographic projections may have labels clipped from view
        :type safeCentroid: bool

        """
        
        centroid = "path.centroid(d)"
        if safeCentroid == True:
            centroid = "getSafeCentroid(d)"
        
        val = """        .attr("d", d3.svg.symbol().type("{sym}").size( function(d) {{ return d.properties.d3S; }}  ))
        .attr("transform", function(d) {{ var centroid = {cent}; return "translate(" + (centroid[0] - (d.properties.d3S / projection.scale() / 2)) + "," + (centroid[1] - (d.properties.d3S / projection.scale() / 2)) + ")";}})\n"""
                           
        inner = val.format(
                           sym = self.getShape(),
                           cent = centroid)         
        
        
        output = pattern.format(inner)
        
        self.__logger.info(output)
        
        return output
    
    def zoomScalingScript(self, outputIndex, safeCentroid):
        """Retrieve the layer zoom script d3 to use
        
        :param outputIndex: int The layer order for the output

        :param safeCentroid: Check whether the label is clipped on the other side of the globe
                             Othrographic version to  return the JavaScript for creating the SVG text elements
                             Orthographic projections may have labels clipped from view
        :type safeCentroid: bool

        """
        centroid = "path.centroid(d)"
        if safeCentroid == True:
            centroid = "getSafeCentroid(d)"
            
        template = """      vector{i}.each(function(d, i) {{
        var centroid = {cent};    
        var g = d3.select(vector{i}[0][i]);           
        if (centroid[0] == 0 && centroid[1] == 0) {{
          g.style("display", "none");
        }} else {{
          g.style("display", null);
        }}
        g.attr("transform", "translate(" + (centroid[0] - (d.properties.d3S / projection.scale() / 2)) + "," + (centroid[1] - (d.properties.d3S / projection.scale() / 2)) + ")")        
         .attr("d", d3.svg.symbol().type("{sym}").size( function(d) {{ return d.properties.d3S; }}  ));
      }});\n"""
        
        return template.format(
                               i = outputIndex,
                               sym = self.getShape(),
                               cent = centroid)
        
    def getShape(self):
        """Retrieve the d3 equivalent shape"""
        d3Shape = "circle"
        
        if self.name == "cross":
            d3Shape = "cross"
        elif self.name == "rectangle":
            d3Shape = "square"
        elif self.name == "diamond":
            d3Shape = "diamond"        
        elif self.name == "triangle":
            d3Shape = "triangle-up" 
        elif self.name == "equilateral_triangle":
            d3Shape = "triangle-up" 
        else:
            d3Shape = "circle"
            
        return d3Shape
       
class simpleFillSymbol(symbol):
    
    def __init__(self, geoType, parentSymbol, layerTransparency):
        """ 
        
        :param geoType: Layer.geometryType() GeometryType of the layer.
        :type geoType: GeometryType  e.g. QGis.WKBPolygon
        
        :param parentSymbol: The parent symbol to use with this layer.
        :type parentSymbol: QgsFillSymbolV2
        
        :param layerTransparency: Transparency setting for the overall layer.
        :type layerTransparency: float
        """
        self.__logger = log(self.__class__.__name__)
        self.isMain = False
        self.geometryType = geoType
        self.transparency = layerTransparency
        self.cssHelper = css3()
        self.index = 0
        self.css = ""
        self.size = "20"
        self.color = "#000000"
        self.colorTrans = 255
        self.symbolTrans = 1.0
        self.outlineWidth = 0.26
        self.outlineColor = "#000000"
        self.outlineStyle = None
        self.outlineTrans = 1.0
        self.brushStyle = 1
        self.legendWidth = "20"
        self.legendHeight = "20"
        
        self.readStyles(parentSymbol)
       
    def readStyles(self, parentSymbol):
        """Read the styles properties from the layer and store for later use
        
        :param parentSymbol: The parent symbol to use with this layer.
        :type parentSymbol: QgsSymbolV2        
        """        
        try:
            self.color = parentSymbol.color().name()
            self.colorTrans = self.cssHelper.convertColorTransToCssOpacity(parentSymbol.color().alpha())
            self.symbolTrans = parentSymbol.alpha()
            self.outlineWidth = parentSymbol.symbolLayer(0).borderWidth() * 4
            self.outlineColor = parentSymbol.symbolLayer(0).borderColor().name()
            self.outlineTrans = self.cssHelper.convertColorTransToCssOpacity(parentSymbol.symbolLayer(0).borderColor().alpha())
            self.outlineStyle = parentSymbol.symbolLayer(0).borderStyle()
        except (AttributeError, TypeError):
            self.__logger.error2()
            pass    
   
    def toCss(self):
        """Get the style for simple polygons"""
        val = ".{c} {{ {m}stroke: {s}; stroke-width: {w}; stroke-opacity: {so}; stroke-dasharray: {d}; fill: {f}; fill-opacity: {fo}; }}"
        
        output = val.format(
            m = "" if self.isMain == True else "pointer-events: none; ",
            c = self.css,
            s = unicode(self.outlineColor),
            w = unicode(self.outlineWidth),
            d = self.cssHelper.getBorderStyle(self.outlineStyle),
            f = unicode(self.color),
            fo = self.getOpacity(),
            so = self.getOutlineOpacity()) 
        
        self.__logger.info(output)
        
        return output 
    
    def getShape(self):
        """Retrieve the d3 equivalent shape
        Just a basic rectangle shape. Needs differentiating from the rect symbol"""        
        return "poly"
        
class svgMarkerSymbol(symbol):
    
    def __init__(self, geoType, parentSymbol, layerTransparency):
        """ 
        
        :param geoType: Layer.geometryType() GeometryType of the layer.
        :type geoType: GeometryType  e.g. QGis.WKBPolygon
        
        :param parentSymbol: The parent symbol to use with this layer.
        :type parentSymbol: QgsMarkerSymbolV2
        
        :param layerTransparency: Transparency setting for the overall layer.
        :type layerTransparency: float
        """
        self.__logger = log(self.__class__.__name__)
        self.isMain = False
        self.geometryType = geoType
        self.transparency = layerTransparency
        self.cssHelper = css3()
        self.index = 0
        self.css = ""
        self.size = "8"            
        self.path = u""
        self.color = "#000000"
        self.colorTrans = 255
        self.symbolTrans = 1.0
        self.outlineWidth = 0.26
        self.outlineColor = "#000000"
        # No outline style set by this symbol, force a solid border
        self.outlineStyle = 1
        self.outlineTrans = 1.0
        self.brushStyle = 1
        self.legendWidth = "20"
        self.legendHeight = "20"
        
        self.readStyles(parentSymbol)
        
        
    def readStyles(self, parentSymbol):
        """Read the styles properties from the layer and store for later use
        
        :param parentSymbol: The parent symbol to use with this layer.
        :type parentSymbol: QgsSymbolV2        
        """        
        try:           
            self.color = parentSymbol.color().name()
            self.colorTrans = self.cssHelper.convertColorTransToCssOpacity(parentSymbol.color().alpha())
            self.symbolTrans = parentSymbol.alpha()
            self.outlineWidth = parentSymbol.symbolLayer(0).outlineWidth() * 4
            self.size = str(parentSymbol.size() * 3)           
            self.outlineColor = parentSymbol.symbolLayer(0).outlineColor().name()
            self.outlineTrans = self.cssHelper.convertColorTransToCssOpacity(parentSymbol.symbolLayer(0).outlineColor().alpha())
            self.path = parentSymbol.symbolLayer(0).path()
            self.legendWidth = self.size
            self.legendHeight = self.size
            
        except (AttributeError, TypeError):
            self.__logger.error2()
            pass
    
    def getAdditionalScripts(self):
        """Get any additional JavaScript functions for the symbols to render correctly """
        
        return """    function loadImage(parent, child, css, size){
      var str = "{0}px".replace("{0}", size == "null" ? 0 : size);
      var n = parent.appendChild(child.cloneNode(true)); 
      d3.select(n)
        .attr("width", str)
        .attr("height", str)
        .attr("x", -size / 2)
        .attr("y", -size / 2)
        .attr("class", css);
    }\n""" 
        
    def hasImage(self):
        """Does the symbology include an external image?"""
        return True    
        
    def toCss(self):
        """Get the style for SVG images
        !important used to cram colour choice down the SVG XML child nodes
        Works on most well designed SVGs produced for QGIS, but not SVGs with specific colours within the XML"""

        val = ".{c} {{ {m}stroke: {s} !important; stroke-width: {w} !important; stroke-opacity: {so} !important; stroke-dasharray: {d} !important; fill: {f} !important; fill-opacity: {fo} !important; }}"
        
        output = val.format(
            m = "" if self.isMain == True else "pointer-events: none; ",
            c = self.css,
            s = unicode(self.outlineColor),
            w = unicode(self.outlineWidth),
            d = self.cssHelper.getBorderStyle(self.outlineStyle),
            f = unicode(self.color),
            fo = self.getOpacity(),
            so = self.getOutlineOpacity()) 
        
        self.__logger.info(output)
        
        return output   

    def toLayerScript(self, outputIndex, pattern, safeCentroid):
        """Retrieve the layer render script for d3 to use.
        
        :param outputIndex: int The layer order for the output
        
        :param pattern: The string.format to use as the layer JavaScript.
        :type pattern: string
        
        :param safeCentroid: Check whether the label is clipped on the other side of the globe
                             Othrographic version to  return the JavaScript for creating the SVG text elements
                             Orthographic projections may have labels clipped from view
        :type safeCentroid: bool
        
        Output for SVG marker symbols is completely different, as it requires the image to be loaded into XML
        before then being cloned and added to the html document. This is required in order to color the image
        as specified within QGIS. Previous attempts resulted in the image not being scaled correctly, or the 
        colours being ignored. e.g.:
        
        The following results in a scaled image, but does NOT reflect the chosen colors:
      vector1.enter()
        .append("image")
        .attr("xlink:href", "img/poi_tower_power.svg")
        .attr("width", function(d) { return d.properties.d3S; })
        .attr("height", function(d) { return d.properties.d3S; })
        .attr("transform", function(d) { return "translate(" + (path.centroid(d)[0] - (d.properties.d3S / projection.scale() / 2)) + "," + (path.centroid(d)[1] - (d.properties.d3S / projection.scale() / 2)) + ")"; })
        .attr("class", function (d) { return d.properties.d3Css; });
        
        But then, on Orthographic projections, d3 stamps out any d attribute of any path elements within image loaded as an SVG...
        WTF? reload the image, or try and prevent d3's natural workings 
        """
        
        val = """      vector{i} = vectors{i}.selectAll("path").data(object{i}.features);
      d3.xml("img/{svg}", "image/svg+xml", function(xml) {{  
        vector{i}Node = document.importNode(xml.documentElement, true);
        vector{i}.enter()
          .append("g")
          .attr("transform", function(d) {{ var centroid = {cent}; return "translate(" + (centroid[0] - (d.properties.d3S / projection.scale() / 2)) + "," + (centroid[1] - (d.properties.d3S / projection.scale() / 2)) + ")"; }})
          .attr("width", function(d) {{ return d.properties.d3S; }})
          .attr("height", function(d) {{ return d.properties.d3S; }})
          .each( function(d) {{ return loadImage(this, vector{i}Node, d.properties.d3Css, d.properties.d3S); }} );    
      }});\n"""

        head, tail = os.path.split(self.path)
        
        centroid = "path.centroid(d)"
        if safeCentroid == True:
            centroid = "getSafeCentroid(d)"
        
        output = val.format(
            i = outputIndex,
            svg = tail,
            cent = centroid)
                    
        self.__logger.info(output)
        
        return output        
    
    def zoomScalingScript(self, outputIndex, safeCentroid):
        """Retrieve the layer zoom script d3 to use
        
        :param outputIndex: int The layer order for the output

        :param safeCentroid: Check whether the label is clipped on the other side of the globe
                             Othrographic version to  return the JavaScript for creating the SVG text elements
                             Orthographic projections may have labels clipped from view
        :type safeCentroid: bool

        """
        centroid = "path.centroid(d)"
        imgReload = ""
        if safeCentroid == True:
            centroid = "getSafeCentroid(d)"
            imgReload = """// Remove the corrupted SVG and reload
          var img = g[0][0].childNodes[0];
          if (img != null){{
            g[0][0].removeChild(img);
          }}
          loadImage(g[0][0], vector{0}Node, d.properties.d3Css, d.properties.d3S);""".format(self.index)
            
        template = """      vector{i}.each(function(d, i) {{
        var centroid = {cent};    
        var g = d3.select(vector{i}[0][i]);           
        if (centroid[0] == 0 && centroid[1] == 0) {{
          g.style("display", "none");
        }} else {{
          g.style("display", null);
          {img}
        }}
        g.attr("transform", "translate(" + (centroid[0] - (d.properties.d3S / projection.scale() / 2)) + "," + (centroid[1] - (d.properties.d3S / projection.scale() / 2)) + ")");
      }});\n"""
        
        return template.format(
                               i = outputIndex,
                               cent = centroid,
                               img = imgReload)
    
    def safeSvgNode(self, outputIndex, safeCentroid):
        """Retrieve any special node creation variables
        
        :param outputIndex: int The layer order for the output

        :param safeCentroid: Check whether the label is clipped on the other side of the globe
                             Othrographic version to  return the JavaScript for creating the SVG text elements
                             Orthographic projections may have labels clipped from view
        :type safeCentroid: bool

        """
        if safeCentroid == True:
            return """    var vector{0}Node = void 0;\n""".format(outputIndex)
        else:
            return ""
        
                
class singleSymbol(object):
    """Single symbol renderer for tracking symbology within a layer"""
    
    def __init__(self, geoType, parentSymbol, layerTransparency):
        """Initialise the symbol range
                
        :param geoType: Layer.geometryType() GeometryType of the layer.
        :type geoType: GeometryType  e.g. QGis.WKBPolygon
        
        :param parentSymbol: The parent symbol to use with this layer.
        :type parentSymbol: QgsSymbolV2
        
        :param layerTransparency: Transparency setting for the overall layer.
        :type layerTransparency: float
        
        """
        self.__logger = log(self.__class__.__name__) 
        self.symbols = []
        s = self.symbolFactory(geoType, parentSymbol, layerTransparency)
        self.symbols.append(s)
        
        
    def symbolFactory(self, geoType, parentSymbol, layerTransparency):
        """Create the supported symbol type, or default to a single symbol
        
        :param geoType: Layer.geometryType() GeometryType of the layer.
        :type geoType: GeometryType  e.g. QGis.WKBPolygon
        
        :param parentSymbol: The parent symbol to use with this layer.
        :type parentSymbol: QgsSymbolV2

        
        :param layerTransparency: Transparency setting for the overall layer.
        :type layerTransparency: float
        """
        p = type(parentSymbol)
        
        if p is QgsLineSymbolV2:
            """ TODO: Currently Works if parentSymbol.symbolLayer(0) is a QgsSimpleLineSymbolLayerV2
            Consider supporting QgsMarkerLineSymbolLayerV2, QgsArrowSymbolLayer, or QgsGeometryGeneratorSymbolLayerV2"""
            return simpleLineSymbol(geoType, parentSymbol, layerTransparency)
        
        elif p is QgsMarkerSymbolV2:
            if parentSymbol.symbolLayer(0) is not None:
                #Examine the first symbolLayer
                c = type(parentSymbol.symbolLayer(0))
                
                if c is QgsSvgMarkerSymbolLayerV2:
                    return svgMarkerSymbol(geoType, parentSymbol, layerTransparency)
                
            # Drop through to simple marker symbol
            return simpleMarkerSymbol(geoType, parentSymbol, layerTransparency)
        
        elif p is QgsFillSymbolV2:
            if parentSymbol.symbolLayer(0) is not None:
                #Currently limited to the first symbolLayer
                c = type(parentSymbol.symbolLayer(0))
                
                # TODO: Implement more complex symbols
                if  c is QgsSimpleLineSymbolLayerV2: 
                    return simpleLineSymbol(geoType, parentSymbol, layerTransparency)
                else:
                    return simpleFillSymbol(geoType, parentSymbol, layerTransparency)
            
            else:
                return simpleFillSymbol(geoType, parentSymbol, layerTransparency) 
    
    def setOutputCss(self, index, cssstub):
        """Set the output index for the layer and the CSS classname stub on the symbols        
                
        :param index: Index in the layer output order (those in front have a higher index).
        :type index: int
        
        :param cssstub: The CSS class name used for styling the symbol
        :type cssstub: string
        """
        # TODO: When implementing multiple symbols per renderer each symbol will need a further index adding to the class
        for s in self.symbols:
            s.index = index
            s.css = cssstub
            
    def getFilterExpression(self, isLowest):
        """Get the filter expression for selecting features based on their attribute"""
        # Single symbols apply to every feature
        return ""
        
    def isValueInRange(self, value):
        """Is the specified value in the range? Always is for single classification"""
        return True
   
class categorized(singleSymbol):
    """Categorized renderer class"""
    
    def __init__(self, geoType, field, fieldType, category, layerTransparency):
        """Initialise the symbol range
                
        :param geoType: Layer.geometryType() GeometryType of the layer.
        :type geoType: GeometryType  e.g. QGis.WKBPolygon
        
        :param field: Name of the attribute field used in the symbology
        :type field: str
        
        :param fieldType: Type of the attribute field used in the symbology (Integer, Real, String, Date)
        :type fieldType: str
        
        :param category: The category range object.
        :type category: QgsRendererCategoryV2
        
        :param layerTransparency: Transparency setting for the overall layer.
        :type layerTransparency: float
        
        """
        self.__logger = log(self.__class__.__name__) 
        self.symbols = []
        s = self.symbolFactory(geoType, category.symbol(), layerTransparency)
        self.symbols.append(s)
        
        self.field = field
        self.fieldType = fieldType
        self.label = category.label()
        self.value = str(category.value())

        
    def getFilterExpression(self, isLowest):
        """Get the filter expression for selecting features based on their attribute"""
        if self.fieldType == "String" or self.fieldType == "Date":
            output = "\"{c}\" = '{v}'"
            return output.format(
                c = self.field,
                v = self.value)
        else:
            output = "\"{c}\" = {v}"
            return output.format(
                c = self.field,
                v = self.value)
    
    def isValueInRange(self, value):
        """Is the specified value in the range?"""
        if self.value == value:
            return True
        else:
            return False
    
class graduated(singleSymbol):
    """Graduated renderer class"""
    
    def __init__(self, geoType, field, graduation, layerTransparency):
        """Initialise the symbol range
        
        :param geoType: Layer.geometryType() GeometryType of the layer.
        :type geoType: GeometryType  e.g. QGis.WKBPolygon
        
        :param field: Name of the attribute field used in the symbology
        :type field: str
        
        :param graduation: The graduation range object.
        :type graduation: QgsRendererRangeV2
        
        :param layerTransparency: Transparency setting for the overall layer.
        :type layerTransparency: float
        
        """
        self.__logger = log(self.__class__.__name__) 
        self.symbols = []
        s = self.symbolFactory(geoType, graduation.symbol(), layerTransparency)
        self.symbols.append(s)
        
        self.field = field
        self.label = str(graduation.label())
        self.value = str(graduation.lowerValue()) + " - " + str(graduation.upperValue())
        # Range values are always numerics
        self.lowValue = graduation.lowerValue()
        self.highValue = graduation.upperValue()
                    
    def getFilterExpression(self, isLowest):
        """Get the filter expression for selecting features based on their attribute"""
        lowRange = ">"
        if isLowest == True:
            lowRange = ">="

        output = "\"{c}\" >= {l} and \"{c}\" <= {h}"
        return output.format(
            c = self.field,
            e = lowRange,
            l = self.lowValue,
            h = self.highValue)
    
    def isValueInRange(self, value):
        """Is the specified value in the range?"""
        if self.lowValue < value and self.highValue > value:
            return True
        else:
            return False