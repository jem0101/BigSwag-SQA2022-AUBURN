from qgis.core import *
from logger import log

from d3MapRenderer.projections import orthographic

class labeling(object):
    """Helper class for labeling"""  
    
    def __init__(self, layer, index):
        """Retrieve any label settings
        
        :param layer: The QgsVectorLayer to examine for label settings 
        :type layer: QgsVectorLayer   
        
        :param index: The index of the layer in the array of layers for output 
        :type index: int   
        
        """
        self.__logger = log(self.__class__.__name__)
        
        self.layer = layer
        self.index = index
        
        self.enabled = self.getBooleanProperty("labeling/enabled")
        self.fieldName = self.layer.customProperty("labeling/fieldName", u"")
        self.fontFamily = self.layer.customProperty("labeling/fontFamily", u"")
        self.fontBold = self.getBooleanProperty("labeling/fontBold")
        self.fontItalic = self.getBooleanProperty("labeling/fontItalic")
        self.fontStrikeout = self.getBooleanProperty("labeling/fontStrikeout")
        self.fontUnderline = self.getBooleanProperty("labeling/fontUnderline")  
        self.fontSize = float(self.layer.customProperty("labeling/fontSize", "10")) 
        """At present QGIS does not allow the text outline to have a border
        Just set to 0.1 for now in order for advanced users to style it separately"""
        self.strokeWidth = 0.1
  
        self.textColorA = int(self.layer.customProperty("labeling/textColorA", "255"))
        self.textColorB = int(self.layer.customProperty("labeling/textColorB", "0"))
        self.textColorG = int(self.layer.customProperty("labeling/textColorG", "0"))
        self.textColorR = int(self.layer.customProperty("labeling/textColorR", "0"))
        self.textTransparency = float(self.layer.customProperty("labeling/textTransp", "0")) 

        self.bufferDraw = self.getBooleanProperty("labeling/bufferDraw")
        self.bufferColorA = int(self.layer.customProperty("labeling/bufferColorA", "255"))
        self.bufferColorB = int(self.layer.customProperty("labeling/bufferColorB", "255"))
        self.bufferColorG = int(self.layer.customProperty("labeling/bufferColorG", "255"))
        self.bufferColorR = int(self.layer.customProperty("labeling/bufferColorR", "255"))
        self.bufferTransparency = float(self.layer.customProperty("labeling/bufferTransp", "0"))
        self.bufferSize = float(self.layer.customProperty("labeling/bufferSize", "1"))

        self.shadowDraw = self.getBooleanProperty("labeling/shadowDraw")
        self.shadowColorB = int(self.layer.customProperty("labeling/shadowColorB", "0"))
        self.shadowColorG = int(self.layer.customProperty("labeling/shadowColorG", "0"))
        self.shadowColorR = int(self.layer.customProperty("labeling/shadowColorR", "0"))
        self.shadowTransparency = float(self.layer.customProperty("labeling/shadowTransparency", "30"))
        self.shadowOffsetAngle = int(self.layer.customProperty("labeling/shadowOffsetAngle", "135"))
        self.shadowOffsetDist = int(self.layer.customProperty("labeling/shadowOffsetDist", "1"))
        self.shadowRadius = float(self.layer.customProperty("labeling/shadowRadius", "1.5"))
        
        self.placement = int(self.layer.customProperty("labeling/placement", "0"))
        self.quadOffset = int(self.layer.customProperty("labeling/quadOffset", "4"))
        
        self.isExpression = self.getBooleanProperty("labeling/isExpression")

    def getBooleanProperty(self, prop):
        """Not all booleans are treated equally"""
        val = False
        
        try:
            # A real boolean?
            val = (self.layer.customProperty(prop, False) == True)
        except AttributeError:
            try:
                # A text value for a boolean?
                val = (self.layer.customProperty(prop, u"").lower() == "true")
            except AttributeError:
                self.__logger.info("No idea what the value for {0} is".format(prop))
                pass
            pass
    
        return val

    def hasLabels(self):
        """Does the layer have labels enabled and a field specified?
        At the moment expressions are not supported"""
        return self.enabled == True and self.fieldName != u"" and self.isExpression == False

 
    def zoomLabelScript(self, safeCentroid):
        """The script to resize SVG text elements
        
        :param safeCentroid: Check whether the label is clipped on the other side of the globe? 
                             Orthographic projection requires no resizing of labels, 
                             but does require labels to be shown / hidden depending on the rotation of the globe 
        :type safeCentroid: bool 
        """
        if safeCentroid == True:        
            return """label{i}.each(function(d, i) {{
        var centroid = getSafeCentroid(d);    
        var label = d3.select(label{i}[0][i]);            
        //console.log(label.text(), i, clipped, path.centroid(d));            
        if (centroid[0] == 0 && centroid[1] == 0) {{
          label.style("display", "none");
        }} else {{
          label.style("display", null);
        }}
        label.attr("transform", "translate(" + centroid + ")");
      }});\n""".format(i = self.index)
      
        else:
            return """label{i}.style("stroke-width", {sw} / d3.event.scale);
      label{i}.style("font-size", labelSize({ls}, d3.event.scale)  + "pt");\n""".format(
                                                                                        i = self.index,
                                                                                        sw = self.strokeWidth,
                                                                                        ls = self.fontSize)
      
    def getLabelObjectScript(self, safeCentroid):
        """The Javascript for creating the SVG text elements
        
        :param safeCentroid: Check whether the label is clipped on the other side of the globe
                             Othrographic version to  return the Javascript for creating the SVG text elements
                             Orthographic projections may have labels clipped from view
        :type safeCentroid: bool """
        
        centroid = "path.centroid(d)"
        if safeCentroid == True:
            centroid = "getSafeCentroid(d)"        
        
        return """      label{i} = vectors{i}.selectAll("text").data(object{i}.features);
      label{i}.enter()
        .append("text")
        .attr("transform", function(d){{ var centroid = {cent}; return "translate(" + centroid + ")"; }})
        .style("display", function(d){{ var centroid = {cent}; return (centroid[0] == 0 && centroid[1] == 0) ? "none": null; }})
        .text(function(d) {{ return d.properties.{f}; }})
        .attr("class", "label{i}");\n""".format(
                                                  i = self.index,
                                                  f = self.fieldName,
                                                  cent = centroid)

    
    def getStyle(self):
        """Convert the label settings to CSS3"""
        if self.hasLabels() == True:
            return """.label{i}{{ 
    pointer-events: none; 
    {fill}
    {fontFamily}
    {fontSize} 
    {fontWeight}
    {fontStyle}
    {textDecoration}
    {fillOpacity} 
    {stroke}
    {strokeWidth} 
    {strokeOpacity}
    {textAnchor}
    {alignmentBaseline}
    {textShadow}
}}""".format(              
           i = self.index,
           fill = self.getFill(),
           fontFamily = self.getFontFamily(),
           fontSize = self.getFontSize(),
           fontWeight = self.getFontWeight(),
           fontStyle = self.getFontStyle(),
           textDecoration = self.getTextDecoraction(),
           fillOpacity = self.getFillOpacity(),
           stroke = self.getStroke(),
           strokeWidth = self.getStrokeWidth(),
           strokeOpacity = self.getStrokeOpacity(),
           textAnchor = self.getTextAnchor(),
           alignmentBaseline = self.getAlignmentBaseline(),
           textShadow = self.getTextShadow(),
           )
        
        else:
            return ""


    def getFill(self): 
        template = "fill: rgba({r},{g},{b},{a});"
        
        return template.format(
                               r = str(self.textColorR),
                               g = str(self.textColorG),
                               b = str(self.textColorB),
                               a = str(self.textColorA)
                               )
        
    def getFillOpacity(self):
        template = "fill-opacity: {0};"
        
        return template.format(str(self.getOpacity(self.textTransparency)))
        
    def getOpacity(self, transparency):
        """Get the opacity value in a range between 1.0 (opaque) to 0 (transparent)
        Rather than as a percentage of transparency""" 
        return (100 - transparency) / 100  
        
    def getAlphaOpacity(self, alpha, transparency):
        """Get the opacity based on the color alpha and a transparency percentage"""       
        return str(alpha * self.getOpacity(transparency))
    
    def getFontFamily(self):
        template = "font-family: {0};"
        
        if " " in self.fontFamily:
            self.fontFamily = "'{0}'".format(self.fontFamily)
    
        return template.format(self.fontFamily)
    
    def getFontSize(self):
        template = "font-size: {0}pt;"
        
        return template.format(str(self.fontSize))
    
    def getFontWeight(self):
        if self.fontBold == True:
            return "font-weight: bold;"
        else:
            return "font-weight: normal;"    

    def getFontStyle(self):
        if self.fontItalic == True:
            return "font-style: italic;"
        else:
            return "font-style: normal;"  
        
    def getTextDecoraction(self):
        if self.fontStrikeout == True or self.fontUnderline == True:
            template = "text-decoration:{0}{1};";
            underline = " underline" if self.fontUnderline == True  else ""
            strikeout = " line-through" if self.fontStrikeout == True  else ""
            
            return template.format(underline, strikeout)
        
        else:
            return ""
        
    def getStroke(self): 
        template = "stroke: rgba({r},{g},{b},{a});"
        
        return template.format(
                               r = str(self.textColorR),
                               g = str(self.textColorG),
                               b = str(self.textColorB),
                               a = str(self.textColorA)
                               )
        
    def getStrokeWidth(self):
        """Get the text outline border"""
        return "stroke-width = {0};".format(str(self.strokeWidth))  
        
    def getStrokeOpacity(self):
        template = "stroke-opacity: {0};"
        
        return template.format(str(self.getOpacity(self.textTransparency)))        

    def getTextAnchor(self):
        """Text alignment Left -> Right
        QGIS labels have placement values of 0 = 'Around Centroid' aka centered, and 1 = 'Offset from Centroid'
        When using 'Offset from Centroid' the position can be top-left trough bottom-right with values such as:
        0 1 2
        3 4 5
        6 7 8"""
        template = "text-anchor: {0};"
        
        if self.placement == 1:            
            position = ["end", "middle", "start", "end", "middle", "start", "end", "middle", "start"]
            
            return template.format(position[self.quadOffset])
        else:
            return template.format("middle")
        
        
    def getAlignmentBaseline(self):
        """Text alignment Top -> Bottom
        QGIS labels have placement values of 0 = 'Around Centroid' aka centered, and 1 = 'Offset from Centroid'
        When using 'Offset from Centroid' the position can be top-left trough bottom-right with values such as:
        0 1 2
        3 4 5
        6 7 8"""
        template = "alignment-baseline: {0};"
        if self.placement == 1:
            position = ["alphabetic", "alphabetic", "alphabetic", "middle", "middle", "middle", "hanging", "hanging", "hanging"]
            
            return template.format(position[self.quadOffset])
        else:
            return template.format("middle") 
        
    def getTextShadow(self):
        """Get a text shadow to display any buffer and drop shadow implemented in the label
        NOTE: QGIS settings do not directly map onto CSS attributes""" 
        output = []
        
        if self.bufferDraw == True or self.shadowDraw == True:
            output.append("text-shadow: ")
            template = "{x}px {y}px {blur}px rgba({r},{g},{b},{a})"
                        
            if self.bufferDraw == True:
                output.append(template.format(
                                              x = "1",
                                              y = "1",
                                              blur = str(self.bufferSize),
                                              r = self.bufferColorR,
                                              g = self.bufferColorG,
                                              b = self.bufferColorB,
                                              a = self.getAlphaOpacity(255, self.bufferTransparency)))
                output.append(", ") 
                output.append(template.format(
                                              x = "-1",
                                              y = "1",
                                              blur = str(self.bufferSize),
                                              r = self.bufferColorR,
                                              g = self.bufferColorG,
                                              b = self.bufferColorB,
                                              a = self.getAlphaOpacity(255, self.bufferTransparency)))
                output.append(", ") 
                output.append(template.format(
                                              x = "-1",
                                              y = "-1",
                                              blur = str(self.bufferSize),
                                              r = self.bufferColorR,
                                              g = self.bufferColorG,
                                              b = self.bufferColorB,
                                              a = self.getAlphaOpacity(255, self.bufferTransparency)))
                output.append(", ") 
                output.append(template.format(
                                              x = "1",
                                              y = "-1",
                                              blur = str(self.bufferSize),
                                              r = self.bufferColorR,
                                              g = self.bufferColorG,
                                              b = self.bufferColorB,
                                              a = self.getAlphaOpacity(255, self.bufferTransparency)))
                                
            if self.shadowDraw == True:
                if self.bufferDraw == True:
                    #text-shadow attributes are CSV
                    output.append(", ")    
                #CSS and QGIS differ on the starting angle. 
                #QGIS stores angles as 0 through 180 with negative values for the LHS of the compass
                angle = (360 +self.shadowOffsetAngle) if self.shadowOffsetAngle < 0 else self.shadowOffsetAngle
                
                #CSS is +90 degrees, starting at East on the compass
                angle = angle -90 if (angle - 90) > -1 else 360 + (angle - 90)
                
                posX = 2
                posY = 0
                if angle <= 10:
                    posX = 2
                    posY = 0
                elif angle >= 10 and angle < 45:
                    posX = 2
                    posY = 1
                elif angle >= 45 and angle < 55:
                    posX = 1
                    posY = 1
                elif angle >= 55 and angle < 90:
                    posX = 1
                    posY = 2
                elif angle >= 90 and angle < 125:
                    posX = 0
                    posY = 2
                elif angle >= 125 and angle < 135:
                    posX = -1
                    posY = 2
                elif angle >= 135 and angle < 140:
                    posX = -1
                    posY = 1
                elif angle >= 140 and angle < 170:
                    posX = -2
                    posY = 1
                elif angle >= 170 and angle < 185:
                    posX = -2
                    posY = 0
                elif angle >= 185 and angle < 225:
                    posX = -2
                    posY = -1
                elif angle >= 225 and angle < 235:
                    posX = -1
                    posY = -1
                elif angle >= 235 and angle < 260:
                    posX = -1
                    posY = -2
                elif angle >= 260 and angle < 285:
                    posX = 0
                    posY = -2                    
                elif angle >= 285 and angle < 315:
                    posX = 1
                    posY = -1
                elif angle >= 315 and angle < 350:
                    posX = 2
                    posY = -1
                else:
                    posX = 2
                    posY = 0
                    
                output.append(template.format(
                                              x = str(posX * self.shadowOffsetDist),
                                              y = str(posY * self.shadowOffsetDist),
                                              blur = str(self.shadowRadius * 3),
                                              r = self.shadowColorR,
                                              g = self.shadowColorG,
                                              b = self.shadowColorB,
                                              a = self.getAlphaOpacity(255, self.shadowTransparency)
                                              ))

                
          
                              
            output.append(";")
        
        return "".join(output)        
