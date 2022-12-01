# -*- coding: utf-8 -*-
"""
/***************************************************************************
 d3MapRenderer
                                 A QGIS plugin
 Vector logic and data visualisation with the d3.js library.
                              -------------------
        begin                : 2015-06-17
        git sha              : $Format:%H$
        copyright            : (C) 2015 by Simon Benten
        email                : swbenten@gmail.com
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/
"""

# Initialize Qt resources from file resources.py
import os
import sys
import tempfile
import resources_rc
import traceback

from qgis.PyQt.QtCore import QSettings, QTranslator, qVersion, QCoreApplication, Qt
from qgis.PyQt.QtGui import *
from qgis.core import *
from qgis.gui import *

# Import the code for the dialog
from d3_Map_dialog import d3MapRendererDialog
from d3MapSettings import d3MapSettings
from settings import globalSettings

from models import model
from logic import process
from logger import log
from layer import vector
from tree import vectorItem, fieldItem
from d3MapRenderer import settings
from d3MapRenderer.output import topoJson, geoJson


class d3MapRenderer:
    """QGIS Plugin Implementation."""

    def __init__(self, iface):
        """Constructor.

        :param iface: An interface instance that will be passed to this class
            which provides the hook by which you can manipulate the QGIS
            application at run time.
        :type iface: QgsInterface
        """
        # Save reference to the QGIS interface
        self.iface = iface
        # initialize plugin directory
        self.plugin_dir = os.path.dirname(__file__)
        # initialize locale
        locale = QSettings().value('locale/userLocale')[0:2]
        locale_path = os.path.join(
            self.plugin_dir,
            'i18n',
            'd3MapRenderer_{}.qm'.format(locale))

        if os.path.exists(locale_path):
            self.translator = QTranslator()
            self.translator.load(locale_path)

            if qVersion() > '4.3.3':
                QCoreApplication.installTranslator(self.translator)

        # Create the dialog (after translation) and keep reference
        self.dlg = d3MapRendererDialog()
        
        # Add an additional settings dialog
        self.settingsDlg = d3MapSettings(iface)
        
        # Init global objects
        self.__logger = log(self.__class__.__name__)
        self.model = None
        self.settings = None
        self.process = None
        self.webServerUrl = ""


        # Declare instance attributes
        self.actions = []
        self.menu = self.tr(u'&d3 Map Renderer')
        self.toolbar = self.iface.addToolBar(u'd3MapRenderer')
        self.toolbar.setObjectName(u'd3MapRenderer')
        
        
    def runSettings(self):
        """Run the settings dialog"""
        self.settingsDlg.run()
       

    # noinspection PyMethodMayBeStatic
    def tr(self, message):
        """Get the translation for a string using Qt translation API.

        We implement this ourselves since we do not inherit QObject.

        :param message: String for translation.
        :type message: str, QString

        :returns: Translated version of message.
        :rtype: QString
        """
        # noinspection PyTypeChecker,PyArgumentList,PyCallByClass
        return QCoreApplication.translate('d3MapRenderer', message)


    def add_action(
        self,
        icon_path,
        text,
        callback,
        enabled_flag=True,
        add_to_menu=True,
        add_to_toolbar=True,
        status_tip=None,
        whats_this=None,
        parent=None):
        """Add a toolbar icon to the toolbar.

        :param icon_path: Path to the icon for this action. Can be a resource
            path (e.g. ':/plugins/foo/bar.png') or a normal file system path.
        :type icon_path: str

        :param text: Text that should be shown in menu items for this action.
        :type text: str

        :param callback: Function to be called when the action is triggered.
        :type callback: function

        :param enabled_flag: A flag indicating if the action should be enabled
            by default. Defaults to True.
        :type enabled_flag: bool

        :param add_to_menu: Flag indicating whether the action should also
            be added to the menu. Defaults to True.
        :type add_to_menu: bool

        :param add_to_toolbar: Flag indicating whether the action should also
            be added to the toolbar. Defaults to True.
        :type add_to_toolbar: bool

        :param status_tip: Optional text to show in a popup when mouse pointer
            hovers over the action.
        :type status_tip: str

        :param parent: Parent widget for the new action. Defaults None.
        :type parent: QWidget

        :param whats_this: Optional text to show in the status bar when the
            mouse pointer hovers over the action.

        :returns: The action that was created. Note that the action is also
            added to self.actions list.
        :rtype: QAction
        """

        icon = QIcon(icon_path)
        action = QAction(icon, text, parent)
        action.triggered.connect(callback)
        action.setEnabled(enabled_flag)

        if status_tip is not None:
            action.setStatusTip(status_tip)

        if whats_this is not None:
            action.setWhatsThis(whats_this)

        if add_to_toolbar:
            self.toolbar.addAction(action)

        if add_to_menu:
            self.iface.addPluginToWebMenu(
                self.menu,
                action)

        self.actions.append(action)

        return action

    def initGui(self):
        """Create the menu entries and toolbar icons inside the QGIS GUI."""

        icon_path = ':/plugins/d3MapRenderer/icon.png'
        self.add_action(
            icon_path,
            text=self.tr(u'd3 Map'),
            callback=self.run,
            parent=self.iface.mainWindow())
        
        setting_path = ':/plugins/d3MapRenderer/settings.png'
        self.add_action(
            setting_path,
            text=self.tr(u'Settings'),
            callback=self.runSettings,
            parent=self.iface.mainWindow())


    def unload(self):
        """Removes the plugin menu item and icon from QGIS GUI."""
        for action in self.actions:
            self.iface.removePluginWebMenu(
                self.tr(u'&d3 Map Renderer'),
                action)
            self.iface.removeToolBarIcon(action)
        # remove the toolbar
        del self.toolbar

    # Events and Validation ----------------------------------------------------
    def changedMainLayerComboBox(self):
        """Main layer has been reselected, redraw popup information"""
        self.model.setMainLayer(self.dlg.mainLayerComboBox.currentIndex())
        self.populateExtraLayers()
        self.populatePopupTreeWidget()
        self.populateIdFields()
        self.populateVizTreeWidget(True)
        
    def changedIdComboBox(self):
        """Synd the Id Field with the model"""
        self.model.idField = str(self.dlg.idComboBox.currentText())
        self.validateId()
        
    def changedFormatComboBox(self):
        """Sync the selected output format with the model"""    
        self.model.selectedFormat  = self.dlg.formatComboBox.itemData(self.dlg.formatComboBox.currentIndex())
        
        t = topoJson()
        enabled = (self.dlg.formatComboBox.currentText() == t.name)
        self.dlg.simplificationSlider.setEnabled(enabled)
        if enabled == True:
            self.dlg.chosenSimpLabel.setVisible(True)
        else:
            self.dlg.chosenSimpLabel.setVisible(False)                                               
        
        
    def changedProjectionComboBox(self):
        """Sync the selected projection with the model"""
        self.model.selectedProjection = self.dlg.projectionComboBox.itemData(self.dlg.projectionComboBox.currentIndex())
        
        #update preview
        self.dlg.projPreview.setPixmap( QPixmap(os.path.join(os.path.dirname(__file__), "img", self.model.selectedProjection.preview)).scaledToWidth(112) )

        
    def changedSimplificationSlider(self):
        """Sync the simplification level with the model"""
        val = self.dlg.simplificationSlider.value()
        txt = self.model.steradians[val]
        
        self.model.simplification = txt
        self.dlg.chosenSimpLabel.setText(txt)

    def populateMainLayers(self):
        """Populate the layer combobox with the vector layer list"""
        self.dlg.mainLayerComboBox.clear()
        topMost = ""
        
        for v in self.model.vectors:
            if v.isVisible == True and topMost == "":
                topMost = v.name
            
            self.dlg.mainLayerComboBox.addItem(v.name, v.name) 
            
        if topMost != "":
            self.dlg.mainLayerComboBox.setCurrentIndex(self.dlg.mainLayerComboBox.findText(topMost))
            
    def populateFormats(self):
        """Populate the output format list"""            
        for f in self.model.formats:
            self.dlg.formatComboBox.addItem(f.name, f)
                
        if self.process.hasTopoJson() == False:
            t = topoJson()
            self.dlg.formatComboBox.removeItem(self.dlg.formatComboBox.findText(t.name))
          
    def populateIdFields(self):
        """Populate the field list from the main layer"""
        self.dlg.idComboBox.clear()   
        
        main = self.model.getMainLayer()
        if main is not None:
            for f in main.fields:
                self.dlg.idComboBox.addItem(f, f)
            
            if len(main.defaultId) > 0:
                self.dlg.idComboBox.setCurrentIndex(self.dlg.idComboBox.findData(main.defaultId))
             
    def populateExtraLayers(self):
        """Populate the extra layer treeview with the vector layer list"""
        self.dlg.extraVectorTreeWidget.clear()
            
        # create a root node in the tree
        rootNode = QTreeWidgetItem()
        rootNode.setText(0, "Vector Layers")
        
        # add all vector layers to it
        for v in self.model.vectors:
            if v.main == False:
                item = vectorItem(v)
                rootNode.addChild(item)
        
        # add all items to the tree
        self.dlg.extraVectorTreeWidget.addTopLevelItem(rootNode)
        
        # sort out visibility
        self.dlg.extraVectorTreeWidget.expandAll()
        self.dlg.extraVectorTreeWidget.resizeColumnToContents(0)
        self.dlg.extraVectorTreeWidget.resizeColumnToContents(1)
        
    def changedPopupCheckBox(self):    
        """Sync with the model and dis/en able the options for popup information"""
        self.model.popup = self.dlg.incPopupCheckBox.isChecked()
        self.dlg.popupTreeWidget.setEnabled(self.dlg.incPopupCheckBox.isChecked())
        self.dlg.popupPositionComboBox.setEnabled(self.dlg.incPopupCheckBox.isChecked())  
        self.dlg.popupPreviewEdit.setEnabled(self.dlg.incPopupCheckBox.isChecked())     
        
    def populatePopupTreeWidget(self):
        """Populate the list of attributes from the main vector layer"""
        
        self.dlg.popupTreeWidget.clear()
        self.dlg.popupPreviewEdit.clear()
        
        # create a root node in the tree
        rootNode = QTreeWidgetItem()
        rootNode.setText(0, "Popup fields")
        
        # add all attributes to it
        main = self.model.getMainLayer()
        if main is not None:
            for f in main.fields:
                item = fieldItem(f)
                rootNode.addChild(item)
        
        # add all items to the tree
        self.dlg.popupTreeWidget.addTopLevelItem(rootNode)
        
        # sort out visibility
        self.dlg.popupTreeWidget.expandAll()
        self.dlg.popupTreeWidget.resizeColumnToContents(0)
        self.dlg.popupTreeWidget.resizeColumnToContents(1)
        
    def populateVizTreeWidget(self, wipe):
        """Populate the list of attributes available for visualizations of the main layer
        
        :param wipe: Clear down the UI and model when redrawing the main layer
        :type wipe: Boolean
        
        """
        
        self.dlg.vizTreeWidget.clear()
        self.dlg.vizPreviewEdit.clear()
        
        if wipe == True:
            self.model.resetRanges()
            self.model.resetSelectedVizFields()
            self.dlg.vizLabelsLineEdit.clear()
               
        # create a root node in the tree
        rootNode = QTreeWidgetItem()
        rootNode.setText(0, "Data fields")
        
        # add all attributes to it
        main = self.model.getMainLayer()
        if main is not None:
            for f in main.vizFields:
                item = fieldItem(f)
                rootNode.addChild(item)
        
        # add all items to the tree
        self.dlg.vizTreeWidget.addTopLevelItem(rootNode)
        
        # sort out visibility
        self.dlg.vizTreeWidget.expandAll()
        self.dlg.vizTreeWidget.resizeColumnToContents(0)
        self.dlg.vizTreeWidget.resizeColumnToContents(1)
        

    def populateProjections(self):
        """Populate the projections combo box with supported d3 projections"""
        self.dlg.projectionComboBox.clear()

        for p in self.model.projections:
            self.dlg.projectionComboBox.addItem(p.name, p)    
    
    def populateFromSettings(self):
        """Restore dialog settings from those stored in the project settings"""
        
        self.webServerUrl = self.settings.webServerUrl()
        self.dlg.outputEdit.setText(self.settings.outputPath())
        
        # Last used format
        lastFormat = self.settings.outputFormat()
        if lastFormat != "":
            t = topoJson()
            if lastFormat != t.name or self.process.hasTopoJson() == True:
                formatIndex = self.dlg.formatComboBox.findText(lastFormat)
                if formatIndex > -1:
                    self.dlg.formatComboBox.setCurrentIndex(formatIndex)
        
        # Last used projection
        lastProj = self.settings.projection()
        if lastProj != "":
            projIndex = self.dlg.projectionComboBox.findText(lastProj)
            if projIndex > -1:
                self.dlg.projectionComboBox.setCurrentIndex(projIndex)
        
        
    def populateVizChartTypes(self):
        """Add all the supported chart types"""
        self.dlg.vizTypeComboBox.clear()

        for c in self.model.charts:
            self.dlg.vizTypeComboBox.addItem(c.name, c)
        
    def populateLegendPoition(self):
        """Populate the legend position combo box with supported positions"""
        self.dlg.legendPositionComboBox.clear()

        # retrieve positions from model
        for l in self.model.legendPositions:
            self.dlg.legendPositionComboBox.addItem(l, l)
    

    def populatePopupPoition(self):
        """Populate the popup position combo box with supported positions"""
        self.dlg.popupPositionComboBox.clear()

        # retrieve positions from model
        for p in self.model.popupPositions:
            self.dlg.popupPositionComboBox.addItem(p, p)
            
    def checkModified(self):
        """Check if the layers have been modified"""
        modified = self.process.areLayersModified()
        
        if modified == True:
            response = QMessageBox.warning(self.iface.mainWindow(), 
                            "Unsaved layer edits",
                            "Editing session in progress, continue the export anyway?", 
                            buttons=QMessageBox.Ok,
                            defaultButton=QMessageBox.Cancel)
            # return true if the user selects to cancel
            modified = (response == QMessageBox.Cancel)
        
        return modified

    def validate(self):
        """Perform basic input validation"""
        first = self.validateOutput()
        second = self.validateTitle()
        third = self.validateId()
        
        return first and second and third
    
    def validateId(self):
        """Perform validation on the Id field - just ensure an Id has been selected"""
        result = True
        if str(self.dlg.idComboBox.currentText()) == "":
            result = False        
        
        self.validationResult(self.dlg.idComboBox, result)
        
        return result
    
    def validationResult(self, field, result):
        """Highlight a particular UI field if validation failed"""
        fieldType = type(field).__name__
        
        if result == False:
            field.setStyleSheet(fieldType + " { background-color: #f6989d }")
            field.setFocus()
        else:
            field.setStyleSheet(fieldType + " { background-color: #ffffff }")
        

    def validateOutput(self):
        """Perform validation on the output directory"""
        result  = True
        if len(self.dlg.outputEdit.text()) == 0:
            result = False
        else:
            if os.path.exists(self.dlg.outputEdit.text()) == False:
                result = False
                
        if self.process.isWindows() == True:
            # Restriction on windows command call to ASCII only characters
            # Prevent input of Unicode characters here
            try:
                self.dlg.outputEdit.text().decode('ascii')
            except UnicodeEncodeError: 
                result = False

        self.validationResult(self.dlg.outputEdit, result)
            
        return result
            
    def changedOutput(self):
        """Perform validation when the output directory is altered"""
        self.model.outputFolder = self.dlg.outputEdit.text()
        self.validateOutput()
            
    def validateTitle(self):
        """Perform validation on the title"""
        result = True
        if len(self.dlg.titleEdit.text()) == 0: 
            result = False
            
        self.validationResult(self.dlg.titleEdit, result)
        
        return result
            
    def changedTitle(self):
        """Perform validation when the title is altered"""
        self.model.title = self.dlg.titleEdit.text()
        self.validateTitle()
        
    def changedWidth(self):
        """Sync width setting with the model"""
        self.model.width = self.safeConvertToInt(self.dlg.widthEdit.text())
                                                 
    def safeConvertToInt(self, s):
        """Cautiously convert user input to an integer"""
        n = 0
        if len(s) > 0:
            n = int(s) 
        
        return n
        
    def changedHeight(self):
        """Sync height setting with the model"""
        self.model.height = self.safeConvertToInt(self.dlg.heightEdit.text())

    def changedLabels(self):
        """Sync the labels setting with the model"""
        self.model.showLabels = self.dlg.labelsCheckBox.isChecked()  
        
    def changedPanZoom(self):
        """Sync the pan and zoom setting with the model"""
        self.model.panZoom = self.dlg.panZoomCheckBox.isChecked()    
        
    def changedLegendPoitionComboBox(self):
        """Sync selection with the model"""
        self.model.selectedLegendPosition = self.dlg.legendPositionComboBox.currentIndex()
        
    def changedLegend(self):
        """Dis/en able the legend options based on whether the user chooses to include one or not"""
        self.model.legend = self.dlg.legendCheckBox.isChecked()
        self.dlg.legendPositionComboBox.setEnabled(self.dlg.legendCheckBox.isChecked())
    
    def changedHeaderCheckBox(self):
        """Dis/en able the options for adding a header"""
        self.model.showHeader = self.dlg.headerCheckBox.isChecked()
        
    def changedExtraVector(self):
        """Dis/en able the options for extra vector layers"""
        self.model.extraVectors = self.dlg.extraVectorCheckBox.isChecked()
        self.dlg.extraVectorTreeWidget.setEnabled(self.dlg.extraVectorCheckBox.isChecked())
        
    def changedExtraVectorItems(self, item, column):
        """Sync the selected vector layers with the model"""
        if item.checkState(column) == Qt.Checked:
            self.model.setSelectedLayer(item.text(column), True)
        else:
            self.model.setSelectedLayer(item.text(column), False)
        
    def changedPopupPositionComboBox(self):
        """Sync the selected popup position with the model"""
        self.model.selectedPopupPosition = self.dlg.popupPositionComboBox.currentIndex()  
    
    def changedPopupItems(self, item, column):
        """Sync the popup fields with the model"""
        if item.checkState(column) == Qt.Checked:
            self.model.setSelectedPopupField(item.text(column), True)
        else:
            self.model.setSelectedPopupField(item.text(column), False)
        
        # add the templated popup to the preview
        self.dlg.popupPreviewEdit.clear()
        self.dlg.popupPreviewEdit.appendPlainText(self.model.getPopupTemplate())
    
    def changedVizCheckBox(self):
        """Keep UI and model in sync with current state of viz requirement"""
        self.model.hasViz = self.dlg.incVizCheckBox.isChecked()
        self.dlg.vizTypeComboBox.setEnabled(self.dlg.incVizCheckBox.isChecked())
        self.dlg.vizWidthEdit.setEnabled(self.dlg.incVizCheckBox.isChecked())
        self.dlg.vizHeightEdit.setEnabled(self.dlg.incVizCheckBox.isChecked())
        self.dlg.vizTreeWidget.setEnabled(self.dlg.incVizCheckBox.isChecked())
        self.dlg.vizPreviewEdit.setEnabled(self.dlg.incVizCheckBox.isChecked())
        self.dlg.vizLabelsLineEdit.setEnabled(self.dlg.incVizCheckBox.isChecked())
        self.dlg.addRangeButton.setEnabled(self.dlg.incVizCheckBox.isChecked())
        self.dlg.delRangeButton.setEnabled(self.dlg.incVizCheckBox.isChecked())

    def changedVizItems(self, item, column):
        """Sync the viz fields with the model"""
        if item.checkState(column) == Qt.Checked:
            self.model.setSelectedVizField(item.text(column), True)
        else:
            self.model.setSelectedVizField(item.text(column), False)
            
        
    def addVizRange(self):
        """Add the selected fields as a data range"""
        items = self.model.getCurrentRangeLength()
        if items > 0:
            if items < self.model.selectedVizChart.getMinFields() or items > self.model.selectedVizChart.getMaxFields():
                # Failed the input validation
                QMessageBox.warning(self.dlg, 
                            "Wrong number of fields",
                            "A {0} {1}".format(self.model.selectedVizChart.name, self.model.selectedVizChart.getFieldErrMessage()), 
                            buttons=QMessageBox.Ok,
                            defaultButton=QMessageBox.NoButton)
                                    
            else:
                text, ok = QInputDialog.getText(self.dlg,
                                                "Data range name",
                                                "Provide a name for the data range",
                                                QLineEdit.Normal, 
                                                "DataRange" + str(self.model.getRangeCount() + 1))
                if ok == True:
                    self.model.addCurrentRange(text) 
                    self.alterVizPreview()  
                
                
                
                
    def alterVizPreview(self):
        """UI changes required for viz updates"""
        # Redraw the tree
        self.populateVizTreeWidget(False)
        # Redraw the preview
        self.dlg.vizPreviewEdit.appendPlainText(self.model.getDataRangePreview())
        # Redraw the inputMask     
        self.dlg.vizLabelsLineEdit.setInputMask(self.model.getVizLabelMask())       
        
    def removeVizRange(self):
        """Pop a data range off the list"""
        self.model.deleteLastRange()
        self.alterVizPreview()   
        
    def changedVizLabels(self):
        """Convert the user input to a list of values"""
        out = []
        s = self.dlg.vizLabelsLineEdit.text()
        labels = s.split(",")
        for label in labels:
            out.append(label.strip())
            
        self.model.vizLabels = out
      
    def changedVizTypeComboBox(self):
        """Keep the selected chart in sync with the model"""
        self.model.selectedVizChart = self.dlg.vizTypeComboBox.itemData(self.dlg.vizTypeComboBox.currentIndex())
        # Wipe any existing data ranges
        self.populateVizTreeWidget(True)
        
    def changedVizWidth(self):
        """Store the chart width in the model"""
        self.model.vizWidth = self.safeConvertToInt(self.dlg.vizWidthEdit.text())
        
    def changedVizHeight(self):
        """Store the chart height in the model"""
        self.model.vizHeight = self.safeConvertToInt(self.dlg.vizHeightEdit.text())
            
    def doShowFolderDialog(self):
        """Display a folder dialog for the output directory"""
        folder = self.dlg.outputEdit.text()
        if len(folder) == 0:
            # no previous folder specified, set to the os temporary directory
            folder =  tempfile.gettempdir()   
        folder = QFileDialog.getExistingDirectory(self.dlg, "Select Output Directory", folder, QFileDialog.ShowDirsOnly)
        if len(folder) > 0:
            self.dlg.outputEdit.setText(folder)
    
    def closeDialog(self):
        """Cancel clicked closed the dialog"""
        self.disposeUI()
        self.dlg.close()
    
    def performOutput(self):
        """Ok clicked run the process and close dialog"""       
        if self.validate() == True and self.checkModified() == False:
            progressMessageBar = self.iface.messageBar().createMessage("Exporting...")
            progress = QProgressBar()   
            progress.setMaximum(self.process.getProgressTicks())
            progress.setAlignment(Qt.AlignLeft|Qt.AlignVCenter)
            progressMessageBar.layout().addWidget(progress)
            self.iface.messageBar().pushWidget(progressMessageBar, self.iface.messageBar().INFO)

            try:
                # Save settings
                self.settings.setOutputFormat(self.model.selectedFormat.name)
                self.settings.setProjection(self.model.selectedProjection.name)
                self.settings.setProjOutputPath(self.model.outputFolder)
                
                self.dlg.hide()
                self.process.export(progress, self.settings.webServerUrl())   
               
            except Exception as e:
                # What? log and then re-throw
                self.__logger.error("Exception\r\n" + traceback.format_exc(None))
                raise e
            finally:
                self.iface.messageBar().clearWidgets()
                self.closeDialog()
        else:
            self.dlg.tab.setCurrentIndex(0)

    def run(self):
        """Run method that performs all the real work"""
        # initialize the model
        if self.model is not None:
            # Remove running instance. Only one at a time please
            self.closeDialog()
            
        self.model = model( self.iface.mapCanvas().canvasColor().name() )
        self.settings = globalSettings()
        self.setupUI()        
          
            
    def setupUI(self):
        """Show the dialog and bind events"""
        self.dlg.show()
        
        # Create the process object to handle output and OS checks
        self.process = process(self.model)
        
        # Set UI state to match the model
        self.resetFields()
        
        #log some details
        self.__logger.info(QGis.QGIS_VERSION)
        self.__logger.info(sys.version)
        
        # build the model
        """Get the vector layers from QGIS and perform other startup actions"""
        # Reset
        del self.model.vectors[:]
        
        layers = self.iface.legendInterface().layers()
        found = False
        i = 0
        for layer in layers:
            if layer.type() == QgsMapLayer.VectorLayer and layer.rendererV2() is not None:
                found = True
                self.model.vectors.append(vector(self.iface, layer))
                
        # At __init__ the first in the list will be the main vector layer
        if found == True:
            self.model.vectors[0].setMain(True)
        
        # attach events
        # logic tab
        self.dlg.titleEdit.textChanged.connect(self.changedTitle)
        self.dlg.headerCheckBox.stateChanged.connect(self.changedHeaderCheckBox)
        self.dlg.widthEdit.textChanged.connect(self.changedWidth)
        self.dlg.heightEdit.textChanged.connect(self.changedHeight)
        self.dlg.mainLayerComboBox.currentIndexChanged.connect(self.changedMainLayerComboBox)
        self.dlg.idComboBox.currentIndexChanged.connect(self.changedIdComboBox)
        self.dlg.projectionComboBox.currentIndexChanged.connect(self.changedProjectionComboBox)
        self.dlg.formatComboBox.currentIndexChanged.connect(self.changedFormatComboBox)
        self.dlg.simplificationSlider.valueChanged.connect(self.changedSimplificationSlider)
        self.dlg.outputEdit.textChanged.connect(self.changedOutput)
        self.dlg.outputButton.clicked.connect(self.doShowFolderDialog)
        # extras tab        
        self.dlg.extraVectorCheckBox.stateChanged.connect(self.changedExtraVector)
        self.dlg.extraVectorTreeWidget.itemClicked.connect(self.changedExtraVectorItems)
        self.dlg.labelsCheckBox.stateChanged.connect(self.changedLabels)
        self.dlg.panZoomCheckBox.stateChanged.connect(self.changedPanZoom)
        self.dlg.legendCheckBox.stateChanged.connect(self.changedLegend)
        self.dlg.legendPositionComboBox.currentIndexChanged.connect(self.changedLegendPoitionComboBox)
        # popup tab
        self.dlg.incPopupCheckBox.stateChanged.connect(self.changedPopupCheckBox)
        self.dlg.popupPositionComboBox.currentIndexChanged.connect(self.changedPopupPositionComboBox)
        self.dlg.popupTreeWidget.itemClicked.connect(self.changedPopupItems)
        # viz tab
        self.dlg.incVizCheckBox.stateChanged.connect(self.changedVizCheckBox)
        self.dlg.vizTypeComboBox.currentIndexChanged.connect(self.changedVizTypeComboBox)
        self.dlg.vizWidthEdit.textChanged.connect(self.changedVizWidth)
        self.dlg.vizHeightEdit.textChanged.connect(self.changedVizHeight)
        self.dlg.vizTreeWidget.itemClicked.connect(self.changedVizItems)
        self.dlg.addRangeButton.clicked.connect(self.addVizRange)
        self.dlg.delRangeButton.clicked.connect(self.removeVizRange)
        self.dlg.vizLabelsLineEdit.textChanged.connect(self.changedVizLabels)
        # buttons
        self.dlg.buttonBox.button(QDialogButtonBox.Cancel).clicked.connect(self.closeDialog)
        self.dlg.buttonBox.button(QDialogButtonBox.Ok).clicked.connect(self.performOutput)
        
        # populate the controls 
        self.populateMainLayers()
        self.populateProjections()
        self.populateFormats()
        self.populateLegendPoition()
        self.populatePopupPoition()
        self.populateExtraLayers()
        self.populatePopupTreeWidget()
        self.populateVizChartTypes() 
        
        # read the stored settings
        self.populateFromSettings()
        
        # Run the dialog event loop
        self.dlg.exec_()
        
    def resetFields(self):
        """Reset the fields to the starting state"""
        self.dlg.tab.setCurrentIndex(0)
        self.dlg.titleEdit.setText("")
        self.dlg.headerCheckBox.setChecked(False)
        self.dlg.widthEdit.setText("800")
        self.dlg.heightEdit.setText("600")
        self.dlg.mainLayerComboBox.clear()
        self.dlg.idComboBox.clear()
        self.dlg.projectionComboBox.clear()
        self.dlg.formatComboBox.clear()
        self.dlg.simplificationSlider.setValue(0)
        self.dlg.chosenSimpLabel.setText("")
        self.dlg.outputEdit.setText("")
        # extras tab        
        self.dlg.extraVectorCheckBox.setChecked(False)
        self.dlg.extraVectorTreeWidget.clear()
        self.dlg.extraVectorTreeWidget.setEnabled(False)
        self.dlg.labelsCheckBox.setChecked(False)
        self.dlg.panZoomCheckBox.setChecked(False)
        self.dlg.legendCheckBox.setChecked(False)
        self.dlg.legendPositionComboBox.clear()
        # popup tab
        self.dlg.incPopupCheckBox.setChecked(False)
        self.dlg.popupPositionComboBox.clear()
        self.dlg.popupTreeWidget.clear()
        self.dlg.popupTreeWidget.setEnabled(False)
        self.dlg.popupPreviewEdit.setEnabled(False)
        # viz tab
        self.dlg.incVizCheckBox.setChecked(False)
        self.dlg.vizTypeComboBox.clear()
        self.dlg.vizWidthEdit.setText("240")
        self.dlg.vizHeightEdit.setText("240")
        self.dlg.vizTreeWidget.clear()
        self.dlg.vizTreeWidget.setEnabled(False)
        self.dlg.vizPreviewEdit.clear()
        self.dlg.vizPreviewEdit.setEnabled(False)
        self.dlg.vizLabelsLineEdit.clear()
        self.dlg.vizLabelsLineEdit.setEnabled(False)
        self.dlg.addRangeButton.setEnabled(False)
        self.dlg.delRangeButton.setEnabled(False)

    def disposeUI(self):
        """Unbind events and reset state to match the model"""
        try:
            self.dlg.titleEdit.textChanged.disconnect()
            self.dlg.headerCheckBox.stateChanged.disconnect()
            self.dlg.widthEdit.textChanged.disconnect()
            self.dlg.heightEdit.textChanged.disconnect()
            self.dlg.mainLayerComboBox.currentIndexChanged.disconnect()
            self.dlg.idComboBox.currentIndexChanged.disconnect()
            self.dlg.projectionComboBox.currentIndexChanged.disconnect()
            self.dlg.formatComboBox.currentIndexChanged.disconnect()
            self.dlg.simplificationSlider.valueChanged.disconnect()
            self.dlg.outputEdit.textChanged.disconnect()
            self.dlg.outputButton.clicked.disconnect()
            # extras tab        
            self.dlg.extraVectorCheckBox.stateChanged.disconnect()
            self.dlg.extraVectorTreeWidget.itemClicked.disconnect()
            self.dlg.labelsCheckBox.stateChanged.disconnect()
            self.dlg.panZoomCheckBox.stateChanged.disconnect()
            self.dlg.legendCheckBox.stateChanged.disconnect()
            self.dlg.legendPositionComboBox.currentIndexChanged.disconnect()
            # popup tab
            self.dlg.incPopupCheckBox.stateChanged.disconnect()
            self.dlg.popupPositionComboBox.currentIndexChanged.disconnect()
            self.dlg.popupTreeWidget.itemClicked.disconnect()
            # viz tab
            self.dlg.incVizCheckBox.stateChanged.disconnect()
            self.dlg.vizTypeComboBox.currentIndexChanged.disconnect()
            self.dlg.vizWidthEdit.textChanged.disconnect()
            self.dlg.vizHeightEdit.textChanged.disconnect()
            self.dlg.vizTreeWidget.itemClicked.disconnect()
            self.dlg.addRangeButton.clicked.disconnect()
            self.dlg.delRangeButton.clicked.disconnect()
            self.dlg.vizLabelsLineEdit.textChanged.disconnect()
            # buttons
            self.dlg.buttonBox.button(QDialogButtonBox.Cancel).clicked.disconnect()
            self.dlg.buttonBox.button(QDialogButtonBox.Ok).clicked.disconnect()
        except TypeError:
            pass
        
        self.resetFields()
        
        self.model = None
        
