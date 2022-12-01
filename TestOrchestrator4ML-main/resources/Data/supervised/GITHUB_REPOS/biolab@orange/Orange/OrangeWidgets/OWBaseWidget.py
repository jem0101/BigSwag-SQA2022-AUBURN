#
# OWWidget.py
# Orange Widget
# A General Orange Widget, from which all the Orange Widgets are derived
#
import os
import sys
import cPickle
import logging
import warnings
import shutil
import time

import random
import user

from Orange.utils import environ
from Orange.orng.orngEnviron import directoryNames as old_directory_names
from PyQt4.QtGui import *
from PyQt4.QtCore import *

# Define  pyqtConfigure not available in PyQt4 versions prior to 4.6
if not hasattr(QObject, "pyqtConfigure"):
    def pyqtConfigure(obj, **kwargs):
        meta = obj.metaObject()
        for name, val in kwargs.items():
            if meta.indexOfProperty(name) >= 0:
                obj.setProperty(name, QVariant(val))
            elif meta.indexOfSignal(meta.normalizedSignature(name)) >= 0:
                obj.connect(obj, SIGNAL(name), val)
    QObject.pyqtConfigure = pyqtConfigure

from OWContexts import *

import orange
from orange import ExampleTable

import Orange.utils
from Orange.utils import debugging as orngDebugging
from string import *

from Orange.OrangeCanvas.registry.description import (
    Default, NonDefault, Single, Multiple, Explicit, Dynamic,
    InputSignal, OutputSignal
)

from Orange.OrangeCanvas.scheme.widgetsscheme import (
    SignalLink, WidgetsSignalManager, SignalWrapper
)

import OWGUI


_log = logging.getLogger(__name__)

ERROR = 0
WARNING = 1

TRUE = 1
FALSE = 0


def _deprecation_warning(name):
    warnings.warn(
        "{0!r} is deprecated. It will be removed in Orange 2.8".format(name),
        DeprecationWarning,
        stacklevel=2
    )


def unisetattr(self, name, value, grandparent):
    if "." in name:
        names = name.split(".")
        lastname = names.pop()
        obj = reduce(lambda o, n: getattr(o, n, None),  names, self)
    else:
        lastname, obj = name, self

    if not obj:
        print "unable to set setting ", name, " to value ", value
    else:
        if hasattr(grandparent, "__setattr__") and isinstance(obj, grandparent):
            grandparent.__setattr__(obj, lastname,  value)
        else:
            setattr(obj, lastname, value)
#            obj.__dict__[lastname] = value

    controlledAttributes = hasattr(self, "controlledAttributes") and getattr(self, "controlledAttributes", None)
    controlCallback = controlledAttributes and controlledAttributes.get(name, None)
    if controlCallback:
        for callback in controlCallback:
            callback(value)
#        controlCallback(value)

    # controlled things (checkboxes...) never have __attributeControllers
    else:
        if hasattr(self, "__attributeControllers"):
            for controller, myself in self.__attributeControllers.keys():
                if getattr(controller, myself, None) != self:
                    del self.__attributeControllers[(controller, myself)]
                    continue

                controlledAttributes = hasattr(controller, "controlledAttributes") and getattr(controller, "controlledAttributes", None)
                if controlledAttributes:
                    fullName = myself + "." + name

                    controlCallback = controlledAttributes.get(fullName, None)
                    if controlCallback:
                        for callback in controlCallback:
                            callback(value)

                    else:
                        lname = fullName + "."
                        dlen = len(lname)
                        for controlled in controlledAttributes.keys():
                            if controlled[:dlen] == lname:
                                self.setControllers(value, controlled[dlen:], controller, fullName)
                                # no break -- can have a.b.c.d and a.e.f.g; needs to set controller for all!


    # if there are any context handlers, call the fastsave to write the value into the context
    if hasattr(self, "contextHandlers") and hasattr(self, "currentContexts"):
        for contextName, contextHandler in self.contextHandlers.items():
            contextHandler.fastSave(self.currentContexts.get(contextName), self, name, value)


class ControlledAttributesDict(dict):
    def __init__(self, master):
        self.master = master

    def __setitem__(self, key, value):
        if not self.has_key(key):
            dict.__setitem__(self, key, [value])
        else:
            dict.__getitem__(self, key).append(value)
        self.master.setControllers(self.master, key, self.master, "")


class AttributeList(list):
    pass


class ExampleList(list):
    pass

widgetId = 0


_SETTINGS_VERSION_KEY = "__settingsDataVersion"


class OWBaseWidget(QDialog):
    def __new__(cls, *arg, **args):
        self = QDialog.__new__(cls)

        self.currentContexts = {}   # the "currentContexts" MUST be the first thing assigned to a widget
        self._useContexts = 1       # do you want to use contexts
        self._owInfo = 1            # currently disabled !!!
        self._owWarning = 1         # do we want to see warnings
        self._owError = 1           # do we want to see errors
        self._owShowStatus = 0      # do we want to see warnings and errors in status bar area of the widget
        self._guiElements = []      # used for automatic widget debugging
        for key in args:
            if key in ["_owInfo", "_owWarning", "_owError", "_owShowStatus", "_useContexts", "_category", "_settingsFromSchema"]:
                self.__dict__[key] = args[key]        # we cannot use __dict__.update(args) since we can have many other

        return self

    def __init__(self, parent=None, signalManager=None, title="Orange BaseWidget", modal=FALSE, savePosition=False, resizingEnabled=1, **args):
        if resizingEnabled:
            QDialog.__init__(self, parent, Qt.Dialog)
        else:
            QDialog.__init__(self, parent, Qt.Dialog |
                             Qt.MSWindowsFixedSizeDialogHint)

        # do we want to save widget position and restore it on next load
        self.savePosition = savePosition
        if savePosition:
            self.settingsList = getattr(self, "settingsList", []) + ["widgetShown", "savedWidgetGeometry"]

        self.setCaption(title)
        self.setFocusPolicy(Qt.StrongFocus)

        # XXX: Shadows a base class method. Find all uses where 'parent' is
        # being accessed with an instance member lookup and fix them.
        self.parent = parent

        self.needProcessing = 0  # used by the old (pre v2.7) signalManager

        self.signalManager = signalManager

        self.inputs = []     # signalName:(dataType, handler, onlySingleConnection)
        self.outputs = []    # signalName: dataType
        self.wrappers = []    # stored wrappers for widget events
        self.linksIn = {}      # signalName : (dirty, widgetFrom, handler, signalData)
        self.linksOut = {}       # signalName: (signalData, id)
        self.connections = {}   # dictionary where keys are (control, signal) and values are wrapper instances. Used in connect/disconnect
        self.controlledAttributes = ControlledAttributesDict(self)
        self.progressBarHandler = None  # handler for progress bar events
        self.processingHandler = None   # handler for processing events
        self.eventHandler = None
        self.callbackDeposit = []
        self.startTime = time.time()    # used in progressbar

        self.widgetStateHandler = None
        self.widgetState = {"Info":{}, "Warning":{}, "Error":{}}

        if hasattr(self, "contextHandlers"):
            for contextHandler in self.contextHandlers.values():
                contextHandler.initLocalContext(self)
                
        global widgetId
        widgetId += 1
        self.widgetId = widgetId

        self.asyncCalls = []
        self.asyncBlock = False
        self.__wasShown = False
        self.__progressBarValue = -1
        self.__progressState = 0
        self.__statusMessage = ""

    @property
    def widgetDir(self):
        # This seems to be the only use of the orngEnviron.directoryNames
        # usage (used in various ploting widget to access icons/Dlg_* png)
        warnings.warn(
            "widgetDir is deprecated. " +
            "Use Orange.utils.environ.widget_install_dir",
            DeprecationWarning)
        return environ.widget_install_dir

    def setWidgetIcon(self, iconName):
        warnings.warn(
            "setWidgetIcon is deprecated and will be removed in the future. "
            "Use setWindowIcon instead.",
            DeprecationWarning
        )

        def getIconNames(iconName):
            names = []
            name, ext = os.path.splitext(iconName)
            for num in [16, 32, 42, 60]:
                names.append("%s_%d%s" % (name, num, ext))
            fullPaths = []
            for paths in [(self.widgetDir, name), (self.widgetDir, "icons", name), (os.path.dirname(sys.modules[self.__module__].__file__), "icons", name)]:
                for name in names + [iconName]:
                    fname = os.path.join(*paths)
                    if os.path.exists(fname):
                        fullPaths.append(fname)
                if fullPaths != []:
                    break

            if len(fullPaths) > 1 and fullPaths[-1].endswith(iconName):
                fullPaths.pop()     # if we have the new icons we can remove the default icon
            return fullPaths

        if isinstance(iconName, list):
            iconNames = iconName
        else:
            iconNames = getIconNames(iconName)

        icon = QIcon()
        for name in iconNames:
            pix = QPixmap(name)
            icon.addPixmap(pix)
        self.setWindowIcon(icon)

    def createAttributeIconDict(self):
        return OWGUI.getAttributeIcons()

    def isDataWithClass(self, data, wantedVarType = None, checkMissing=False):
        self.error([1234, 1235, 1236])
        if not data:
            return 0
        if not data.domain.classVar:
            self.error(1234, "A data set with a class attribute is required.")
            return 0
        if wantedVarType and data.domain.classVar.varType != wantedVarType:
            self.error(1235, "Unable to handle %s class." % (data.domain.classVar.varType == orange.VarTypes.Discrete and "discrete" or "continuous"))
            return 0
        if checkMissing and not orange.Preprocessor_dropMissingClasses(data):
            self.error(1236, "Unable to handle data set with no known classes")
            return 0
        return 1

    def restoreWidgetPosition(self):
        """
        Restore the widget's position from the saved settings.

        This is called from the widget's :func:`showEvent`
        """
        if self.savePosition:
            geometry = getattr(self, "savedWidgetGeometry", None)
            restored = False
            if geometry is not None:
                restored = self.restoreGeometry(QByteArray(geometry))

            if restored:
                space = qApp.desktop().availableGeometry(self)
                frame, geometry = self.frameGeometry(), self.geometry()

                # Fix the widget size to fit inside the available space
                width = min(space.width() - (frame.width() - geometry.width()), geometry.width())
                height = min(space.height() - (frame.height() - geometry.height()), geometry.height())
                self.resize(width, height)

                # Move the widget to the center of available space if it
                # is currently outside it
                if not space.contains(self.frameGeometry()):
                    x = max(0, space.width() / 2 - width / 2)
                    y = max(0, space.height() / 2 - height / 2)

                    self.move(x, y)

    def restoreWidgetStatus(self):
        _deprecation_warning("restoreWidgetStatus")
        if self.savePosition and getattr(self, "widgetShown", None):
            self.show()

    def resizeEvent(self, ev):
        QDialog.resizeEvent(self, ev)
        # Don't store geometry if the widget is not visible
        # (the widget receives the resizeEvent before showEvent and we must not
        # overwrite the the savedGeometry before then)
        if self.savePosition and self.isVisible():
            self.savedWidgetGeometry = str(self.saveGeometry())

    def showEvent(self, ev):
        QDialog.showEvent(self, ev)
        if self.savePosition:
            self.widgetShown = 1

            if not self.__wasShown:
                self.__wasShown = True
                self.restoreWidgetPosition()

    def hideEvent(self, ev):
        if self.savePosition:
            self.widgetShown = 0
            self.savedWidgetGeometry = str(self.saveGeometry())
        QDialog.hideEvent(self, ev)

    def closeEvent(self, ev):
        if self.savePosition and self.isVisible() and self.__wasShown:
            # self.geometry() is 'invalid' (not yet resized/layout) until the
            # widget is made explicitly visible or it might be invalid if
            # widget was hidden (in this case hideEvent already saved a valid
            # geometry).
            self.savedWidgetGeometry = str(self.saveGeometry())
        QDialog.closeEvent(self, ev)

    def wheelEvent(self, event):
        """ Silently accept the wheel event. This is to ensure combo boxes
        and other controls that have focus don't receive this event unless
        the cursor is over them.

        """
        event.accept()

    def setCaption(self, caption):
        if self.parent != None and isinstance(self.parent, QTabWidget):
            self.parent.setTabText(self.parent.indexOf(self), caption)
        else:
            # we have to save caption title in case progressbar will change it
            self.captionTitle = unicode(caption)
            self.setWindowTitle(caption)

    # put this widget on top of all windows
    def reshow(self):
        self.show()
        self.raise_()
        self.activateWindow()

    def send(self, signalName, value, id = None):
        if self.linksOut.has_key(signalName):
            self.linksOut[signalName][id] = value
        else:
            self.linksOut[signalName] = {id:value}

        if self.signalManager is not None:
            self.signalManager.send(self, signalName, value, id)

    def getdeepattr(self, attr, **argkw):
        try:
            return reduce(lambda o, n: getattr(o, n, None),  attr.split("."), self)
        except:
            if argkw.has_key("default"):
                return argkw[default]
            else:
                raise AttributeError, "'%s' has no attribute '%s'" % (self, attr)

    def setSettings(self, settings):
        """
        Set/restore the widget settings.

        :param dict settings: A settings dictionary.

        """

        if settings.get(_SETTINGS_VERSION_KEY, None) == \
                getattr(self, "settingsDataVersion", None):
            if _SETTINGS_VERSION_KEY in settings:
                settings = settings.copy()
                del settings[_SETTINGS_VERSION_KEY]

            for key in settings:
                self.__setattr__(key, settings[key])

    def getSettings(self, alsoContexts=True, globalContexts=False):
        """
        Return a dictionary with all settings for serialization.
        """
        settings = {}
        if hasattr(self, "settingsList"):
            for name in self.settingsList:
                try:
                    settings[name] = self.getdeepattr(name)
                except Exception:
                    #print "Attribute %s not found in %s widget. Remove it from the settings list." % (name, self.captionTitle)
                    pass
            settings[_SETTINGS_VERSION_KEY] = getattr(self, "settingsDataVersion", None)

        if alsoContexts:
            self.synchronizeContexts()
            contextHandlers = getattr(self, "contextHandlers", {})
            for contextHandler in contextHandlers.values():
                contextHandler.mergeBack(self)
#                settings[contextHandler.localContextName] = contextHandler.globalContexts
# Instead of the above line, I found this; as far as I recall this was a fix
# for some bugs related to, say, Select Attributes not handling the context
# attributes properly, but I dare not add it without understanding what it does.
# Here it is, if these contexts give us any further trouble.
                if (contextHandler.syncWithGlobal and contextHandler.globalContexts is getattr(self, contextHandler.localContextName)) or globalContexts:
                    settings[contextHandler.localContextName] = contextHandler.globalContexts 
                else:
                    contexts = getattr(self, contextHandler.localContextName, None)
                    if contexts:
                        settings[contextHandler.localContextName] = contexts
###
                settings[contextHandler.localContextName+"Version"] = (contextStructureVersion, contextHandler.contextDataVersion)

        return settings

    def getDefaultSettingsFilename(self):
        """
        Return a default widget settings filename.
        """
        settings_dir = environ.widget_settings_dir

        class_ = type(self)
        version = getattr(class_, "settingsDataVersion", None)
        if version is not None:
            version = ".".join(str(subv) for subv in version)
            basename = "%s.%s.%s.pck" % (class_.__module__, class_.__name__,
                                         version)
        else:
            basename = "%s.%s.pck" % (class_.__module__, class_.__name__)
        filename = os.path.join(settings_dir, basename)

        if os.path.exists(filename):
            return filename

        # Try to find the old filename format ({caption}.ini) and
        # copy it to the new place
        fs_encoding = sys.getfilesystemencoding()
        basename = self.captionTitle + ".ini"
        legacy_filename = os.path.join(
            settings_dir,  # is assumed to be a str in FS encoding
            basename.encode(fs_encoding))

        if os.path.isfile(legacy_filename):
            # Copy the old settings file to the new place.
            shutil.copy(legacy_filename, filename)

        return filename

    def getSettingsFile(self, file):
        if file is None:
            file = self.getDefaultSettingsFilename()

            if not os.path.exists(file):
                try:
                    f = open(file, "wb")
                    cPickle.dump({}, f)
                    f.close()
                except IOError:
                    return

        if isinstance(file, basestring):
            if os.path.exists(file):
                return open(file, "r")
        else:
            return file

    # Loads settings from the widget's settings file.
    def loadSettings(self, file=None):
        file = self.getSettingsFile(file)
        if file:
            try:
                settings = cPickle.load(file)
            except Exception, ex:
                print >> sys.stderr, "Failed to load settings!", repr(ex)
                settings = {}

            if hasattr(self, "_settingsFromSchema"):
                settings.update(self._settingsFromSchema)

            # can't close everything into one big try-except since this would mask all errors in the below code
            if settings:
                if hasattr(self, "settingsList"):
                    self.setSettings(settings)

                contextHandlers = getattr(self, "contextHandlers", {})
                for contextHandler in contextHandlers.values():
                    localName = contextHandler.localContextName

                    structureVersion, dataVersion = settings.get(localName+"Version", (0, 0))
                    if (structureVersion < contextStructureVersion or dataVersion < contextHandler.contextDataVersion) \
                       and settings.has_key(localName):
                        del settings[localName]
                        delattr(self, localName)
                        contextHandler.initLocalContext(self)
                    if not hasattr(self, "_settingsFromSchema"): #When running stand alone widgets
                        if contextHandler.syncWithGlobal:
                            contexts = settings.get(localName, None)
                            if contexts is not None:
                                contextHandler.globalContexts = contexts
                        else:
                            setattr(self, localName, contextHandler.globalContexts)

    def saveSettings(self, file=None):
        settings = self.getSettings(globalContexts=True)
        if settings:
            if file is None:
                file = self.getDefaultSettingsFilename()

            if isinstance(file, basestring):
                file = open(file, "w")
            cPickle.dump(settings, file)

    # Loads settings from string str which is compatible with cPickle
    def loadSettingsStr(self, str):
        if str == None or str == "":
            return

        settings = cPickle.loads(str)
        self.setSettings(settings)

        contextHandlers = getattr(self, "contextHandlers", {})
        for contextHandler in contextHandlers.values():
            localName = contextHandler.localContextName
            if settings.has_key(localName):
                structureVersion, dataVersion = settings.get(localName+"Version", (0, 0))
                if structureVersion < contextStructureVersion or dataVersion < contextHandler.contextDataVersion:
                    del settings[localName]
                    delattr(self, localName)
                    contextHandler.initLocalContext(self)
                else:
                    setattr(self, localName, settings[localName])

    # return settings in string format compatible with cPickle
    def saveSettingsStr(self):
        settings = self.getSettings()
        return cPickle.dumps(settings)

    def onDeleteWidget(self):
        """
        Called when the widget is deleted from a scheme by the user.

        Subclasses can override this and cleanup any resources they have.
        """
        pass

    # this function is only intended for derived classes to send appropriate
    # signals when all settings are loaded.
    # NOTE: This is useless, this does not get called by the base widget at
    # any time. The subclasses are expected to call this themselves. It only
    # remains if one tries to call the base implementation.
    def activateLoadedSettings(self):
        pass

    def handleNewSignals(self):
        # this is called after all new signals have been handled
        # implement this in your widget if you want to process something only after you received multiple signals
        pass

    # ########################################################################
    def connect(self, control, signal, method, type=Qt.AutoConnection):
        wrapper = SignalWrapper(self, method)
        self.connections[(control, signal)] = wrapper   # save for possible disconnect
        self.wrappers.append(wrapper)
        QDialog.connect(control, signal, wrapper, type)
        #QWidget.connect(control, signal, method)        # ordinary connection useful for dialogs and windows that don't send signals to other widgets

    def disconnect(self, control, signal, method=None):
        wrapper = self.connections[(control, signal)]
        QDialog.disconnect(control, signal, wrapper)

    #===============================================================
    # The following methods are used only by the old signal manager
    # (orngSignalManager) and possibly the 'Saved Applications' from
    # the old Canvas.
    # ==============================================================

    # does widget have a signal with name in inputs
    def hasInputName(self, name):
        _deprecation_warning("hasInputName")
        for input in self.inputs:
            if name == input[0]: return 1
        return 0

    # does widget have a signal with name in outputs
    def hasOutputName(self, name):
        _deprecation_warning("hasOutputName")
        for output in self.outputs:
            if name == output[0]: return 1
        return 0

    def getInputType(self, signalName):
        _deprecation_warning("getInputType")
        for input in self.inputs:
            if input[0] == signalName: return input[1]
        return None

    def getOutputType(self, signalName):
        _deprecation_warning("getOutputType")
        for output in self.outputs:
            if output[0] == signalName: return output[1]
        return None

    def signalIsOnlySingleConnection(self, signalName):
        _deprecation_warning("signalIsOnlySingleConnection")
        for i in self.inputs:
            input = InputSignal(*i)
            if input.name == signalName: return input.single

    def addInputConnection(self, widgetFrom, signalName):
        _deprecation_warning("addInputConnection")
        for i in range(len(self.inputs)):
            if self.inputs[i][0] == signalName:
                handler = self.inputs[i][2]
                break

        existing = []
        if self.linksIn.has_key(signalName):
            existing = self.linksIn[signalName]
            for (dirty, widget, handler, data) in existing:
                if widget == widgetFrom: return             # no need to add new tuple, since one from the same widget already exists
        self.linksIn[signalName] = existing + [(0, widgetFrom, handler, [])]    # (dirty, handler, signalData)
        #if not self.linksIn.has_key(signalName): self.linksIn[signalName] = [(0, widgetFrom, handler, [])]    # (dirty, handler, signalData)

    # delete a link from widgetFrom and this widget with name signalName
    def removeInputConnection(self, widgetFrom, signalName):
        _deprecation_warning("removeInputConnection")
        if self.linksIn.has_key(signalName):
            links = self.linksIn[signalName]
            for i in range(len(self.linksIn[signalName])):
                if widgetFrom == self.linksIn[signalName][i][1]:
                    self.linksIn[signalName].remove(self.linksIn[signalName][i])
                    if self.linksIn[signalName] == []:  # if key is empty, delete key value
                        del self.linksIn[signalName]
                    return

    # return widget, that is already connected to this singlelink signal. If this widget exists, the connection will be deleted (since this is only single connection link)
    def removeExistingSingleLink(self, signal):
        _deprecation_warning("removeExistingSingleLink")
        for i in self.inputs:
            input = InputSignal(*i)
            if input.name == signal and not input.single: return None

        for signalName in self.linksIn.keys():
            if signalName == signal:
                widget = self.linksIn[signalName][0][1]
                del self.linksIn[signalName]
                return widget

        return None

    # signal manager calls this function when all input signals have updated the data
    def processSignals(self):
        _deprecation_warning("processSignals")
        if self.processingHandler:
            self.processingHandler(self, 1)    # focus on active widget
        newSignal = 0        # did we get any new signals

        # we define only a way to handle signals that have defined a handler function
        for signal in self.inputs:        # we go from the first to the last defined input
            key = signal[0]
            if self.linksIn.has_key(key):
                for i in range(len(self.linksIn[key])):
                    (dirty, widgetFrom, handler, signalData) = self.linksIn[key][i]
                    if not (handler and dirty): continue
                    newSignal = 1

                    qApp.setOverrideCursor(Qt.WaitCursor)
                    try:
                        for (value, id, nameFrom) in signalData:
                            if self.signalIsOnlySingleConnection(key):
                                self.printEvent("ProcessSignals: Calling %s with %s" % (handler, value), eventVerbosity = 2)
                                handler(value)
                            else:
                                self.printEvent("ProcessSignals: Calling %s with %s (%s, %s)" % (handler, value, nameFrom, id), eventVerbosity = 2)
                                handler(value, (widgetFrom, nameFrom, id))
                    except:
                        type, val, traceback = sys.exc_info()
                        sys.excepthook(type, val, traceback)  # we pretend that we handled the exception, so that we don't crash other widgets
                    qApp.restoreOverrideCursor()

                    self.linksIn[key][i] = (0, widgetFrom, handler, []) # clear the dirty flag

        if newSignal == 1:
            self.handleNewSignals()
        
        while self.isBlocking():
            self.thread().msleep(50)
            qApp.processEvents()

        if self.processingHandler:
            self.processingHandler(self, 0)    # remove focus from this widget
        self.needProcessing = 0

    # set new data from widget widgetFrom for a signal with name signalName
    def updateNewSignalData(self, widgetFrom, signalName, value, id, signalNameFrom):
        _deprecation_warning("updateNewSignalData")
        if not self.linksIn.has_key(signalName): return
        for i in range(len(self.linksIn[signalName])):
            (dirty, widget, handler, signalData) = self.linksIn[signalName][i]
            if widget == widgetFrom:
                if self.linksIn[signalName][i][3] == []:
                    self.linksIn[signalName][i] = (1, widget, handler, [(value, id, signalNameFrom)])
                else:
                    found = 0
                    for j in range(len(self.linksIn[signalName][i][3])):
                        (val, ID, nameFrom) = self.linksIn[signalName][i][3][j]
                        if ID == id and nameFrom == signalNameFrom:
                            self.linksIn[signalName][i][3][j] = (value, id, signalNameFrom)
                            found = 1
                    if not found:
                        self.linksIn[signalName][i] = (1, widget, handler, self.linksIn[signalName][i][3] + [(value, id, signalNameFrom)])
        self.needProcessing = 1

    # ############################################
    # PROGRESS BAR FUNCTIONS

    #: Progress bar value has changed
    progressBarValueChanged = pyqtSignal(float)

    #: Processing state has changed
    processingStateChanged = pyqtSignal(int)

    def progressBarInit(self):
        """
        Initialize the widget's progress bar (i.e show and set progress to 0%)
        """
        self.startTime = time.time()
        self.setWindowTitle(self.captionTitle + " (0% complete)")
        if self.progressBarHandler:
            self.progressBarHandler(self, 0)

        if self.__progressState != 1:
            self.__progressState = 1
            self.processingStateChanged.emit(1)

        self.progressBarValue = 0

    def progressBarSet(self, value, processEventsFlags=QEventLoop.AllEvents):
        """
        Set the current progress bar to `value`.

        .. note::
            This method will also call `qApp.processEvents` with the
            `processEventsFlags` unless the processEventsFlags equals
            ``None``.

        """
        old = self.__progressBarValue
        self.__progressBarValue = value

        if value > 0:
            if self.__progressState != 1:
                warnings.warn("progressBarSet() called without a "
                              "preceding progressBarInit()",
                              stacklevel=2)
                self.__progressState = 1
                self.processingStateChanged.emit(1)

            usedTime = max(1, time.time() - self.startTime)
            totalTime = (100.0 * usedTime) / float(value)
            remainingTime = max(0, totalTime - usedTime)
            h = int(remainingTime / 3600)
            min = int((remainingTime - h * 3600) / 60)
            sec = int(remainingTime - h * 3600 - min * 60)
            if h > 0:
                text = "%(h)d:%(min)02d:%(sec)02d" % vars()
            else:
                text = "%(min)d:%(sec)02d" % vars()
            self.setWindowTitle(self.captionTitle + " (%(value).2f%% complete, remaining time: %(text)s)" % vars())
        else:
            self.setWindowTitle(self.captionTitle + " (0% complete)")
        if self.progressBarHandler:
            self.progressBarHandler(self, value)

        if old != value:
            self.progressBarValueChanged.emit(value)

        if processEventsFlags is not None:
            qApp.processEvents(processEventsFlags)

    def progressBarValue(self):
        """
        Current progress bar value (-1 if the progress bar is not initialized).
        """
        return self.__progressBarValue if self.__progressState == 1 else -1.0

    progressBarValue = pyqtProperty(
        float,
        fset=lambda self, val:
            OWBaseWidget.progressBarSet(self, val, processEventsFlags=None),
        fget=progressBarValue
    )

    processingState = pyqtProperty(int, fget=lambda self: self.__progressState)

    def progressBarAdvance(self, value, processEventsFlags=QEventLoop.AllEvents):
        self.progressBarSet(self.progressBarValue + value, processEventsFlags)

    def progressBarFinished(self):
        """
        Reset and hide the progress bar.
        """
        self.setWindowTitle(self.captionTitle)
        if self.progressBarHandler:
            self.progressBarHandler(self, 101)

        if self.__progressState != 0:
            self.__progressState = 0
            self.processingStateChanged.emit(0)

    #: Widget's status message has changed.
    statusMessageChanged = pyqtSignal(unicode)

    def setStatusMessage(self, text):
        if self.__statusMessage != text:
            self.__statusMessage = text
            self.statusMessageChanged.emit(text)

    def statusMessage(self):
        return self.__statusMessage

    # handler must be a function, that receives 2 arguments.
    # First is the widget instance, the second is the value between
    # -1 and 101
    def setProgressBarHandler(self, handler):
        _deprecation_warning("setProgressBarHandler")
        self.progressBarHandler = handler

    def setProcessingHandler(self, handler):
        _deprecation_warning("setProcessingHandler")
        self.processingHandler = handler

    def setEventHandler(self, handler):
        _deprecation_warning("setEventHandler")
        self.eventHandler = handler

    def setWidgetStateHandler(self, handler):
        _deprecation_warning("setWidgetStateHandler")
        self.widgetStateHandler = handler

    # if we are in debug mode print the event into the file
    def printEvent(self, text, eventVerbosity=1):
        _deprecation_warning("printEvent")
        text = self.captionTitle + ": " + text

        if eventVerbosity > 0:
            _log.debug(text)
        else:
            _log.info(text)

        if self.eventHandler:
            self.eventHandler(text, eventVerbosity)

    def openWidgetHelp(self):
        _deprecation_warning("openWidgetHelp")
        if "widgetInfo" in self.__dict__ and hasattr(qApp, "canvasDlg"):
            # This widget is on a canvas.
            qApp.canvasDlg.helpWindow.showHelpFor(self.widgetInfo, True)

    def keyPressEvent(self, e):
        if e.key() in (Qt.Key_Help, Qt.Key_F1):
            if "widgetInfo" in self.__dict__ and hasattr(qApp, "canvasDlg"):
                self.openWidgetHelp()
        elif (int(e.modifiers()), e.key()) in OWBaseWidget.defaultKeyActions:
            OWBaseWidget.defaultKeyActions[int(e.modifiers()), e.key()](self)
        else:
            QDialog.keyPressEvent(self, e)

    def information(self, id=0, text=None):
        self.setState("Info", id, text)

    def warning(self, id=0, text=""):
        self.setState("Warning", id, text)

    def error(self, id=0, text=""):
        self.setState("Error", id, text)

    def setState(self, stateType, id, text):
        changed = 0
        if type(id) == list:
            for val in id:
                if self.widgetState[stateType].has_key(val):
                    self.widgetState[stateType].pop(val)
                    changed = 1
        else:
            if isinstance(id, basestring):
                # if we call information(), warning(), or error() function
                # with only one parameter - a string - then set id = 0
                text = id
                id = 0
            if not text:
                if self.widgetState[stateType].has_key(id):
                    self.widgetState[stateType].pop(id)
                    changed = 1
            else:
                self.widgetState[stateType][id] = text
                changed = 1

        if changed:
            if self.widgetStateHandler:
                self.widgetStateHandler()
            elif text:
                _log.info(stateType + " - " + text)

            if type(id) == list:
                for i in id:
                    self.emit(SIGNAL("widgetStateChanged(QString, int, QString)"),
                              QString(stateType), i,QString(""))
            else:
                self.emit(SIGNAL("widgetStateChanged(QString, int, QString)"),
                             QString(stateType), id, QString(text or ""))
        return changed

    widgetStateChanged = pyqtSignal(QString, int, QString)
    """Widget state has changed first arg is the state type
    ('Info', 'Warning' or 'Error') the second is the message id
    and finally the message string."""

    def widgetStateToHtml(self, info=True, warning=True, error=True):
        pixmaps = self.getWidgetStateIcons()
        items = [] 
        iconPath = {"Info": "canvasIcons:information.png",
                    "Warning": "canvasIcons:warning.png",
                    "Error": "canvasIcons:error.png"}
        for show, what in [(info, "Info"), (warning, "Warning"),(error, "Error")]:
            if show and self.widgetState[what]:
                items.append('<img src="%s" style="float: left;"> %s' % (iconPath[what], "\n".join(self.widgetState[what].values())))
        return "<br>".join(items)
        
    @classmethod
    def getWidgetStateIcons(cls):
        if not hasattr(cls, "_cached__widget_state_icons"):
            iconsDir = os.path.join(environ.canvas_install_dir, "icons")
            QDir.addSearchPath("canvasIcons",os.path.join(environ.canvas_install_dir,
                "icons/"))
            info = QPixmap("canvasIcons:information.png")
            warning = QPixmap("canvasIcons:warning.png")
            error = QPixmap("canvasIcons:error.png")
            cls._cached__widget_state_icons = \
                    {"Info": info, "Warning": warning, "Error": error}
        return cls._cached__widget_state_icons

    def synchronizeContexts(self):
        if hasattr(self, "contextHandlers"):
            for contextName, handler in self.contextHandlers.items():
                context = self.currentContexts.get(contextName, None)
                if context:
                    handler.settingsFromWidget(self, context)

    def openContext(self, contextName="", *arg):
        if not self._useContexts:
            return
        handler = self.contextHandlers[contextName]
        context = handler.openContext(self, *arg)
        if context:
            self.currentContexts[contextName] = context

    def closeContext(self, contextName=""):
        if not self._useContexts:
            return
        curcontext = self.currentContexts.get(contextName)
        if curcontext:
            self.contextHandlers[contextName].closeContext(self, curcontext)
            del self.currentContexts[contextName]

    def settingsToWidgetCallback(self, handler, context):
        pass

    def settingsFromWidgetCallback(self, handler, context):
        pass

    def setControllers(self, obj, controlledName, controller, prefix):
        while obj:
            if prefix:
#                print "SET CONTROLLERS: %s %s + %s" % (obj.__class__.__name__, prefix, controlledName)
                if obj.__dict__.has_key("attributeController"):
                    obj.__dict__["__attributeControllers"][(controller, prefix)] = True
                else:
                    obj.__dict__["__attributeControllers"] = {(controller, prefix): True}

            parts = controlledName.split(".", 1)
            if len(parts) < 2:
                break
            obj = getattr(obj, parts[0], None)
            prefix += parts[0]
            controlledName = parts[1]

    def __setattr__(self, name, value):
        return unisetattr(self, name, value, QDialog)
    
    defaultKeyActions = {}
    
    if sys.platform == "darwin":
        defaultKeyActions = {
            (Qt.ControlModifier, Qt.Key_M): lambda self: self.showMaximized if self.isMinimized() else self.showMinimized(),
            (Qt.ControlModifier, Qt.Key_W): lambda self: self.setVisible(not self.isVisible())}

    def scheduleSignalProcessing(self):
        """
        Schedule signal processing by the signal manager.

        ..note:: The processing is already scheduled at the most appropriate
                 time so you should have few uses for this method.
        """
        _deprecation_warning("scheduleSignalProcessing")
        if self.signalManager is not None:
            self.signalManager.scheduleSignalProcessing(self)

    def setBlocking(self, state=True):
        """ Set blocking flag for this widget. While this flag is set this
        widget and all its descendants will not receive any new signals from
        the signal manager
        """
        self.asyncBlock = state
        self.emit(SIGNAL("blockingStateChanged(bool)"), self.asyncBlock)
        if not self.isBlocking():
            self.scheduleSignalProcessing()

    def isBlocking(self):
        """ Is this widget blocking signal processing. Widget is blocking if
        asyncBlock value is True or any AsyncCall objects in asyncCalls list
        has blocking flag set
        """
        return self.asyncBlock or any(a.blocking for a in self.asyncCalls)

    def asyncExceptionHandler(self, (etype, value, tb)):
        sys.excepthook(etype, value, tb)

    def asyncFinished(self, async, string):
        """ Remove async from asyncCalls, update blocking state
        """
        index = self.asyncCalls.index(async)
        async = self.asyncCalls.pop(index)

        if async.blocking and not self.isBlocking():
            # if we are responsible for unblocking
            self.emit(SIGNAL("blockingStateChanged(bool)"), False)
            self.scheduleSignalProcessing()

        async.disconnect(async, SIGNAL("finished(PyQt_PyObject, QString)"), self.asyncFinished)
        self.emit(SIGNAL("asyncCallsStateChange()"))

    def asyncCall(self, func, args=(), kwargs={}, name=None, onResult=None, onStarted=None, onFinished=None, onError=None, blocking=True, thread=None, threadPool=None):
        """ Return an OWConcurent.AsyncCall object func, args and kwargs
        set and signals connected. 
        """
        _deprecation_warning("asyncCall")
        from functools import partial
        from OWConcurrent import AsyncCall
        
        asList = lambda slot: slot if isinstance(slot, list) else ([slot] if slot else [])
        
        onResult = asList(onResult)
        onStarted = asList(onStarted) #+ [partial(self.setBlocking, True)]
        onFinished = asList(onFinished) #+ [partial(self.blockSignals, False)]
        onError = asList(onError) or [self.asyncExceptionHandler]
        
        async = AsyncCall(func, args, kwargs, thread=thread, threadPool=threadPool)
        async.name = name if name is not None else ""
            
        for slot in  onResult:
            async.connect(async, SIGNAL("resultReady(PyQt_PyObject)"), slot, Qt.QueuedConnection)
        for slot in onStarted:
            async.connect(async, SIGNAL("starting()"), slot, Qt.QueuedConnection)
        for slot in onFinished:
            async.connect(async, SIGNAL("finished(QString)"), slot, Qt.QueuedConnection)
        for slot in onError:
            async.connect(async, SIGNAL("unhandledException(PyQt_PyObject)"), slot, Qt.QueuedConnection)
        
        self.addAsyncCall(async, blocking)
            
        return async
    
    def addAsyncCall(self, async, blocking=True):
        """ Add AsyncCall object to asyncCalls list (will be removed
        once it finishes processing).
        
        """
        _deprecation_warning("addAsyncCall")
        
        async.connect(async, SIGNAL("finished(PyQt_PyObject, QString)"), self.asyncFinished)
        
        async.blocking = blocking
        
        if blocking:
            # if we are responsible for blocking
            state = any(a.blocking for a in self.asyncCalls)
            self.asyncCalls.append(async)
            if not state:
                self.emit(SIGNAL("blockingStateChanged(bool)"), True)
        else:
            self.asyncCalls.append(async)
            
        self.emit(SIGNAL("asyncCallsStateChange()"))


def blocking(method):
    """ Return method that sets blocking flag while executing
    """
    from functools import wraps

    @wraps(method)
    def wrapper(self, *args, **kwargs):
        old = self._blocking
        self.setBlocking(True)
        try:
            return method(self, *args, **kwargs)
        finally:
            self.setBlocking(old)

if __name__ == "__main__":
    a = QApplication(sys.argv)
    oww = OWBaseWidget()
    oww.show()
    a.exec_()
    oww.saveSettings()
