# coding=utf-8
"""
<name>SVM Regression</name>
<description>Support Vector Machine Regression.</description>
<icon>icons/SVMRegression.svg</icon>
<contact>Ales Erjavec (ales.erjavec(@at@)fri.uni-lj.si)</contact>
<priority>100</priority>
<keywords>Support, Vector, Machine, Regression</keywords>

"""

import orngSVM

from OWSVM import *

import Orange
from Orange.classification import svm

class OWSVMRegression(OWSVM):
    settingsList = OWSVM.settingsList + ["C_epsilon", "C_nu"]

    def __init__(self, parent=None, signalManager=None, title="SVM Regression"):
        OWWidget.__init__(self, parent, signalManager, title,
                          wantMainArea=False, resizingEnabled=False)

        self.inputs=[("Data", Orange.data.Table, self.setData), 
                     ("Preprocess", PreprocessedLearner, self.setPreprocessor)]

        self.outputs=[("Learner", orange.Learner, Default),
                      ("Predictor", orange.Classifier, Default),
                      ("Support Vectors", Orange.data.Table)]

        ##########
        # Settings
        ##########
        self.kernel_type = 2
        self.gamma = 0.0
        self.coef0 = 0.0
        self.degree = 3
        self.C_epsilon = 1.0
        self.C_nu = 1.0
        self.p = 0.1
        self.eps = 1e-3
        self.nu = 0.5
        self.shrinking = 1
        self.probability = 1
        self.useNu = 0
        self.nomogram = 0
        self.normalization = 1
        self.data = None
        self.selFlag = False
        self.preprocessor = None
        self.name = "SVM Regression"

        self.loadSettings()

        OWGUI.lineEdit(self.controlArea, self, 'name',
                       box='Learner/predictor Name',
                       tooltip='Name to be used by other widgets to identify your learner/predictor.')
        OWGUI.separator(self.controlArea)

        b = OWGUI.radioButtonsInBox(self.controlArea, self, "useNu", [], 
                                    box="SVM Type",
                                    orientation = QGridLayout(),
                                    addSpace=True)

        # Epsilon SVR
        b.layout().addWidget(OWGUI.appendRadioButton(b, self, 
                                                "useNu", u"??-SVR",
                                                tooltip="Epsilon SVR",
                                                addToLayout=False),
                             0, 0, Qt.AlignLeft)
        
        b.layout().addWidget(QLabel("Cost (C)", b), 0, 1, Qt.AlignRight)
        b.layout().addWidget(OWGUI.doubleSpin(b, self, "C_epsilon", 0.1, 512.0, 0.1,
                          decimals=2,
                          addToLayout=False,
                          callback=lambda *x: self.setType(0),
                          alignment=Qt.AlignRight,
                          tooltip="Cost for out of epsilon training points."),
                          0, 2)
    
        b.layout().addWidget(QLabel(u"Loss Epsilon (??)", b), 1, 1, Qt.AlignRight)
        b.layout().addWidget(OWGUI.doubleSpin(b, self, "p", 0.05, 1.0, 0.05,
                                      addToLayout=False,
                                      callback=lambda *x: self.setType(0),
                                      alignment=Qt.AlignRight,
                                      tooltip="Epsilon bound (all points inside this interval are not penalized)."
                                      ),
                             1, 2)

        # Nu SVR
        b.layout().addWidget(OWGUI.appendRadioButton(b, self,
                                                "useNu", u"??-SVR",
                                                tooltip="Nu SVR",
                                                addToLayout=False),
                             2, 0, Qt.AlignLeft)
        
        b.layout().addWidget(QLabel("Cost (C)", b),
                             2, 1, Qt.AlignRight)
        b.layout().addWidget(OWGUI.doubleSpin(b, self, "C_nu", 0.1, 512.0, 0.1,
                        decimals=2, 
                        addToLayout=False,
                        callback=lambda *x: self.setType(1),
                        alignment=Qt.AlignRight,
                        tooltip="Cost for out of epsilon training points."),
                        2, 2)
        
        b.layout().addWidget(QLabel(u"Complexity bound (\u03bd)", b),
                             3, 1, Qt.AlignRight)
        b.layout().addWidget(OWGUI.doubleSpin(b, self, "nu", 0.05, 1.0, 0.05,
                        tooltip="Lower bound on the ratio of support vectors",
                        addToLayout=False, 
                        callback=lambda *x: self.setType(1), 
                        alignment=Qt.AlignRight),
                        3, 2)
        
        # Kernel
        self.kernelBox=b = OWGUI.widgetBox(self.controlArea, "Kernel")
        self.kernelradio = OWGUI.radioButtonsInBox(b, self, "kernel_type", 
                                btnLabels=[u"Linear,   x???y", 
                                           u"Polynomial,   (g x???y + c)^d",
                                           u"RBF,   exp(-g|x-y|??)", 
                                           u"Sigmoid,   tanh(g x???y + c)"],
                                callback=self.changeKernel)

        OWGUI.separator(b)
        self.gcd = OWGUI.widgetBox(b, orientation="horizontal")
        self.leg = OWGUI.doubleSpin(self.gcd, self, "gamma", 0.0, 10.0, 0.0001,
                                    decimals=5,
                                    label="  g: ", orientation="horizontal",
                                    callback=self.changeKernel, 
                                    alignment=Qt.AlignRight)
        
        self.led = OWGUI.doubleSpin(self.gcd, self, "coef0", 0.0, 10.0, 0.0001,
                                    label="  c: ", orientation="horizontal", 
                                    callback=self.changeKernel, 
                                    alignment=Qt.AlignRight)
        
        self.lec = OWGUI.doubleSpin(self.gcd, self, "degree", 0.0,10.0,0.5, 
                                    label="  d: ", orientation="horizontal", 
                                    callback=self.changeKernel, 
                                    alignment=Qt.AlignRight)

        OWGUI.separator(self.controlArea)
        
        self.optionsBox=b=OWGUI.widgetBox(self.controlArea, "Options", addSpace=True)
        
        OWGUI.doubleSpin(b,self, "eps", 0.0005, 1.0, 0.0005, 
                         label=u"Numerical tolerance", 
                         labelWidth = 180, 
                         orientation="horizontal",
                         tooltip="Numerical tolerance of termination criterion.", 
                         alignment=Qt.AlignRight)

        OWGUI.checkBox(b, self, "normalization", 
                       label="Normalize data", 
                       tooltip="Use data normalization")

        self.paramButton = OWGUI.button(self.controlArea, self,
                                         "Automatic parameter search", 
                                         callback=self.parameterSearch,
                                         tooltip="Automatically searches for parameters that optimize classifier accuracy", 
                                         debuggingEnabled=0)
        
        self.paramButton.setDisabled(True)

        OWGUI.button(self.controlArea, self,"&Apply", 
                     callback=self.applySettings, 
                     default=True)
        
        OWGUI.rubber(self.controlArea)
        
        
        self.changeKernel()
        self.searching=False
        self.applySettings()

    def setData(self, data=None):
        self.data = self.isDataWithClass(data, 
                    wantedVarType=Orange.core.VarTypes.Continuous,
                    checkMissing=True) and data or None
        self.paramButton.setDisabled(not self.data)
        
    def applySettings(self):
        if self.useNu:
            svm_type = svm.SVMLearner.Nu_SVR
            C = self.C_nu
        else:
            svm_type = svm.SVMLearner.Epsilon_SVR
            C = self.C_epsilon

        learner = svm.SVMLearner(svm_type=svm_type,
                                 C=C,
                                 p=self.p,
                                 nu=self.nu,
                                 kernel_type=self.kernel_type,
                                 gamma=self.gamma,
                                 degree=self.degree,
                                 coef0=self.coef0,
                                 eps=self.eps,
                                 probability=self.probability,
                                 normalization=self.normalization,
                                 name=self.name)
        predictor = None
        support_vectors = None
        if self.preprocessor:
            learner = self.preprocessor.wrapLearner(learner)

        if self.data is not None:
            predictor = learner(self.data)
            support_vectors = predictor.support_vectors
            predictor.name = self.name

        self.send("Learner", learner)
        self.send("Predictor", predictor)
        self.send("Support Vectors", support_vectors)

    def sendReport(self):
        if self.useNu:
            settings = [("Type", "Nu SVM regression"),
                        ("Cost (C)", "%.3f" % self.C_nu),
                        ("Complexity bound (nu)", "%.3f" % self.nu)]
        else:
            settings = [("Type", "Epsilon SVM regression"),
                        ("Cost (C)", "%.3f" % self.C_epsilon),
                        ("Loss epsilon", "%.3f" % self.p)]

        if self.kernel_type == 0:
            kernel = "Linear, x.y"
        elif self.kernel_type == 1:
            kernel = "Polynomial, (%.4f*x.y+%.4f)<sup>%.4f</sup>" % (self.gamma, self.coef0, self.degree)
        elif self.kernel_type == 2:
            kernel = "RBF, e<sup>-%.4f*(x-y).(x-y)</sup>" % self.gamma
        else:
            kernel = "Sigmoid, tanh(%.4f*x.y+%.4f)" % (self.gamma, self.coef0)

        settings.extend([("Kernel", kernel),
                         ("Tolerance", self.eps),
                         ("Normalize data", OWGUI.YesNo[self.normalization])])

        self.reportSettings("Settings", settings)
        self.reportData(self.data)

    def search_(self):
        learner = orngSVM.SVMLearner()
        for attr in ("name", "kernel_type", "degree", "shrinking", "probability", "normalization"):
            setattr(learner, attr, getattr(self, attr))

        for attr in ("gamma", "coef0", "p", "eps", "nu"):
            setattr(learner, attr, float(getattr(self, attr)))

        if self.useNu:
            learner.svm_type = learner.Nu_SVR
            learner.C = self.C_nu
        else:
            learner.svm_type = learner.Epsilon_SVR
            learner.C = self.C_epsilon

        params = []        
        params.append("C")
        if self.kernel_type in [1, 2]:
            params.append("gamma")
        if self.kernel_type == 1:
            params.append("degree")
        try:
            learner.tuneParameters(self.data, params, 4, verbose=0,
                                   progressCallback=self.progres)
        except UnhandledException:
            pass
        for param in params:
            setattr(self, param, getattr(learner, param))

        if self.useNu:
            self.C_nu = learner.C
        else:
            self.C_epsilon = learner.C

        self.finishSearch()

if __name__ == "__main__":
    app = QApplication([])
    w = OWSVMRegression()
    w.show()
    data = Orange.data.Table("housing")
    w.setData(data)
    app.exec_()
