"""
<name>Data Sampler</name>
<description>Randomly selects a subset of instances from the data set</description>
<icon>icons/DataSamplerA.svg</icon>
<priority>10</priority>
"""
# [start-snippet-1]
import Orange
from OWWidget import *
import OWGUI

class OWDataSamplerA(OWWidget):

    def __init__(self, parent=None, signalManager=None):
        OWWidget.__init__(self, parent, signalManager)

        self.inputs = [("Data", Orange.data.Table, self.set_data)]
        self.outputs = [("Sampled Data", Orange.data.Table)]

        # GUI
        box = OWGUI.widgetBox(self.controlArea, "Info")
        self.infoa = OWGUI.widgetLabel(box, 'No data on input yet, waiting to get something.')
        self.infob = OWGUI.widgetLabel(box, '')
        self.resize(100,50)
# [end-snippet-1]

# [start-snippet-2]
    def set_data(self, dataset):
        if dataset is not None:
            self.infoa.setText('%d instances in input data set' % len(dataset))
            indices = Orange.data.sample.SubsetIndices2(p0=0.1)
            ind = indices(dataset)
            sample = dataset.select(ind, 0)
            self.infob.setText('%d sampled instances' % len(sample))
            self.send("Sampled Data", sample)
        else:
            self.infoa.setText('No data on input yet, waiting to get something.')
            self.infob.setText('')
            self.send("Sampled Data", None)
# [end-snippet-2]

# [start-snippet-3]
if __name__=="__main__":
    appl = QApplication(sys.argv)
    ow = OWDataSamplerA()
    ow.show()
    dataset = Orange.data.Table('iris.tab')
    ow.set_data(dataset)
    appl.exec_()
# [end-snippet-3]