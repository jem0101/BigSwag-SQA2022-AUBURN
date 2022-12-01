import logging

def getSQALogger():
    logging.basicConfig(filename='app.log', filemode='a', format='%(name)s - %(levelname)s - %(message)s', level=logging.INFO)
    loggerOBJ = logging.getLogger('SQA-Logger')
    loggerOBJ.setLevel(logging.DEBUG)
    return loggerOBJ