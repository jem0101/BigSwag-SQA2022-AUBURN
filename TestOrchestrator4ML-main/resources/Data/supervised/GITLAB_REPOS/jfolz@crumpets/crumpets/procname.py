try:
    from procname import setprocname
except ImportError:
    def setprocname(*_, **__):
        pass
