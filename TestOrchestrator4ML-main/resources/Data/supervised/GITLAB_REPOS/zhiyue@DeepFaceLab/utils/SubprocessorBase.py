import traceback
from tqdm import tqdm
import multiprocessing
import time
import sys

class SubprocessorBase(object):

    #overridable
    def __init__(self, name, no_response_time_sec = 60):
        self.name = name
        self.no_response_time_sec = no_response_time_sec
        self.is_host = True
        
    #overridable    
    def process_info_generator(self):
        #yield name, host_dict, client_dict - per process
        yield 'first process', {}, {}
        
    #overridable
    def get_no_process_started_message(self):
        return "No process started."

    #overridable
    def onHostGetProgressBarDesc(self):
        return "Processing"
        
    #overridable
    def onHostGetProgressBarLen(self):
        return 0
        
    #overridable        
    def onHostGetData(self, host_dict):
        #return data here
        return None
    
    #overridable
    def onHostDataReturn (self, host_dict, data):
        #input_data.insert(0, obj['data'])   
        pass
    
    #overridable
    def onClientInitialize(self, client_dict):
        #return fail message or None if ok
        return None
        
    #overridable
    def onClientFinalize(self):
        pass
        
    #overridable
    def onClientProcessData(self, data):
        #return result object
        return None  
        
    #overridable
    def onClientGetDataName (self, data):
        #return string identificator of your data
        return "undefined"
        
    #overridable
    def onHostClientsInitialized(self):
        pass
        
    #overridable
    def onHostResult (self, host_dict, data, result):
        #return count of progress bar update
        return 1
    
    #overridable
    def onFinalizeAndGetResult(self):
        return None
        
    def inc_progress_bar(self, c):        
        if self.is_host:
            self.progress_bar.n += c
            self.progress_bar.refresh()
        else:
            self.cq.put ( {'op': 'inc_bar', 'c':c} )
    
    def safe_print(self, msg):
        self.print_lock.acquire()
        print (msg)
        self.print_lock.release()
    
    def process(self):
        #returns start_return
        
        self.processes = []
        
        self.print_lock = multiprocessing.Lock()        
        for name, host_dict, client_dict in self.process_info_generator():            
            sq = multiprocessing.Queue()
            cq = multiprocessing.Queue()
            
            client_dict.update ( {'print_lock' : self.print_lock} ) 
            try:
                p = multiprocessing.Process(target=self.subprocess, args=(sq,cq,client_dict))
                p.daemon = True
                p.start()
                self.processes.append ( { 'process' : p,
                                          'sq' : sq,
                                          'cq' : cq,
                                          'state' : 'busy',
                                          'sent_time': time.time(),
                                          'name': name,
                                          'host_dict' : host_dict
                                        } )
            except:
                print ("Unable to start subprocess %s" % (name))

        if len(self.processes) == 0:
            raise Exception ("Unable to start Subprocessor '%s' " % (self.name))

        while True:
            for p in self.processes[:]:
                while not p['cq'].empty():
                    obj = p['cq'].get()
                    obj_op = obj['op']
                        
                    if obj_op == 'init_ok':
                        p['state'] = 'free'
                    elif obj_op == 'error':                       
                        if obj['close'] == True:
                            p['process'].terminate()
                            p['process'].join()
                            self.processes.remove(p)
                            break                
                    
            if all ([ p['state'] == 'free' for p in self.processes ] ):
                break
                
        if len(self.processes) == 0:
            print ( self.get_no_process_started_message() )
            return None
        
        self.progress_bar = tqdm( total=self.onHostGetProgressBarLen(), desc=self.onHostGetProgressBarDesc(), ascii=True )                  
        self.onHostClientsInitialized()        
        
        try: 
            while True:
                for p in self.processes[:]:
                    while not p['cq'].empty():
                        obj = p['cq'].get()
                        obj_op = obj['op']
                        
                        if obj_op == 'success':   
                            data = obj['data']
                            result = obj['result']
                            
                            c = self.onHostResult (p['host_dict'], data, result)                        
                            if c > 0:
                                self.progress_bar.update(c)
                                
                            p['state'] = 'free'
                        elif obj_op == 'inc_bar':                            
                            self.inc_progress_bar(obj['c'])
                        elif obj_op == 'error':
                            if 'data' in obj.keys():
                                self.onHostDataReturn (p['host_dict'], obj['data'] )
            
                            if obj['close'] == True:
                                p['sq'].put ( {'op': 'close'} )
                                p['process'].join()
                                self.processes.remove(p)
                                break
                            p['state'] = 'free'
                            
                for p in self.processes[:]:
                    if p['state'] == 'free':
                        data = self.onHostGetData(p['host_dict'])
                        if data is not None:
                            p['sq'].put ( {'op': 'data', 'data' : data} )
                            p['sent_time'] = time.time()
                            p['sent_data'] = data
                            p['state'] = 'busy'
                        
                    elif p['state'] == 'busy':
                        if self.no_response_time_sec != 0 and (time.time() - p['sent_time']) > self.no_response_time_sec:
                            print ( '%s doesnt response, terminating it.' % (p['name']) )
                            self.onHostDataReturn (p['host_dict'], p['sent_data'] )
                            p['process'].terminate()
                            self.processes.remove(p)
                            
                if all ([p['state'] == 'free' for p in self.processes]):
                    break
                    
                time.sleep(0.005)
        except:
            print ("Exception occured in Subprocessor.start(): %s" % (traceback.format_exc()) )
        
        self.progress_bar.close()
        
        for p in self.processes[:]:
            p['sq'].put ( {'op': 'close'} )
            p['sent_time'] = time.time()
            
        while True:
            for p in self.processes[:]:
                terminate_it = False
                while not p['cq'].empty():
                    obj = p['cq'].get()
                    obj_op = obj['op']                    
                    if obj_op == 'finalized':
                        terminate_it = True
                        break
                        
                if self.no_response_time_sec != 0 and (time.time() - p['sent_time']) > self.no_response_time_sec:                
                    terminate_it = True
                    
                if terminate_it:
                    p['state'] = 'finalized'
                    p['process'].terminate()
                    
            if all ([p['state'] == 'finalized' for p in self.processes]):
                break
         
        return self.onFinalizeAndGetResult()

    def subprocess(self, sq, cq, client_dict):    
        self.is_host = False
        self.print_lock = client_dict['print_lock']
        self.cq = cq
        try:
            fail_message = self.onClientInitialize(client_dict)
        except:
            fail_message = 'Exception while initialization: %s' % (traceback.format_exc())
    
        if fail_message is None:
            cq.put ( {'op': 'init_ok'} )
        else:
            print (fail_message)
            cq.put ( {'op': 'error', 'close': True} )
            return 

        while True:
            obj = sq.get()
            obj_op = obj['op']

            if obj_op == 'data':
                data = obj['data']
                try:
                    result = self.onClientProcessData (data)
                    cq.put ( {'op': 'success', 'data' : data, 'result' : result} )                    
                except:
                    
                    print ( 'Exception while process data [%s]: %s' % (self.onClientGetDataName(data), traceback.format_exc()) )
                    cq.put ( {'op': 'error', 'close': True, 'data' : data } )
            elif obj_op == 'close':
                break
                
            time.sleep(0.005)
                
        self.onClientFinalize()
        cq.put ( {'op': 'finalized'} )
        while True:
            time.sleep(0.1)