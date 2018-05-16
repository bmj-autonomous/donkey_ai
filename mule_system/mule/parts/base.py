import abc
import threading
import time


class BasePart(abc.ABC):
    ''' Common interface for vehicle parts '''
    @abc.abstractmethod
    def start(self):
        ''' Starts part components '''
        pass


    @abc.abstractmethod
    def transform(self, state):
        ''' Transforms vehicle state

            Arguments

            state: dict
                stores elements comprising vehicle state, for example,
                latest image received from camera
        '''
        pass


    @abc.abstractmethod
    def stop(self):
        ''' Stops part components '''
        pass


class ThreadComponent:
    ''' Component held by part that added threading capability

        Apart from knowing its own state of activity, it holds a copy of the target
        function that is responsible for updating the part's state.
    '''
    def __init__(self, target, **kwargs):
        ''' 
            Arguments

            target: function
                function that updates part's state

            kwargs: 
                arguments to be passed to target

        '''
        self.running = False
        self.target = target
        self.thread = threading.Thread(target=self._threaded_target, kwargs=kwargs)


    def start(self):
        ''' Starts thread '''
        self.running = True
        self.thread.start()


    def stop(self):
        ''' Causes self._threaded_target to return, thereby stopping the thread '''
        self.running = False
        time.sleep(1)


    def _threaded_target(self, **kwargs):
        ''' Wraps target function, generally the _update function from a part
            so that updating continues indefinitely unless the thread is stopped
        '''
        while self.running:
            self.target(**kwargs)

