import time


# TODO: could implement as a loop regulator, which would reduce code in 
#       guts of loop and clean up the Regulator internals
class Regulator:
    ''' Regulates any segment 

        Users should mark two points in the iteration, call regulate to 
        extend the time

    '''
    def __init__(self, hertz):
        ''' Arguments

            hertz: float
                number of steps/loops/revolutions etc to perform per second

            Members
            
            interval: float
                time interval to regulate

            marks: list
                contains marked time points

        '''
        self.interval = 1 / hertz
        self.marks = []

    def mark(self):
        ''' Marks a time instance '''
        self.marks.append(time.time())

    def regulate(self):
        if len(self.marks) == 2:
            margin = self.interval - self.marks[1] + self.marks[0]
            if margin > 0.0:
                time.sleep(margin)
            self.marks = []

        else:
            msg = 'Two time marks required for regulator, {} present.'
            raise ValueError(msg.format(len(self.marks)))



