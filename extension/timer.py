import time


class TimeMeter:
    '''
    A time counter class
    '''
    def __init__(self, start_epoch=0, epochs=1, num_steps=1):
        self.last_time = time.time()
        self.total_steps = (epochs - start_epoch) * num_steps
        self.steps = 0
        self.used = 0
        self.total = 0
    
    def reset(self):
        self.steps = 0
        self.used = 0
        self.total = 0

    def start(self):
        '''
        set time
        '''
        self.last_time = time.time()

    def step(self, repeat=1):
        '''
        update time
        '''
        self.steps += repeat
        self.used = time.time() - self.last_time
        self.last_time = time.time()
        self.total += self.used

    def format_time(self, second):
        days = int(second / 3600 / 24)
        second = second - days * 3600 * 24
        hours = int(second / 3600)
        second = second - hours * 3600
        minutes = int(second / 60)
        second = second - minutes * 60
        seconds = int(second)
        millisecond = 1000 * (second - seconds)

        if days > 0:
            return '{:2d}D{:02}h'.format(days, hours)
        elif hours > 0:
            return "{:2d}h{:02d}m".format(hours, minutes)
        elif minutes > 0:
            return "{:2d}m{:02d}s".format(minutes, seconds)
        elif seconds > 0:
            return "{:2d}s{:03d}".format(seconds, int(millisecond))
        else:
            return '{:.4f}'.format(millisecond)[:4] + "ms"

    @property
    def value(self):
        return self.format_time(self.used)

    @property
    def average(self):
        return self.total / max(1e-5, self.steps)

    @property
    def avg_str(self):
        return self.format_time(self.total / max(1e-5, self.steps))

    @property
    def sum(self):
        return self.format_time(self.total)

    @property
    def expect(self):
        return self.format_time((self.total_steps - self.steps) * (self.total / max(1e-5, self.steps)))
