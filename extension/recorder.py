from collections import defaultdict

class DictMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.9):
        self.momentum = momentum
        self.data = defaultdict(lambda: defaultdict(float))
        self.count = defaultdict(int)

    def reset(self):
        self.data = defaultdict(lambda: defaultdict(float))
        self.count = defaultdict(int)

    def update(self, val: dict, n=1):

        for k, v in val.items():
            v = v.item()
            self.data[k]['val'] = v
            self.data[k]['ravg'] = self.momentum * self.data[k]['ravg'] + (1.0 - self.momentum) * v
            self.data[k]['sum'] += v * n
            self.count[k] += n

    @property
    def value(self):
        return ', '.join(['{}={:6.3f}'.format(k, self.data[k]['val']) for k in self.data.keys()])

    @property
    def average(self):
        return ', '.join(['{}={:6.3f}'.format(k, self.data[k]['sum'] / self.count[k]) for k in self.data.keys()])

    @property
    def running_average(self):
        return ', '.join(['{}={:6.3f}'.format(k, self.data[k]['ravg']) for k in self.data.keys()])

    @property
    def sum(self):
        return ', '.join(['{}={:6.3f}'.format(k, self.data[k]['sum']) for k in self.data.keys()])

    def get_average(self):
        return {k: self.data[k]['sum'] / self.count[k] for k in self.data.keys()}

    def get_sum(self):
        return {k: self.data[k]['sum'] for k in self.data.keys()}
