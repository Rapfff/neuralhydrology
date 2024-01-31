class RoutingReservoir:
    def __init__(self, rate) -> None:
        self.rate = rate
        self.reset()
    
    def __call__(self, p) -> float:
        return self.forward(p)
    
    def reset(self) -> None:
        self.storage = 0.0
    
    def forward(self, p) -> float:
        p *= p>0
        self.storage += p
        q = self.rate * self.storage
        q = min(q, self.storage)
        self.storage -= q
        return q

class LagFunction:
    def __init__(self, time_steps:int) -> None:
        self.time_steps = time_steps
        self.reset()

    def __call__(self, p) -> float:
        return self.forward(p)

    def reset(self) -> None:
        self.storage = [0.0 for i in range(self.time_steps)]

    def forward(self, p) -> float:
        out = self.storage[-1]
        self.storage = [p]+self.storage[:-1]
        return out
    
class Network:
    def __init__(self,k,ts) -> None:
        self.r = RoutingReservoir(k)
        self.l = LagFunction(ts)
    
    def __call__(self, p) -> float:
        return self.forward(p)
    
    def reset(self) -> None:
        self.r.reset()
        self.l.reset()

    def forward(self,p) -> float:
        return self.l(self.r(p))


def generate_data(name,k,ts):
    precip = []
    dates = []
    with open("../../data/fake/basin_mean_forcing/daymet/02/"+name+"_lump_cida_forcing_leap.txt") as f:
        for _ in range(5):
            l = f.readline()
        while l:
            l = l.split('\t')
            precip.append(float(l[2]))
            dates.append(l[0])
            l = f.readline()
    network = Network(k,ts)
    with open("../../data/fake/usgs_streamflow/02/"+name+"_streamflow_qc.txt",'w') as f:
        for d,p in zip(dates, precip):
            f.write(' '.join([name, d[:-3], str(network(p)), 'A']))
            f.write('\n')

generate_data('01510000',0.1,6)
generate_data('01516500',0.3,6)