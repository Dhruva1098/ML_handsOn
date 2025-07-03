class Set:
    def __init__(self, values=None):
        self.dict = {}
        if values is not None:
            for value in values:
                self.add(value)
    def __repr__(self):
        return "Set: " + str(self.dict.keys())

    def add(self, value):
        self.dict[value] = True

    def contains(self, value):
        return value in self.dict
    
    def remove(self,value):
        del self.dict[value]


s = Set([1,2,3])
s.add(4)
print(s.contains(4))
s.remove(3)
print(s.contains(3))
