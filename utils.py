import random

class Func(dict):

    def __getitem__(self, key):
        self.setdefault(key, 0)
      
        return dict.__getitem__(self, key)
    
    def argmax(self, prefix, postfix):
        m = float('-inf')
        p = None

        options = []
        for i, post in enumerate(postfix):
            val = self.__getitem__(prefix + (post, ))
            if val > m:
                options = [i]
                m = val
                p = i
            elif val == m:
                options.append(i)
        return p#random.choice(options)
    
    def max(self, prefix, postfix):
        m = 0
        for post in postfix:

            # print(prefix + (post, ))
            val = self.__getitem__(prefix + (post, ))
            if val > m:
                m = val
        return m
