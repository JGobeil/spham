from time import perf_counter


class TicToc:
    level = 0
    
    @classmethod
    def _prein(cls):
        l = cls.level
        if l > 0:
            return (">"*l) + " "
        else:
            return ""
        
    @classmethod
    def _preout(cls):
        l = cls.level
        if l > 0:
            return ("<"*l) + " "
        else:
            return ""
    
    def __init__(self, timing=None):
        if timing is None:
            self.timing = {}
        else:
            self.timing = timing.copy()
        
    def tic(self, name):
        self.timing[name] = perf_counter()
        print("%sCalculating %s" % (TicToc._prein(), name))
        TicToc.level = TicToc.level + 1
        
    def toc(self, name):
        self.timing[name] = perf_counter() - self.timing[name]
        TicToc.level = TicToc.level - 1
        print("%s%s calculation done in %s" % 
              (TicToc._preout(), name, self._fmt(self.timing[name])))
        
    @staticmethod
    def _fmt(s):
        if s < 1:
            return "%.3gms" % (s*1000)
        if s < 60:
            return "%.3gs" % s
        m, s = divmod(s, 60)
        if m < 60:
            return "%2dm %.3fs" % (m, s)
        h, m = divmod(m, 60)
        return "%dh %2dm %.1fs" % (h, m, s)
    
    def nice_timing(self, name=None):
        if name is None:
            return "\n".join([
                "%20s: %s" % (name, self.nice_timing(name))
                for name in self.timing.keys()])
        else:
            return self._fmt(self.timing[name])
