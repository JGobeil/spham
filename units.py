from scipy.constants import physical_constants
from collections import UserDict

class UnitSystem(UserDict):
    """ class to deal with unit system with unit conversion. By default active 'atomic units' (AU).
    
    Unit system availible: AU, SI
    
    for example:
        bias_AU = 1.3 * units.eV  # 1.3 eV in AU
        bias_AU = 1.3 * units["meV"]  # 1.3 meV in AU
    
    # to convert back
        bias_SI = bias_AU / units.eV  # to eV
        bias_SI = bias_AU / units["meV"]  # to meV
    """
    
    def __init__(self, unit_system='AU'):
        super().__init__()
        pc = physical_constants

        """ Some constant used for calculation """
        self["me [SI]"] = pc["electron mass"][0]
        self["e [SI]"] = pc["elementary charge"][0]
        self["ħ [SI]"] = pc["Planck constant over 2 pi"][0]
        self["ke [SI]"] = pc["electric constant"][0]
        self["kB [SI]"] = pc["Boltzmann constant"][0]
        self["μB [SI]"] = pc["Bohr magneton"][0]
        
        self["Eh [SI]"] = pc["hartree-joule relationship"][0]
        self["eh [SI]"] = self["e [SI]"] / self["ħ [SI]"]
        
        self["J to SI"] = 1.0
        self["eV to SI"] = pc["electron volt"][0]
        self["T to SI"] = 1.0 # Tesla
        self["K to SI"] = 1.0  # Kelvin
        self["V to SI"] = 1.0 # Potential
        self["A to SI"] = 1.0  # Current
        self["S to SI"] = 1.0 # Conductance
        self["seconds to SI"] = 1.0 # Conductance
        
        self["J to AU"] = pc['joule-hartree relationship'][0]
        self["eV to AU"] = pc['joule-hartree relationship'][0] / pc['joule-electron volt relationship'][0]
        self["T to AU"] = 1/pc["atomic unit of mag. flux density"][0] # Tesla
        self["K to AU"] = pc["kelvin-hartree relationship"][0]  # Kelvin
        self["V to AU"] = self["e [SI]"] / self["Eh [SI]"] # Potential
        self["A to AU"] = self["ħ [SI]"] /(self["e [SI]"] * self["Eh [SI]"])  # Current
        self["S to AU"] = self["ħ [SI]"] / (self["e [SI]"] ** 2) # Conductance
        self["s to AU"] = self["Eh [SI]"] / self["ħ [SI]"] # times

        self["me [AU]"]  = 1.0
        self["e [AU]"] = 1.0
        self["ħ [AU]"] = 1.0
        self["ke [AU]"]  = 1.0
        self["kB [AU]"]  = 1.0
        self["μB [AU]"]  = 0.5

        self.set_units(unit_system)

    def set_units(self, unit_system):
        """ Load a unit system for conversion to and from it"""
        for u in list("JTKVASs") + ["eV", ]:
            v = self["%s to %s" % (u, unit_system)]
            setattr(self, u, v)
            self[u] = v
            for pre, factor in [('m', 1e-3), 
                                ('μ', 1e-6), 
                                ('n', 1e-9), 
                                ('p', 1e-12)]:
                
                v = self["%s to %s" % (u, unit_system)]
                uu = pre+u
                vv = v*factor
                setattr(self, uu, vv)
                self[uu] = vv
                
        for u in list("eħ") + ["me", "ke", "kB", "μB"]:
            v = self["%s [%s]" % (u, unit_system)]
            setattr(self, u, v)
            self[u] = v
            
units = UnitSystem("AU")