import numpy as np
from config import SimConfig

class Channel:
    def __init__(self):
        # Frequency
        self.freq = SimConfig.FREQ_GHZ

        # Tx power
        self.tx_macro = SimConfig.TX_POWER_MACRO
        self.tx_small = SimConfig.TX_POWER_SMALL

        # Antenna gain
        self.gain_macro = SimConfig.MACRO_GAIN
        self.gain_small = SimConfig.SMALL_GAIN

        # Pathloss model
        self.pl_const_macro = SimConfig.PL_CONST_MACRO
        self.pl_slope_macro = SimConfig.PL_SLOPE_MACRO
        self.pl_const_small = SimConfig.PL_CONST_SMALL
        self.pl_slope_small = SimConfig.PL_SLOPE_SMALL

        # Noise
        self.noise_dbm = SimConfig.NOISE_DBM
        self.noise_mw = self.dbm_to_mw(self.noise_dbm)
    
    def dbm_to_mw (self, dbm):
        return 10**(dbm/10)
    
    def mw_to_dbm(self, mw):
        return 10*np.log10(mw + 1e-12)
    
    
    def pathloss(self, d, is_small):
        d = max(d, 1.0)
        if is_small:
            c = self.pl_const_small
            s = self.pl_slope_small
        else:
            c = self.pl_const_macro
            s = self.pl_slope_macro
        return c + s*np.log10(d) + 20*np.log10(self.freq)
    
    def rsrp(self, d, is_small=False):
        """Return dbm

        Args:
            d (_type_): _description_
            is_small (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        if is_small:
            tx = self.tx_small
            gain = self.gain_small
        else:
            tx = self.tx_macro
            gain = self.gain_macro

        pl = self.pathloss(d, is_small)
        
        return (tx + gain) - pl
    
    def sinr(self, sig_dbm, interf_dbm):
        """retrun dB

        Args:
            sig_dbm (_type_): _description_
            interf_dbm (_type_): _description_

        Returns:
            _type_: _description_
        """
        sig_mw = self.dbm_to_mw(sig_dbm)
        
        interf_mw = 0
        for x in interf_dbm:
            interf_mw += self.dbm_to_mw(x)
        sinr = sig_mw/(interf_mw + self.noise_mw)    
        return 10 * np.log10(sinr + 1e-12)
    
    def rsrq(self, sig_dbm, interf_dbm):
        sig_mw = self.dbm_to_mw(sig_dbm)
        interf_mw = 0.0
        for x in interf_dbm:
            interf_mw += self.dbm_to_mw(x)
        
        rssi = sig_mw + interf_mw + self.noise_mw
        rsrq = sig_mw/rssi
        return 10* np.log10(rsrq + 1e-12) + 20
    
    def compute_link(self, serving_cell, ue, cells):
        d_serv = np.linalg.norm(serving_cell.pos - ue.pos)
        
        is_small = (serving_cell.id != 0)
        rsrp_serv = self.rsrp(d_serv, is_small)
        
        interf_dbm = []
        for cell in cells:
            if cell.id == serving_cell.id:
                continue

            d = np.linalg.norm(cell.pos - ue.pos)
            is_small_i = (cell.id != 0)

            p = self.rsrp(d, is_small_i)
            interf_dbm.append(p)

        sinr = self.sinr(rsrp_serv, interf_dbm)
        rsrq = self.rsrq(rsrp_serv, interf_dbm)

        return rsrp_serv, sinr, rsrq