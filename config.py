class SimConfig:

    # =========================
    # TOPOLOGY
    # =========================
    N_CELLS = 6          # 1 macro + 5 small
    N_UES   = 5
    AREA   = 2000        # meters

    MAX_STEPS = 1000
    # =========================
    # RADIO
    # =========================
    FREQ_GHZ = 3.5

    TX_POWER_MACRO = 46.0   # dBm
    TX_POWER_SMALL = 30.0   # dBm

    INIT_TX_POWER  = 30.0 
    
    POWER_STEP = 1.0        # dBm
 
    MACRO_GAIN = 18.0
    SMALL_GAIN = 5.0

    NOISE_DBM = -104.0

    # =========================
    # PATHLOSS
    # =========================
    PL_CONST_MACRO = 28.0
    PL_SLOPE_MACRO = 22.0

    PL_CONST_SMALL = 32.4
    PL_SLOPE_SMALL = 21.0

    # =========================
    # CELL LOAD
    # =========================
    CELL_MAX_PRB = 100
    
    LOAD_PER_UE_MACRO = 5.0
    LOAD_PER_UE_SMALL = 15.0

    LOAD_NOISE_MIN = -2.0
    LOAD_NOISE_MAX =  2.0

    # =========================
    # MOBILITY
    # =========================
    UE_SPEED_MIN = 20.0 # m/s
    UE_SPEED_MAX = 30.0 # m/s

    # 
    TRAFFIC_LAMBDA = 5    # Poisson mean
    SINR_THRESHOLD = -5.0
     
    OUTAGE_PENALTY = 5.0
    PRB_PENALTY    = 2.0