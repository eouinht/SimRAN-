class SimConfig:
    N_CELLS = 5
    N_UES = 20
    AREA = 1000
    
    # PHY
    NOISE = 1e-9
    PATHLOSS = 3.5
    
    # Power
    MIN_TX = 10.0
    MAX_TX = 50
    
    # MAC
    MAX_PRB = 100
    
    # Traffic
    LAMDA = 2.0  #Poison distributed 
    
    # Queue / delay
    DELAY_MAX = 10.0
    
    # Mobility
    MAX_SPEED = 10.0
    
    #HO
    HO_PEN = 5.0    # Packet loss khi HO
    HO_TIME = 2.0   # UE bị “mất sóng” trong 2 timestep