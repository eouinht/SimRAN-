import numpy as np
from .channel import Channel
from .ue import UE
from .cell import Cell
from .network import Network

class SimCore:

    def __init__(self, config):
        self.config = config
        self.channel = Channel(config)
        self.network = Network(config)

        # Khoi Tao Cell
        self.cells = []
        r = config.AREA_SIZE / 3
        c = config.AREA_SIZE / 2
        
        for i in range(config.N_CELLS):
            angle = 2*np.pi*i / config.N_CELLS
            pos = np.array([
                c + r*np.cos(angle),
                c + r*np.sin(angle)
            ])
            cell = Cell(i, pos, config.INIT_TX_POWER, config.MAX_PRB)
            self.cells.append(cell)

        # Khoi tao UE
        self.ues = []
        for i in range(config.N_UES):
            pos = np.random.uniform(0, config.AREA_SIZE, 2)
            self.ues.append(
                UE(
                    i, 
                    pos,
                    config.UE_SPEED_MIN,
                    config.UE_SPEED_MAX,
                    config.AREA_SIZE
                )
            )
            
        self.net_metrics = {} 
        self.time = 0   
         
    def step(self, action):
        self.time += 1
        
        # ---- Move UE ----
        for ue in self.ues:
            ue.move()

        # ---- Apply power control ----
        for i, cell in enumerate(self.cells):
            cell.tx_power += action[i] * self.config.POWER_STEP
            cell.tx_power = np.clip(
                cell.tx_power,
                self.config.TX_POWER_SMALL,
                self.config.TX_POWER_MACRO
            )

       
        for cell in self.cells:
            cell.reset()

        # ---- Association ----
        for ue in self.ues:

            best_sinr = -1e9
            best_cell = None

            for cell in self.cells:

                rsrp, sinr, rsrq = self.channel.compute_link(
                    cell, ue, self.cells
                )

                if sinr > best_sinr:
                    best_sinr = sinr
                    best_cell = cell
                    ue.sinr = sinr
                    ue.rsrp = rsrp
                    ue.rsrq = rsrq

            ue.serving_cell = best_cell.id
            best_cell.connected_ues.append(ue.id)
        
        # ---- Load + throughput ----
        for cell in self.cells:

            if len(cell.connected_ues) == 0:
                continue

            prb_per_ue = cell.max_prb / len(cell.connected_ues)

            sinrs, rsrps, rsrqs = [], [], []

            for ue_id in cell.connected_ues:
                ue = self.ues[ue_id]

                demand = np.random.poisson(self.config.TRAFFIC_LAMBDA)
                ue.traffic = demand

                ue.throughput = self.compute_throughput(
                    ue.sinr, prb_per_ue
                )

                cell.prb_usage += prb_per_ue
                cell.total_traffic += demand

                sinrs.append(ue.sinr)
                rsrps.append(ue.rsrp)
                rsrqs.append(ue.rsrq)

            cell.prb_usage = min(cell.prb_usage, cell.max_prb)
            cell.load = cell.prb_usage / cell.max_prb

            cell.avg_sinr = float(np.mean(sinrs))
            cell.avg_rsrp = float(np.mean(rsrps))
            cell.avg_rsrq = float(np.mean(rsrqs))
         
        # ---- Network metrics ----
        self.net_metrics = self.network.compute(self.ues, self.cells)
              
        state = self.get_state()
        reward = self.compute_reward()
        done = False
        info = {}
        
        return state, reward, done, info


    def compute_throughput(self, sinr, prb):
        spectral_eff = np.log2(1 + 10**(sinr/10))
        return prb * self.config.PRBBW * spectral_eff

    def get_state(self):

        state = []

        # Sim 
        state.extend([
            self.time/self.config.MAX_TIME,
            self.config.N_CELLS,
            self.config.N_UES
        ])
        # Network-level
        state.extend([
            self.net_metrics["totalTraffic"],
            self.net_metrics["connectedUEs"] / self.config.N_UES,
            self.net_metrics["maxPrbUsage"],
            self.net_metrics["totalTxPower"]
        ])

        # ---- Cell-level ----
        for cell in self.cells:
            state.extend([
                (cell.tx_power - self.config.TX_POWER_SMALL) /
                (self.config.TX_POWER_MACRO - self.config.TX_POWER_SMALL),

                cell.prb_usage / cell.max_prb,
                len(cell.connected_ues) / self.config.N_UES,
                cell.load,
                cell.avg_rsrp,
                cell.avg_rsrq,
                cell.avg_sinr,
                cell.total_traffic
            ])

        return np.array(state, dtype=np.float32)


    def compute_reward(self):

        avg_tp = np.mean([u.throughput for u in self.ues])
        outage = np.sum(
            [1 for u in self.ues if u.sinr < self.config.SINR_THRESHOLD]
        )

        prb_violation = sum(
            1 for c in self.cells
            if c.prb_usage > c.max_prb
        )

        reward = (
            avg_tp
            - outage * self.config.OUTAGE_PENALTY
            - prb_violation * self.config.PRB_PENALTY
        )

        return reward