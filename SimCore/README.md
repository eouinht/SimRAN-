# Simcore logic
## 1. Topology
* N cell | mặc định n_cells = 10
* Hai cell các nhau (Inter Site Distance) = 200m
    - Urban micro ISD: 200m
    - Macro ISD: 500 -> 1000m
* Tọa độ UE trong Topology: Xác định bằng 1 trục Ox
    - x = vị trí dọc theo 1 tuyến
    - UE di chuyển qua các cell liên tiếp
    - Mỗi cell được tính là 200m trên trục
        VD: cell 0: x [0, 100]
            cell 1: x [100, 200]
            ...

    - Lý do không dùng tọa độ (x,y) hay (x, y, z):
        SINR phụ thộc vào khoảng cách Euclidean, góc phức tạp

        Xử dụng 1D giúp giảm state dimension

        DQN học nhanh hơn

        Dễ dàng scale số lượng cell
    
## 2.UE - Mobility + traffic source 
* UE di chuyển với tốc độc: speed (m/s)
* UE có nhu cầu sử dụng tài nguyên trong lúc di chuyển (Mbps)
* Mobility -> HandOver trigger
* Demand -> load stress lên scheduler
## 3.Channel - Kênh truyền sóng (đơn giản)
* RSRP(Reference Signal Received Power) - Công suất tín hiệu tham chiếu nhận được từ 1 cell cụ thể chưa xét đến nhiễu: 
    - RSRP giảm theo khoảng cách
    - Đây không phải công thức chuẩn 3GPP, chỉ là công thức suy hao đơn giản tỷ lệ theo $log_{10}$
    - Dùng để chọn cell phục vụ cho UE, quyết định HO 

* Noise 
* Interference từ cell khác
* SINR(Signal to Interference plus Noise Ratio): Chỉ số đo lường chất lượng tín hiệu tính đến nhiễu.
    $$ SINR = {Psig \over [P(interf) + P(Noise)]} (dB) $$
    - Dùng để tính thông lượng, đánh giá QoS 

## 4. Association - Serving cell & Handover
* Agent có quyền quyết định HO hay không
* HO có cose

## 5. Scheduler - Resource allocation
* TxPower: Công suất phát của gNodeB (dBm) 
* PRB(Physical Resource Block) - Khối tài nguyên vật lý trong 5G NR, đây là đơn vị cơ bản nhỏ nhất để phân bổ tài nguyên vô tuyến (băng thông/ thời gian).
* Prb_allocated: số PRB cấp cho 1 UE
* Prb_total: tổng số PRB của tất cả UE trong 1 cell
* PRB tăng 
* PRB giảm

## 6. KPI - Chỉ tiêu cho RL
* Throughput theo SINR + PRB
* xác suất drop:
    * UE bị drop khi:
        - SINR/RSRP thấp quá thấp trong một khoảng thời gian.
        - HARQ liên tục thất bại
        - HO thất bại
        - Resource starvation 
        ...
    * Trong mô phỏng này ta chỉ sử dụng xác suất biểu thị hành vi drop chứ không mô phỏng các hiện tượng này.
    $$
    p_{\text{drop}} = \sigma(-k \cdot \text{SINR}_{\text{eff}})
    $$

    $$
    \text{drop} =
    \begin{cases}
    1, & \text{if } u < p_{\text{drop}}, \quad u \sim \mathcal{U}(0,1) \\
    0, & \text{otherwise}
    \end{cases}
    $$
    * Trong môi trường, $${drop = 1}$$ -> penalty. Đây là hành vi tệ.

