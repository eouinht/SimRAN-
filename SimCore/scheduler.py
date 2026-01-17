def allocate_prb(serving, prb_ratio, maxPrb, n_cells):
    # Chia PRB theo cell -> phân cho UE trong cell.
    prb_cell = prb_ratio * maxPrb
    allocated = {}
    
    for c in range(n_cells):
        ues = []
        for i in range(len(serving)):   # duyệt từng UE
            if serving[i] == c:         # nếu UE u đang gắn với cell c
                ues.append(i)           # thêm UE đó vào danh sách của cell c
                
        for u in ues:
            allocated[u] = prb_cell[c]/max(len(ues), 1)  # Cell c chia đều toàn bộ tài nguyên của mình cho các UE đang gắn vào nó.
    return allocated
    