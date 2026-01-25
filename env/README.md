# Custom Gym Environment For Cellular Network Core
Custom Env này được thiết kế để train RL agent trong cellular network control.

* Simcore là phần logic, mô phỏng một cách đơn giản các thực thể trong mạng vô tuyến.
* Gym Env: Tập trung định nghĩa state, action, reward cho bài toán theo UE-centric để agent giải đưa ra quyết định nên Handover hay không, phân bổ tài nguyên như nào.

* Mục tiêu thiết kế: 
    - Thể hiện được mối liên hệ giữa hành động của agent và hiệu suất của mạng vô tuyến dưới chuỗi hành vi Markov Deciion Process (MDP).
    - Cho phép agent học chính sách điều khiển tài nguyên cho UE.
    - Tối ưu Qos(throughput, drop) và hiệu quả tài nguyên(PRB, load).
    - Hỗ trợ scale số lượng cell, ue và thêm các thông số mạng phức tạp hơn. 
    
## 1 Các thực thể chính:
* Cell(gNodeB abstraction): là vùng phủ sóng vô tuyến được tạo ra bởi 1 trạm phát sóng (base station), nơi các UE kể nối vào mạng. 5G sử dụng mật độ dày đặc các small cells bên cạnh cách macro cell truyền thống để cung cấp tốc độ vao, độ trễ thấp và tăng cường dung lượng tại các khu vực đông đúc.
* User Equipment (UE): là các thiết bị đầu cuối như smartphone, máy tính và các thiết bị IoT.

## 2 One-Dimensional Geometry
Môi trường sử dụng mô hình 1D đẻ đơn giản hóa việc tính toán vị trí các thực thể, dễ tính nhiễu, giảm chiều state:

* Vị trí Cell: $$ \text{cell\_position} \in [x_{\min}, x_{\max}]  $$     
* Vị trí UE: $$ \text{ue\_position} \in [x_{\min}, x_{\max}] $$

## 3. Custom-Env
### 3.1 Action Space
Env sử dụng Discrete Ation space với 4 hành động:
$$ \text{A = \{0, 1, 2, 3\}}
$$ 

* Action 0 - Không hành động gì: Thể hiện trạng thái ổn định của hệ thống, tránh HO, điều chỉnh PRB một cách không cần thiết gây lãng phí tài nguyên hay nguy cơ gây thiếu tài nguyên (resource starvation).

* Action 1 - Tăng PRB 1 đơn vị:   
Định luật Shannon - Hartley: 
$$\text{C} = {B} \cdot \log_2(1 + \text{SINR})
$$
Trong đó: C là channel capacity, B là băng thông của kênh liên quan trực tiếp PRB việc phân chia tài nguyên cho UE trong bài toán.

-> Khi PRB tăng, UE có: 
+ nhiều tài nguyên thời gian - tần số hơn.
+ Lưu lượng truyền lớn hơn.
+ cải thiện SINR_eff
$$ \text{SINR{eff}} = SINR + \alpha \cdot PRB{allocated}
$$
+ Việc tăng PRB không có nghĩa là tăng công suất vật lý mà là đang lấy tài nguyên của các UE khác cho UE này -> gây tăng load, tổng tài nguyên dành cho 1 cell là không đổi.
* Action 2 - Giảm PRB 1 đơn vị: Giải phóng tài nguyên, giảm load cho cell.
Việc  giảmt tài nguyên tuy có thể tiết kiệm tài nguyên nhưng có thể gây tăng tỉ lệ drop.
* Action 3 - Handover: UE được chuyển sang cell lân cận, giảm áp lực lên cell hiện trước đó, cải thiện SINR.
- Tính đến chi phí Handover.
- Việc handover không hợp lí có thể dẫn đến handover fail.

### 3.2 State Space
* State vector được định nghĩa:
$$
S_t = \begin{bmatrix} 
\text{SINR}_{t} \\ 
\text{Position}_{t} \\ 
\text{Serving Cell}_{t} \\ 
\text{PRB Allocated}_{t} 
\end{bmatrix}
$$

Đây là cách agent nhìn UE, UE không được quyền can thiệp vào tài nguyên hay quyết định mình được Handover mà nó đại diện cho 1 flow, RL dùng UE làm sample của policy chung, cái mà agent cần cải thiện. 
Agent cần trả lời được:

    "Nếu một UE có trạng thái như thế này thì nên hành xử với UE đó như thế nào".
- SINR: Đánh giá chất lượng liên kết, đây là thông số quan trọng nhất quyết định throughput và drop. Chi tiết được mô tả trong [Xem tại](/SimCore/README.md)
- UE position: Phản ánh tinh mobility, giúp agent có thể biết UE đang đến gần biên cell, có nguy cơ cần Handover. UE position không thể hiện tọa độ của UE trong toàn mạng mà chỉ thể hiện tọa độ tương đối của UE trong cell đang chứa nó.
- Serving Cell: Cho biết UE đang kết nối với cell nào. Giúp phân biệt RSRP serving với neighbor và quyết định Handover.
- PRB Allocated: Phản ánh mức tài nguyên đang được cấp cho UE. Cho phép agent học khi nào cần tăng/giảm PRB.
### 3.3 Reward Funtion: 
$$ {R} = {Throughput} - \alpha \cdot Load - \beta \cdot Drop - \gamma \cdot HO
$$
* Throughput (Positive reward): Đại diện cho tốc độ dữ liệu hữu ích UE nhận được.
- Phụ thuộc vào:
    - SINR
    - PRB được cấp
- Đây là động lực chính để agent cải thiện QoS
* Load Penalty: Phản ánh mức độ chiếm dụng tài nguyên cell
$$ Load = {PRB\_{allocated} \over PRB\_{total}}
$$ 
- Nếu cấp quá nhiều tài nguyên cho UE này thì sẽ gây đói tài nguyên tại các điểm khác.
- Load cao gây tắc nghẽn, delay, drop các UE khác.
* Drop penalty: Đại diện cho Radio Link Failure (RLF), Qó violation nghiêm trọng.
$$ Drop \in \{0, 1\}
$$ 
Đây là sự kiện ít xảy ra nhưng rất xấu nên cần phạt mạnh. Drop xảy ra khi:
- SINR quá thấp
- PRB không đủ
- HO quá trễ, thất bại 
* Handover penalty: Thực hiện HO gây ra trễ, ping-pong HO gây bất ổn định cho cell. Tuy nhiên khi HO hợp lí có thể giúp cải thiện SINR lâu dài cho UE.
$$ HO \in \{0, 1\}
$$

Reward phản ánh trải nghiệm của một UE, không phải toàn mạng nhưng Agent phải học hành vi để có lợi cho mạng gián tiếp.

| Nếu làm | Hậu quả |
| :--- | :--- |
| **Max PRB** | Load penalty |
| **Không HO** | Drop penalty |
| **HO liên tục** | HO penalty |
| **Không cấp PRB** | Throughput thấp |

* Có thể thêm delay penalty, energy cost vào reward. 