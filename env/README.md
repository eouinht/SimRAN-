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
Agent cần trả lời được: Nếu một UE có trạng thái như thế này thì nên hành xử với UE đó như thế nào.