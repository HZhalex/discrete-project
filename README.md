# Graph Visualizer & Algorithms Web Application

Một ứng dụng web cho phép **vẽ đồ thị trực quan**, **lưu dữ liệu đồ thị**, và chạy các **thuật toán cơ bản** như BFS, DFS, tìm đường đi ngắn nhất, kiểm tra đồ thị hai phía, và chuyển đổi giữa các dạng biểu diễn đồ thị.

---

## 🚀 Tính năng chính

### 1. Vẽ đồ thị trực quan
- Thêm / xoá đỉnh
- Thêm / xoá cạnh
- Chọn đồ thị vô hướng / có hướng
- Hiển thị theo thời gian thực bằng thư viện JS (Cytoscape.js hoặc Vis.js)

### 2. Lưu & tải đồ thị
- Lưu dữ liệu đồ thị vào file JSON
- Tải lại đồ thị đã lưu
- Quản lý nhiều đồ thị

### 3. Thuật toán đồ thị
- **BFS (Breadth-First Search)**
- **DFS (Depth-First Search)**
- **Dijkstra – đường đi ngắn nhất**
- **Kiểm tra đồ thị 2 phía (Bipartite Graph)**

### 4. Chuyển đổi biểu diễn đồ thị
- Ma trận kề ↔ Danh sách kề ↔ Danh sách cạnh  
- Hỗ trợ cả đồ thị **vô hướng** và **có hướng**

---

## 🏗 Kiến trúc hệ thống

Project được chia làm hai phần:

### **Frontend (HTML + CSS + JavaScript)**
- Giao diện vẽ đồ thị
- Gửi yêu cầu thuật toán đến backend
- Nhận kết quả và hiển thị

### **Backend (Python – Flask hoặc FastAPI)**
- Xử lý thuật toán đồ thị
- Chuyển đổi dạng biểu diễn
- Lưu & nạp file JSON
- Trả kết quả theo dạng API

---

## 📁 Cấu trúc thư mục

```plaintext
graph-visualizer/
│
├── backend/
│   ├── app.py                 # Main backend API
│   ├── algorithms/
│   │   ├── bfs.py
│   │   ├── dfs.py
│   │   ├── dijkstra.py
│   │   └── bipartite.py
│   ├── utils/
│   │   └── converter.py       # Chuyển đổi biểu diễn
│   ├── models/
│   │   └── graph_model.py
│   └── data/
│       └── graphs.json
│
├── frontend/
│   ├── index.html
│   ├── style.css
│   ├── app.js
│   └── libs/                  # cytoscape.js hoặc vis.js
│
├── .gitignore
├── requirements.txt
└── README.md
