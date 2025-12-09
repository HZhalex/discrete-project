from flask import Flask, render_template, request, jsonify
from collections import deque, defaultdict
import heapq

app = Flask(__name__)

class Graph:
    def __init__(self):
        self.graph = defaultdict(list)
        self.vertices = set()
    
    def add_edge(self, u, v, weight=1, directed=False):
        self.graph[u].append((v, weight))
        self.vertices.add(u)
        self.vertices.add(v)
        if not directed:
            self.graph[v].append((u, weight))
    
    def visualize(self):
        """Trả về dữ liệu đồ thị để hiển thị"""
        nodes = []
        edges = []
        seen_edges = set()
        
        for vertex in self.vertices:
            nodes.append({"id": vertex, "label": vertex})
        
        for u in self.graph:
            for v, weight in self.graph[u]:
                edge_key = tuple(sorted([u, v]))
                if edge_key not in seen_edges:
                    edges.append({"from": u, "to": v, "label": str(weight)})
                    seen_edges.add(edge_key)
        
        return {"nodes": nodes, "edges": edges}
    
    def save_graph(self):
        """Lưu đồ thị dưới dạng danh sách kề"""
        result = []
        for vertex in sorted(self.vertices):
            neighbors = [f"{v}(w:{w})" for v, w in self.graph[vertex]]
            result.append(f"{vertex}: {', '.join(neighbors) if neighbors else 'không có kề'}")
        return result
    
    def bfs(self, start):
        """Duyệt đồ thị theo chiều rộng"""
        if start not in self.vertices:
            return [], []
        
        visited = set()
        queue = deque([start])
        visited.add(start)
        order = [start]
        edges_used = []
        
        while queue:
            vertex = queue.popleft()
            for neighbor, _ in self.graph[vertex]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
                    order.append(neighbor)
                    edges_used.append((vertex, neighbor))
        
        return order, edges_used
    
    def dfs(self, start):
        """Duyệt đồ thị theo chiều sâu"""
        if start not in self.vertices:
            return [], []
        
        visited = set()
        order = []
        edges_used = []
        
        def dfs_visit(vertex):
            visited.add(vertex)
            order.append(vertex)
            for neighbor, _ in self.graph[vertex]:
                if neighbor not in visited:
                    edges_used.append((vertex, neighbor))
                    dfs_visit(neighbor)
        
        dfs_visit(start)
        return order, edges_used
    
    def dijkstra(self, start, end):
        """Tìm đường đi ngắn nhất bằng Dijkstra"""
        if start not in self.vertices or end not in self.vertices:
            return None, float('inf'), []
        
        distances = {vertex: float('inf') for vertex in self.vertices}
        distances[start] = 0
        previous = {vertex: None for vertex in self.vertices}
        pq = [(0, start)]
        visited = set()
        
        while pq:
            current_distance, current_vertex = heapq.heappop(pq)
            
            if current_vertex in visited:
                continue
            
            visited.add(current_vertex)
            
            if current_vertex == end:
                break
            
            for neighbor, weight in self.graph[current_vertex]:
                distance = current_distance + weight
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    previous[neighbor] = current_vertex
                    heapq.heappush(pq, (distance, neighbor))
        
        path = []
        current = end
        while current is not None:
            path.append(current)
            current = previous[current]
        path.reverse()
        
        if path[0] != start:
            return None, float('inf'), []
        
        edges_used = [(path[i], path[i+1]) for i in range(len(path)-1)]
        return path, distances[end], edges_used
    
    def prim(self):
        """Tìm cây khung nhỏ nhất bằng Prim"""
        if not self.vertices:
            return [], 0, []
        
        start = next(iter(self.vertices))
        visited = {start}
        edges = []
        total_weight = 0
        edges_used = []
        
        for neighbor, weight in self.graph[start]:
            heapq.heappush(edges, (weight, start, neighbor))
        
        while edges and len(visited) < len(self.vertices):
            weight, u, v = heapq.heappop(edges)
            if v not in visited:
                visited.add(v)
                total_weight += weight
                edges_used.append((u, v, weight))
                
                for neighbor, w in self.graph[v]:
                    if neighbor not in visited:
                        heapq.heappush(edges, (w, v, neighbor))
        
        return edges_used, total_weight, edges_used
    
    def kruskal(self):
        """Tìm cây khung nhỏ nhất bằng Kruskal"""
        edges = []
        for u in self.graph:
            for v, weight in self.graph[u]:
                if u < v:
                    edges.append((weight, u, v))
        
        edges.sort()
        parent = {v: v for v in self.vertices}
        rank = {v: 0 for v in self.vertices}
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px == py:
                return False
            if rank[px] < rank[py]:
                parent[px] = py
            elif rank[px] > rank[py]:
                parent[py] = px
            else:
                parent[py] = px
                rank[px] += 1
            return True
        
        mst_edges = []
        total_weight = 0
        
        for weight, u, v in edges:
            if union(u, v):
                mst_edges.append((u, v, weight))
                total_weight += weight
        
        return mst_edges, total_weight, mst_edges

# Khởi tạo đồ thị mẫu
graph = Graph()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/graph', methods=['GET'])
def get_graph():
    return jsonify(graph.visualize())

@app.route('/api/add_edge', methods=['POST'])
def add_edge():
    data = request.json
    u = data.get('from')
    v = data.get('to')
    weight = int(data.get('weight', 1))
    directed = data.get('directed', False)
    
    graph.add_edge(u, v, weight, directed)
    return jsonify({"success": True, "graph": graph.visualize()})

@app.route('/api/save', methods=['GET'])
def save_graph():
    result = graph.save_graph()
    return jsonify({"adjacency_list": result})

@app.route('/api/bfs', methods=['POST'])
def run_bfs():
    data = request.json
    start = data.get('start')
    order, edges = graph.bfs(start)
    return jsonify({"order": order, "edges": edges})

@app.route('/api/dfs', methods=['POST'])
def run_dfs():
    data = request.json
    start = data.get('start')
    order, edges = graph.dfs(start)
    return jsonify({"order": order, "edges": edges})

@app.route('/api/dijkstra', methods=['POST'])
def run_dijkstra():
    data = request.json
    start = data.get('start')
    end = data.get('end')
    path, distance, edges = graph.dijkstra(start, end)
    return jsonify({"path": path, "distance": distance, "edges": edges})

@app.route('/api/prim', methods=['GET'])
def run_prim():
    edges, weight, edges_used = graph.prim()
    return jsonify({"edges": edges, "total_weight": weight, "edges_used": edges_used})

@app.route('/api/kruskal', methods=['GET'])
def run_kruskal():
    edges, weight, edges_used = graph.kruskal()
    return jsonify({"edges": edges, "total_weight": weight, "edges_used": edges_used})

@app.route('/api/clear', methods=['POST'])
def clear_graph():
    global graph
    graph = Graph()
    return jsonify({"success": True})

@app.route('/api/check_bipartite', methods=['POST'])
def check_bipartite():
    data = request.json
    vertex = data.get('vertex')
    
    if vertex not in graph.vertices:
        return jsonify({"is_bipartite": False, "message": "Đỉnh không tồn tại"})
    
    color = {v: -1 for v in graph.vertices}
    queue = deque([vertex])
    color[vertex] = 0
    
    while queue:
        u = queue.popleft()
        for v, _ in graph.graph[u]:
            if color[v] == -1:
                color[v] = 1 - color[u]
                queue.append(v)
            elif color[v] == color[u]:
                return jsonify({"is_bipartite": False, "message": "Đồ thị không phải 2 phía"})
    
    return jsonify({"is_bipartite": True, "message": "Đồ thị là 2 phía"})

@app.route('/api/convert', methods=['POST'])
def convert_representation():
    data = request.json
    from_type = data.get('from')
    to_type = data.get('to')
    
    result = []
    
    if from_type == 'adjacency' and to_type == 'edge':
        for u in sorted(graph.vertices):
            for v, w in graph.graph[u]:
                if u <= v:
                    result.append(f"{u} -- {v} (trọng số: {w})")
    elif from_type == 'adjacency' and to_type == 'incidence':
        result.append("Ma trận liên thuộc đỉnh-cạnh:")
        edges_list = []
        for u in graph.graph:
            for v, w in graph.graph[u]:
                if u <= v:
                    edges_list.append((u, v, w))
        
        vertices_sorted = sorted(graph.vertices)
        header = "    " + "  ".join([f"e{i}" for i in range(len(edges_list))])
        result.append(header)
        
        for vertex in vertices_sorted:
            row = [vertex]
            for u, v, _ in edges_list:
                if u == vertex or v == vertex:
                    row.append("1")
                else:
                    row.append("0")
            result.append("  ".join(str(x) for x in row))
    
    return jsonify({"result": result})

if __name__ == '__main__':
    app.run(debug=True, port=5000)