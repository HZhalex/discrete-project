from flask import Flask, render_template, request, jsonify
from collections import deque, defaultdict
import heapq
import json

app = Flask(__name__)

class Graph:
    def __init__(self):
        self.graph = defaultdict(list)
        self.vertices = set()
        self.visual_edges = [] 
    
    def add_vertex(self, vertex):
        """Thêm đỉnh đơn lẻ vào đồ thị"""
        self.vertices.add(vertex)
    
    def add_edge(self, u, v, weight=1, directed=False):
        self.graph[u].append((v, weight))
        self.vertices.add(u)
        self.vertices.add(v)
        if not directed:
            self.graph[v].append((u, weight))
        
        if directed:
            edge_id = f"{u}->{v}"
            existing = next((e for e in self.visual_edges if e['id'] == edge_id), None)
            if existing:
                existing['w'] = weight
            else:
                self.visual_edges.append({'u': u, 'v': v, 'w': weight, 'directed': True, 'id': edge_id})
        else:
            u_sorted, v_sorted = sorted([u, v])
            edge_id = f"{u_sorted}-{v_sorted}"
            existing = next((e for e in self.visual_edges if e['id'] == edge_id), None)
            if existing:
                existing['w'] = weight
            else:
                self.visual_edges.append({'u': u_sorted, 'v': v_sorted, 'w': weight, 'directed': False, 'id': edge_id})
    
    def visualize(self):
        nodes = []
        for vertex in sorted(list(self.vertices)):
            nodes.append({"id": vertex, "label": vertex})
        
        edges = []
        for e in self.visual_edges:
            edge_data = {
                "from": e['u'],
                "to": e['v'],
                "id": e['id']
            }
            
            # THAY ĐỔI: Chỉ hiển thị label nếu weight != 0
            if e['w'] != 0:
                edge_data["label"] = str(e['w'])
            else:
                edge_data["label"] = ""  # Không hiển thị label
            
            if e['directed']:
                edge_data["arrows"] = "to"
            else:
                edge_data["arrows"] = ""
            edges.append(edge_data)
        
        return {"nodes": nodes, "edges": edges}
    
    def to_adjacency_list(self):
        """Chuyển sang danh sách kề - format dễ đọc"""
        result = {}
        for vertex in sorted(list(self.vertices)):
            neighbors = []
            if vertex in self.graph:
                # Loại bỏ duplicate cho undirected graph
                seen = set()
                for v, w in self.graph[vertex]:
                    if v not in seen:
                        neighbors.append(v)  # CHỈ LẤY TÊN ĐỈNH, KHÔNG CÓ WEIGHT
                        seen.add(v)
            result[vertex] = neighbors if neighbors else []
        return result

        
    def to_adjacency_matrix(self):
        """Chuyển sang ma trận kề"""
        vertices_list = sorted(list(self.vertices))
        n = len(vertices_list)
        matrix = [[0] * n for _ in range(n)]
        vertex_index = {v: i for i, v in enumerate(vertices_list)}
        
        for u in self.graph:
            for v, w in self.graph[u]:
                i, j = vertex_index[u], vertex_index[v]
                matrix[i][j] = w
        
        return {
            "vertices": vertices_list,
            "matrix": matrix
        }
    
    def to_edge_list(self):
        """Chuyển sang danh sách cạnh - format dễ đọc"""
        edges = []
        for e in self.visual_edges:
            if e['directed']:
                edges.append(f"{e['u']}→{e['v']}")  # BỎ WEIGHT
            else:
                edges.append(f"{e['u']}-{e['v']}")  # BỎ WEIGHT
        return edges
    
    def export_json(self):
        """Xuất đồ thị ra JSON format chuẩn"""
        edge_list = []
        for e in self.visual_edges:
            edge_list.append({
                "from": e['u'],
                "to": e['v'],
                "weight": e['w'],
                "directed": e['directed']
            })
        
        return {
            "vertices": sorted(list(self.vertices)),
            "edges": edge_list
        }
    
    def import_json(self, data):
        """Nhập đồ thị từ JSON"""
        self.graph.clear()
        self.vertices.clear()
        self.visual_edges.clear()
        
        # Thêm các đỉnh
        if "vertices" in data:
            for v in data["vertices"]:
                self.vertices.add(v)
        
        # Thêm các cạnh
        if "edges" in data:
            for edge in data["edges"]:
                u = edge.get("from")
                v = edge.get("to")
                w = edge.get("weight", 1)
                directed = edge.get("directed", False)
                self.add_edge(u, v, w, directed)
    
    def save_graph(self):
        result = []
        for vertex in sorted(list(self.vertices)):
            neighbors = []
            if vertex in self.graph:
                for v, w in self.graph[vertex]:
                    neighbors.append(f"{v}(w:{w})")
            result.append(f"{vertex}: {', '.join(neighbors) if neighbors else 'không có kề'}")
        return result

    def bfs(self, start):
        if start not in self.vertices:
            return [], []
        
        visited = set()
        queue = deque([start])
        visited.add(start)
        order = [start]
        edges_used = []
        
        while queue:
            vertex = queue.popleft()
            if vertex in self.graph:
                for neighbor, _ in self.graph[vertex]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
                        order.append(neighbor)
                        edges_used.append((vertex, neighbor))
        
        return order, edges_used
    
    def dfs(self, start):
        if start not in self.vertices:
            return [], []
        
        visited = set()
        order = []
        edges_used = []
        
        def dfs_visit(vertex):
            visited.add(vertex)
            order.append(vertex)
            if vertex in self.graph:
                for neighbor, _ in self.graph[vertex]:
                    if neighbor not in visited:
                        edges_used.append((vertex, neighbor))
                        dfs_visit(neighbor)
        
        dfs_visit(start)
        return order, edges_used
    
    def has_negative_weight(self):
        """Kiểm tra xem đồ thị có cạnh trọng số âm không"""
        for u in self.graph:
            for v, weight in self.graph[u]:
                if weight < 0:
                    return True
        return False
    
    def dijkstra(self, start, end):
        if start not in self.vertices or end not in self.vertices:
            return None, float('inf'), []
        
        # Kiểm tra trọng số âm
        if self.has_negative_weight():
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
            
            if current_vertex in self.graph:
                for neighbor, weight in self.graph[current_vertex]:
                    distance = current_distance + weight
                    if distance < distances[neighbor]:
                        distances[neighbor] = distance
                        previous[neighbor] = current_vertex
                        heapq.heappush(pq, (distance, neighbor))
        
        path = []
        current = end
        if distances[end] == float('inf'):
             return None, float('inf'), []

        while current is not None:
            path.append(current)
            current = previous[current]
        path.reverse()
        
        if not path or path[0] != start:
            return None, float('inf'), []
        
        edges_used = [(path[i], path[i+1]) for i in range(len(path)-1)]
        return path, distances[end], edges_used
    
    def prim(self):
        if not self.vertices:
            return [], 0, []
        
        start = next(iter(self.vertices))
        visited = {start}
        edges = []
        total_weight = 0
        edges_used = []
        
        if start in self.graph:
            for neighbor, weight in self.graph[start]:
                heapq.heappush(edges, (weight, start, neighbor))
        
        while edges and len(visited) < len(self.vertices):
            weight, u, v = heapq.heappop(edges)
            if v not in visited:
                visited.add(v)
                total_weight += weight
                edges_used.append((u, v, weight))
                
                if v in self.graph:
                    for neighbor, w in self.graph[v]:
                        if neighbor not in visited:
                            heapq.heappush(edges, (w, v, neighbor))
        
        return edges_used, total_weight, edges_used
    
    def kruskal(self):
        edges = []
        seen_edges = set()
        for u in self.graph:
            for v, weight in self.graph[u]:
                edge_set = frozenset([u, v])
                if edge_set not in seen_edges:
                    edges.append((weight, u, v))
                    seen_edges.add(edge_set)
        
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
    
    def ford_fulkerson(self, source, sink):
        if source not in self.vertices or sink not in self.vertices:
            return 0, []
        
        residual = defaultdict(lambda: defaultdict(int))
        for u in self.graph:
            for v, capacity in self.graph[u]:
                residual[u][v] = capacity
        
        parent = {}
        max_flow = 0
        paths = []
        
        def bfs_find_path():
            visited = {source}
            queue = deque([source])
            parent.clear()
            parent[source] = None
            
            while queue:
                u = queue.popleft()
                for v in residual[u]:
                    if v not in visited and residual[u][v] > 0:
                        visited.add(v)
                        queue.append(v)
                        parent[v] = u
                        if v == sink:
                            return True
            return False
        
        while bfs_find_path():
            path_flow = float('inf')
            s = sink
            path = []
            while s != source:
                path.append((parent[s], s))
                path_flow = min(path_flow, residual[parent[s]][s])
                s = parent[s]
            path.reverse()
            paths.append({"path": path, "flow": path_flow})
            
            v = sink
            while v != source:
                u = parent[v]
                residual[u][v] -= path_flow
                residual[v][u] += path_flow
                v = parent[v]
            max_flow += path_flow
        
        return max_flow, paths
    
    def fleury(self, start):
        if start not in self.vertices:
            return [], "Đỉnh không tồn tại"
        
        temp_graph = defaultdict(list)
        for u in self.graph:
            temp_graph[u] = list(self.graph[u])
            
        odd_vertices = []
        for v in self.vertices:
            if len(temp_graph[v]) % 2 == 1:
                odd_vertices.append(v)
        
        if len(odd_vertices) > 2:
            return [], "Không có đường đi Euler (>2 đỉnh bậc lẻ)"
        if len(odd_vertices) == 2 and start not in odd_vertices:
            return [], f"Phải bắt đầu từ đỉnh bậc lẻ: {odd_vertices}"

        def is_bridge(u, v):
            if len(temp_graph[u]) == 1: return False
            
            def count_reachable(s):
                count = 0
                visited = set([s])
                q = deque([s])
                while q:
                    curr = q.popleft()
                    count += 1
                    for neighbor, _ in temp_graph[curr]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            q.append(neighbor)
                return count

            c1 = count_reachable(u)
            
            temp_graph[u].remove((v, next(w for n, w in temp_graph[u] if n == v)))
            temp_graph[v].remove((u, next(w for n, w in temp_graph[v] if n == u)))
            
            c2 = count_reachable(u)
            
            weight = 1
            temp_graph[u].append((v, weight))
            temp_graph[v].append((u, weight))
            
            return c1 > c2

        path = [start]
        curr = start
        edges_used = []
        
        limit = 1000
        while limit > 0:
            limit -= 1
            if not temp_graph[curr]:
                break
                
            next_v = None
            for v, w in temp_graph[curr]:
                if not is_bridge(curr, v):
                    next_v = v
                    break
            
            if next_v is None and temp_graph[curr]:
                next_v = temp_graph[curr][0][0]
                
            if next_v:
                edges_used.append((curr, next_v))
                
                for i, (n, w) in enumerate(temp_graph[curr]):
                    if n == next_v:
                        temp_graph[curr].pop(i)
                        break
                for i, (n, w) in enumerate(temp_graph[next_v]):
                    if n == curr:
                        temp_graph[next_v].pop(i)
                        break
                        
                path.append(next_v)
                curr = next_v
            else:
                break
                
        return path, edges_used

    def hierholzer(self, start):
        if start not in self.vertices:
            return [], "Đỉnh không tồn tại"
            
        temp_graph = defaultdict(list)
        for u in self.graph:
            temp_graph[u] = list(self.graph[u])
            
        stack = [start]
        path = []
        edges_used = []
        
        while stack:
            v = stack[-1]
            if temp_graph[v]:
                neighbor, weight = temp_graph[v][0]
                temp_graph[v].pop(0)
                for i, (n, w) in enumerate(temp_graph[neighbor]):
                    if n == v:
                        temp_graph[neighbor].pop(i)
                        break
                
                stack.append(neighbor)
                edges_used.append((v, neighbor))
            else:
                path.append(stack.pop())
        
        path.reverse()
        return path, edges_used

graph = Graph()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/graph', methods=['GET'])
def get_graph():
    return jsonify(graph.visualize())

@app.route('/api/add_vertex', methods=['POST'])
def add_vertex():
    """Thêm đỉnh đơn lẻ"""
    try:
        data = request.json
        vertex = data.get('vertex', '').strip()
        if not vertex:
            return jsonify({"success": False, "message": "Thiếu tên đỉnh"})
        
        graph.add_vertex(vertex)
        return jsonify({"success": True, "graph": graph.visualize()})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 400

@app.route('/api/add_edge', methods=['POST'])
def add_edge():
    try:
        data = request.json
        u = data.get('from', '').strip()
        v = data.get('to', '').strip()
        if not u or not v:
            return jsonify({"success": False, "message": "Thiếu tên đỉnh"})
        
        # THAY ĐỔI: weight mặc định = 0 thay vì 1
        weight_input = data.get('weight', '')
        weight = int(weight_input) if weight_input != '' else 0
        
        directed = data.get('directed', False)
        
        graph.add_edge(u, v, weight, directed)
        return jsonify({"success": True, "graph": graph.visualize()})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 400

@app.route('/api/save', methods=['GET'])
def save_graph():
    return jsonify({"adjacency_list": graph.save_graph()})

@app.route('/api/clear', methods=['POST'])
def clear_graph():
    global graph
    graph = Graph()
    return jsonify({"success": True})

@app.route('/api/export', methods=['GET'])
def export_graph():
    """Xuất đồ thị ra JSON"""
    return jsonify(graph.export_json())

@app.route('/api/import', methods=['POST'])
def import_graph():
    """Nhập đồ thị từ JSON"""
    try:
        data = request.json
        graph.import_json(data)
        return jsonify({"success": True, "graph": graph.visualize()})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 400

@app.route('/api/convert', methods=['POST'])
def convert():
    """Chuyển đổi giữa các biểu diễn"""
    convert_type = request.json.get('type', 'adjacency_list')
    
    if convert_type == 'adjacency_list':
        return jsonify({"result": graph.to_adjacency_list()})
    elif convert_type == 'adjacency_matrix':
        return jsonify({"result": graph.to_adjacency_matrix()})
    elif convert_type == 'edge_list':
        return jsonify({"result": graph.to_edge_list()})
    else:
        return jsonify({"success": False, "message": "Loại chuyển đổi không hợp lệ"})

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
    path, dist, edges = graph.dijkstra(data.get('start'), data.get('end'))
    return jsonify({"path": path, "distance": dist, "edges": edges})

@app.route('/api/prim', methods=['GET'])
def run_prim():
    # THÊM: Kiểm tra đồ thị có hướng
    has_directed = any(e['directed'] for e in graph.visual_edges)
    if has_directed:
        return jsonify({
            "success": False, 
            "message": "Không thể chạy Prim trên đồ thị có hướng"
        })
    
    edges, w, used = graph.prim()
    return jsonify({"edges": edges, "total_weight": w, "edges_used": used})

@app.route('/api/kruskal', methods=['GET'])
def run_kruskal():
    # THÊM: Kiểm tra đồ thị có hướng
    has_directed = any(e['directed'] for e in graph.visual_edges)
    if has_directed:
        return jsonify({
            "success": False, 
            "message": "Không thể chạy Kruskal trên đồ thị có hướng"
        })
    
    edges, w, used = graph.kruskal()
    return jsonify({"edges": edges, "total_weight": w, "edges_used": used})

@app.route('/api/ford_fulkerson', methods=['POST'])
def run_ff():
    data = request.json
    mf, paths = graph.ford_fulkerson(data.get('source'), data.get('sink'))
    return jsonify({"max_flow": mf, "paths": paths, "success": True})

@app.route('/api/fleury', methods=['POST'])
def run_fleury():
    data = request.json
    res = graph.fleury(data.get('start'))
    if isinstance(res[1], str):
         return jsonify({"success": False, "message": res[1]})
    return jsonify({"success": True, "path": res[0], "edges": res[1]})

@app.route('/api/hierholzer', methods=['POST'])
def run_hierholzer():
    data = request.json
    res = graph.hierholzer(data.get('start'))
    if isinstance(res[1], str):
         return jsonify({"success": False, "message": res[1]})
    return jsonify({"success": True, "path": res[0], "edges": res[1]})

@app.route('/api/check_bipartite', methods=['POST'])
def check_bipartite():
    vertex = request.json.get('vertex')
    if vertex not in graph.vertices:
        return jsonify({"is_bipartite": False, "message": "Đỉnh không tồn tại"})
    
    color = {v: -1 for v in graph.vertices}
    visited_global = set()
    
    for start_node in list(graph.vertices):
        if start_node in visited_global: continue
        
        q = deque([start_node])
        if color[start_node] == -1: color[start_node] = 0
        
        while q:
            u = q.popleft()
            visited_global.add(u)
            if u in graph.graph:
                for v, _ in graph.graph[u]:
                    if color[v] == -1:
                        color[v] = 1 - color[u]
                        q.append(v)
                    elif color[v] == color[u]:
                        return jsonify({"is_bipartite": False, "message": "Không phải đồ thị 2 phía"})
                        
    return jsonify({"is_bipartite": True, "message": "Là đồ thị 2 phía"})
@app.route('/api/delete_vertex', methods=['POST'])
def delete_vertex():
    """Xóa đỉnh và tất cả các cạnh liên quan"""
    try:
        data = request.json
        vertex = data.get('vertex', '').strip()
        
        if not vertex:
            return jsonify({"success": False, "message": "Thiếu tên đỉnh"})
        
        if vertex not in graph.vertices:
            return jsonify({"success": False, "message": "Đỉnh không tồn tại"})
        
        # Xóa đỉnh khỏi set
        graph.vertices.remove(vertex)
        
        # Xóa tất cả các cạnh liên quan đến đỉnh này
        if vertex in graph.graph:
            del graph.graph[vertex]
        
        # Xóa đỉnh này khỏi danh sách kề của các đỉnh khác
        for v in graph.graph:
            graph.graph[v] = [(neighbor, weight) for neighbor, weight in graph.graph[v] if neighbor != vertex]
        
        # Xóa các cạnh visual liên quan
        graph.visual_edges = [e for e in graph.visual_edges if e['u'] != vertex and e['v'] != vertex]
        
        return jsonify({"success": True, "graph": graph.visualize()})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 400

@app.route('/api/delete_edge', methods=['POST'])
def delete_edge():
    """Xóa cạnh"""
    try:
        data = request.json
        u = data.get('from', '').strip()
        v = data.get('to', '').strip()
        directed = data.get('directed', False)
        
        if not u or not v:
            return jsonify({"success": False, "message": "Thiếu tên đỉnh"})
        
        # Xóa cạnh khỏi graph
        if u in graph.graph:
            graph.graph[u] = [(neighbor, weight) for neighbor, weight in graph.graph[u] if neighbor != v]
        
        if not directed and v in graph.graph:
            graph.graph[v] = [(neighbor, weight) for neighbor, weight in graph.graph[v] if neighbor != u]
        
        # Xóa khỏi visual_edges
        if directed:
            edge_id = f"{u}->{v}"
            graph.visual_edges = [e for e in graph.visual_edges if e['id'] != edge_id]
        else:
            u_sorted, v_sorted = sorted([u, v])
            edge_id = f"{u_sorted}-{v_sorted}"
            graph.visual_edges = [e for e in graph.visual_edges if e['id'] != edge_id]
        
        return jsonify({"success": True, "graph": graph.visualize()})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 400
if __name__ == '__main__':
    app.run(debug=True, port=5000)