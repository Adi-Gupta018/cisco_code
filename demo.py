
# import networkx as nx
# import matplotlib.pyplot as plt
# from math import ceil

# def generate_topology(num_cities, users_per_city, bandwidth_matrix):
#     G = nx.Graph()
    
#     # Adding city nodes and Type 1 routers for users
#     for city in range(num_cities):
#         city_name = f"City{city+1}"
#         G.add_node(city_name)
        
#         # Add Type 1 routers for each city based on the number of users
#         num_type1_routers = (users_per_city[city] + 7) // 8  # Each Type 1 router supports 8 users
#         for router in range(num_type1_routers):
#             router_name = f"{city_name}_Type1_Router{router+1}"
#             G.add_node(router_name)
#             G.add_edge(city_name, router_name)
    
#     # Group cities into clusters for Type 2 routers
#     cities_per_router = 8  # Each Type 2 router can handle 8x 400G ports
#     num_clusters = ceil(num_cities / cities_per_router)
#     cluster_routers = []

#     for cluster in range(num_clusters):
#         cluster_router = f"Cluster_Type2_Router_{cluster+1}"
#         cluster_routers.append(cluster_router)
#         G.add_node(cluster_router)
    
#     # Connect cities to their cluster routers
#     for city in range(num_cities):
#         city_name = f"City{city+1}"
#         cluster_index = city // cities_per_router
#         cluster_router = cluster_routers[cluster_index]
#         G.add_edge(city_name, cluster_router)
    
#     # Ensure redundancy by connecting each city to an additional Type 2 router
#     for city in range(num_cities):
#         city_name = f"City{city+1}"
#         primary_cluster_index = city // cities_per_router
#         secondary_cluster_index = (primary_cluster_index + 1) % num_clusters
#         secondary_cluster_router = cluster_routers[secondary_cluster_index]
#         G.add_edge(city_name, secondary_cluster_router)
    
#     # Connect the cluster routers to each other for inter-cluster communication
#     for i in range(num_clusters):
#         for j in range(i+1, num_clusters):
#             G.add_edge(cluster_routers[i], cluster_routers[j])

#     return G

# def draw_topology(G, filename="topology.png"):
#     pos = nx.spring_layout(G)
#     plt.figure(figsize=(12, 8))
#     nx.draw(G, pos, with_labels=True, node_size=3000, node_color='lightblue', font_size=10, font_weight='bold')
#     plt.title("Network Topology")
#     plt.savefig(filename)
#     print(f"Topology saved as {filename}")

# if __name__ == "__main__":
#     num_cities = int(input("Enter the number of cities: "))
#     users_per_city = []
    
#     for i in range(num_cities):
#         users = int(input(f"Enter the number of users in City{i+1}: "))
#         users_per_city.append(users)
    
#     bandwidth_matrix = []
#     print("Enter the peak bandwidth requirements between each pair of cities (in Gbps):")
#     for i in range(num_cities):
#         row = []
#         for j in range(num_cities):
#             if i == j:
#                 row.append(0)
#             elif j < i:
#                 row.append(bandwidth_matrix[j][i])
#             else:
#                 bw = int(input(f"Bandwidth between City{i+1} and City{j+1}: "))
#                 row.append(bw)
#         bandwidth_matrix.append(row)
    
#     G = generate_topology(num_cities, users_per_city, bandwidth_matrix)
#     draw_topology(G)
import networkx as nx
import matplotlib.pyplot as plt
from heapq import heappop, heappush
from math import ceil

def heuristic(a, b):
    # Heuristic function for A* (Euclidean distance for simplicity)
    return abs(a - b)

def a_star_search(graph, start, goal, bandwidth):
    pq = [(0, start, [])]
    visited = set()
    min_bandwidth_path = {}

    while pq:
        (cost, node, path) = heappop(pq)
        if node in visited:
            continue
        visited.add(node)
        path = path + [node]

        if node == goal:
            return (cost, path)

        for next_node, data in graph[node].items():
            if next_node not in visited:
                next_cost = cost + data['weight']
                heappush(pq, (next_cost, next_node, path))

    return float("inf"), []

def generate_mesh_topology(num_cities, bandwidth_matrix):
    G = nx.Graph()
    
    for i in range(num_cities):
        for j in range(i + 1, num_cities):
            bandwidth = bandwidth_matrix[i][j]
            G.add_edge(f"City{i+1}", f"City{j+1}", weight=bandwidth)
    
    return G

def optimize_topology(G, num_cities, bandwidth_matrix):
    optimized_paths = {}
    for i in range(num_cities):
        for j in range(i + 1, num_cities):
            cost, path = a_star_search(G, f"City{i+1}", f"City{j+1}", bandwidth_matrix[i][j])
            optimized_paths[(f"City{i+1}", f"City{j+1}")] = path
    
    return optimized_paths

def allocate_routers(G, optimized_paths, num_cities):
    router_allocation = {}

    # Initialize the router allocation
    for city in range(num_cities):
        router_allocation[f"City{city+1}"] = {'Type1': 0, 'Type2': 0}

    for (start, end), path in optimized_paths.items():
        total_bandwidth = sum(G[path[i]][path[i + 1]]['weight'] for i in range(len(path) - 1))

        if total_bandwidth <= 800:
            router_allocation[start]['Type1'] += 1
            router_allocation[end]['Type1'] += 1
        else:
            router_allocation[start]['Type2'] += 1
            router_allocation[end]['Type2'] += 1

    return router_allocation

def remove_redundant_edges(G, optimized_paths):
    # Create a set of edges that are part of the optimized paths
    required_edges = set()
    for path in optimized_paths.values():
        for i in range(len(path) - 1):
            required_edges.add((path[i], path[i + 1]))

    # Remove edges not in the required edges set
    for edge in list(G.edges()):
        if edge not in required_edges and (edge[1], edge[0]) not in required_edges:
            G.remove_edge(*edge)

    return G

def draw_topology(G, filename="optimized_topology.png"):
    pos = nx.spring_layout(G)
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, with_labels=True, node_size=3000, node_color='lightblue', font_size=10, font_weight='bold')
    plt.title("Optimized Network Topology")
    plt.savefig(filename)
    print(f"Topology saved as {filename}")

if __name__ == "__main__":
    num_cities = int(input("Enter the number of cities: "))
    bandwidth_matrix = []

    print("Enter the peak bandwidth requirements between each pair of cities (in Gbps):")
    for i in range(num_cities):
        row = []
        for j in range(num_cities):
            if i == j:
                row.append(0)
            elif j < i:
                row.append(bandwidth_matrix[j][i])
            else:
                bw = int(input(f"Bandwidth between City{i+1} and City{j+1}: "))
                row.append(bw)
        bandwidth_matrix.append(row)
    
    G = generate_mesh_topology(num_cities, bandwidth_matrix)
    optimized_paths = optimize_topology(G, num_cities, bandwidth_matrix)
    router_allocation = allocate_routers(G, optimized_paths, num_cities)
    G = remove_redundant_edges(G, optimized_paths)
    draw_topology(G)

    print("Router Allocation:")
    for city, allocation in router_allocation.items():
        print(f"{city}: Type1 Routers = {allocation['Type1']}, Type2 Routers = {allocation['Type2']}")
