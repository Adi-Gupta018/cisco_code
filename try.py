# # import numpy as np
# # import networkx as nx
# # import matplotlib.pyplot as plt

# # # Define router specifications
# # TYPE_1_PORTS_100G = 8
# # TYPE_1_PORTS_400G = 2
# # TYPE_1_POWER = 250

# # TYPE_2_PORTS_100G = 0
# # TYPE_2_PORTS_400G = 8
# # TYPE_2_POWER = 350

# # # Example input data
# # cities = ["Bengaluru", "Hyderabad", "Mysuru", "Tumkur"]
# # num_customers = [64, 32, 16, 8]  # Example customer data

# # # Example bandwidth matrix (in Gbps)
# # bandwidth_matrix = np.array([
# #     [[0, 0], [6400, 3200], [3200, 2400], [1600, 800]],
# #     [[3200, 6400], [0, 0], [1600, 800], [800, 400]],
# #     [[1600, 3200], [800, 1600], [0, 0], [400, 200]],
# #     [[800, 1600], [400, 800], [200, 400], [0, 0]]
# # ])

# # # Daily bandwidth patterns
# # time_patterns = [
# #     {'time': '1:00AM to 5:30AM', 'low_bandwidth_cities': ["Bengaluru", "Mysuru"]},
# #     {'time': '5:30AM to 1:00AM', 'low_bandwidth_cities': ["Hyderabad"]},
# # ]

# # def calculate_routers(bandwidth_matrix, num_customers):
# #     num_cities = len(bandwidth_matrix)
# #     type_1_needed = np.zeros(num_cities, dtype=int)
# #     type_2_needed = np.zeros(num_cities, dtype=int)
# #     print("check1")
    
# #     for i in range(num_cities):
# #         print("running check cities")
# #         peak_bandwidth = np.max(bandwidth_matrix[i])
# #         num_ports_100G = num_customers[i]
# #         num_ports_400G = int(peak_bandwidth // 400)
        
# #         while num_ports_100G > 0 or num_ports_400G > 0:
# #             print("entered while")
# #             if num_ports_100G <= TYPE_1_PORTS_100G and num_ports_400G <= 2:
# #                 print("entered if")
# #                 type_1_needed[i] += 1
# #                 num_ports_100G -= 8
# #                 num_ports_400G -= 2
# #             else:
# #                 print("entered else")
# #                 type_2_needed[i] += 2
# #                 num_ports_400G -= 8
# #                 num_ports_100G-=8

# #     return type_1_needed, type_2_needed

# # def create_network_graph(cities, bandwidth_matrix):
# #     G = nx.Graph()
    
# #     for i, city in enumerate(cities):
# #         G.add_node(city)
    
# #     for i in range(len(cities)):
# #         for j in range(i + 1, len(cities)):
# #             if bandwidth_matrix[i, j, 0] > 0 or bandwidth_matrix[i, j, 1] > 0:
# #                 G.add_edge(cities[i], cities[j], weight=bandwidth_matrix[i, j, 0] + bandwidth_matrix[i, j, 1])
    
# #     return G

# # # def visualize_network(G):
# # #     pos = nx.spring_layout(G)
# # #     weights = nx.get_edge_attributes(G, 'weight').values()
# # #     nx.draw(G, pos, with_labels=True, node_size=700, node_color='lightblue')
# # #     nx.draw_networkx_edge_labels(G, pos, edge_labels={(u, v): f'{d["weight"]:.1f} Gbps' for u, v, d in G.edges(data=True)})
# # #     nx.draw_networkx_edges(G, pos, width=list(weights))
# # #     plt.show()
# # def visualize_network(G):
# #     pos = nx.spring_layout(G)
# #     weights = nx.get_edge_attributes(G, 'weight').values()
# #     nx.draw(G, pos, with_labels=True, node_size=700, node_color='lightblue')
# #     nx.draw_networkx_edge_labels(G, pos, edge_labels={(u, v): f'{d["weight"]:.1f} Gbps' for u, v, d in G.edges(data=True)})
# #     nx.draw_networkx_edges(G, pos, width=list(weights))
# #     plt.savefig('network_graph.png')

# # def calculate_power_savings(type_1_needed, type_2_needed, time_patterns, cities):
# #     total_power = np.sum(type_1_needed) * TYPE_1_POWER + np.sum(type_2_needed) * TYPE_2_POWER
# #     power_savings = 0
    
# #     for pattern in time_patterns:
# #         low_bandwidth_cities = pattern['low_bandwidth_cities']
# #         for city in low_bandwidth_cities:
# #             city_index = cities.index(city)
# #             power_savings += type_1_needed[city_index] * TYPE_1_POWER + type_2_needed[city_index] * TYPE_2_POWER

# #     return total_power, power_savings

# # if __name__ == "__main__":
# #     type_1_needed, type_2_needed = calculate_routers(bandwidth_matrix, num_customers)
# #     print("Type 1 Routers Needed:", type_1_needed)
# #     print("Type 2 Routers Needed:", type_2_needed)
    
# #     G = create_network_graph(cities, bandwidth_matrix)
# #     visualize_network(G)
    
# #     total_power, power_savings = calculate_power_savings(type_1_needed, type_2_needed, time_patterns, cities)
# #     print(f"Total Power Consumption: {total_power} W")
# #     print(f"Power Savings: {power_savings} W")

# # import numpy as np
# # import networkx as nx
# # import matplotlib.pyplot as plt
# # import json

# # # Define router specifications
# # TYPE_1_PORTS_100G = 8
# # TYPE_1_PORTS_400G = 2
# # TYPE_1_POWER = 250

# # TYPE_2_PORTS_100G = 0
# # TYPE_2_PORTS_400G = 8
# # TYPE_2_POWER = 350

# # # Example input data
# # cities = ["Bengaluru", "Hyderabad", "Mysuru", "Tumkur"]
# # num_customers = [64, 32, 16, 8]  # Example customer data

# # # Example bandwidth matrix (in Gbps)
# # bandwidth_matrix = np.array([
# #     [[0, 0], [6400, 3200], [3200, 2400], [1600, 800]],
# #     [[3200, 6400], [0, 0], [1600, 800], [800, 400]],
# #     [[1600, 3200], [800, 1600], [0, 0], [400, 200]],
# #     [[800, 1600], [400, 800], [200, 400], [0, 0]]
# # ])

# # def calculate_routers(bandwidth_matrix, num_customers):
# #     num_cities = len(bandwidth_matrix)
# #     type_1_needed = np.zeros(num_cities, dtype=int)
# #     type_2_needed = np.zeros(num_cities, dtype=int)
    
# #     for i in range(num_cities):
# #         peak_bandwidth = np.max(bandwidth_matrix[i])
# #         num_ports_100G = num_customers[i]
# #         num_ports_400G = int(peak_bandwidth // 400)
        
# #         # Allocate Type 1 routers for 100G ports first
# #         while num_ports_100G > 0:
# #             type_1_needed[i] += 1
# #             num_ports_100G -= TYPE_1_PORTS_100G
# #             num_ports_400G -= min(num_ports_400G, TYPE_1_PORTS_400G)

# #         # Allocate Type 2 routers for remaining 400G ports
# #         while num_ports_400G > 0:
# #             type_2_needed[i] += 1
# #             num_ports_400G -= TYPE_2_PORTS_400G

# #     return type_1_needed, type_2_needed

# # def create_network_graph(cities, bandwidth_matrix, type_1_needed, type_2_needed):
# #     G = nx.Graph()
    
# #     city_to_routers = {}
# #     router_ports = {}
    
# #     for i, city in enumerate(cities):
# #         G.add_node(city, type='city')
# #         city_to_routers[city] = []

# #         # Add Type 1 routers
# #         for r in range(type_1_needed[i]):
# #             router_name = f'{city}_R1_{r+1}'
# #             G.add_node(router_name, type='router', router_type='Type 1')
# #             G.add_edge(city, router_name)
# #             city_to_routers[city].append(router_name)
# #             router_ports[router_name] = {
# #                 '100G': TYPE_1_PORTS_100G,
# #                 '400G': TYPE_1_PORTS_400G
# #             }
        
# #         # Add Type 2 routers
# #         for r in range(type_2_needed[i]):
# #             router_name = f'{city}_R2_{r+1}'
# #             G.add_node(router_name, type='router', router_type='Type 2')
# #             G.add_edge(city, router_name)
# #             city_to_routers[city].append(router_name)
# #             router_ports[router_name] = {
# #                 '400G': TYPE_2_PORTS_400G
# #             }

# #     # Connect cities through their routers
# #     for i in range(len(cities)):
# #         for j in range(i + 1, len(cities)):
# #             if bandwidth_matrix[i, j, 0] > 0 or bandwidth_matrix[i, j, 1] > 0:
# #                 total_bandwidth = bandwidth_matrix[i, j, 0] + bandwidth_matrix[i, j, 1]
# #                 remaining_bandwidth = total_bandwidth
                
# #                 # Try to connect with available 400G ports on Type 1 routers first
# #                 for router1 in city_to_routers[cities[i]]:
# #                     if remaining_bandwidth <= 0:
# #                         break
# #                     if router_ports[router1]['400G'] > 0:
# #                         for router2 in city_to_routers[cities[j]]:
# #                             if remaining_bandwidth <= 0:
# #                                 break
# #                             if '400G' in router_ports[router2] and router_ports[router2]['400G'] > 0:
# #                                 used_ports = min(remaining_bandwidth // 400, router_ports[router1]['400G'], router_ports[router2]['400G'])
# #                                 if used_ports > 0:
# #                                     G.add_edge(router1, router2, weight=used_ports * 400, upload=used_ports * 400, download=used_ports * 400)
# #                                     router_ports[router1]['400G'] -= used_ports
# #                                     router_ports[router2]['400G'] -= used_ports
# #                                     remaining_bandwidth -= used_ports * 400

# #     return G

# # def graph_to_adjacency_matrix(G):
# #     nodes = list(G.nodes)
# #     adj_matrix = np.zeros((len(nodes), len(nodes)), dtype=int)
    
# #     node_index = {node: idx for idx, node in enumerate(nodes)}
    
# #     for u, v in G.edges():
# #         i, j = node_index[u], node_index[v]
# #         adj_matrix[i, j] = 1
# #         adj_matrix[j, i] = 1  # Assuming undirected graph
    
# #     return adj_matrix, nodes

# # def visualize_network(G, cities):
# #     # Create positions for cities in a grid
# #     pos = {}
# #     num_cities = len(cities)
# #     grid_size = int(np.ceil(np.sqrt(num_cities)))
    
# #     for idx, city in enumerate(cities):
# #         row, col = divmod(idx, grid_size)
# #         pos[city] = (col, -row)
    
# #     # Adjust router positions around their respective cities
# #     for city in cities:
# #         city_x, city_y = pos[city]
# #         routers = [node for node in G.nodes if node.startswith(city) and G.nodes[node]['type'] == 'router']
# #         for i, router in enumerate(routers):
# #             if 'R1' in router:
# #                 pos[router] = (city_x + (i % 2) * 0.5, city_y + (i // 2) * 0.5)
# #             elif 'R2' in router:
# #                 pos[router] = (city_x - 1 + (i % 2) * 0.5, city_y - 1 + (i // 2) * 0.5)

# #     adj_matrix, nodes = graph_to_adjacency_matrix(G)
# #     print("Adjacency Matrix:")
# #     print(adj_matrix)
# #     print("Nodes:")
# #     print(nodes)

# #     plt.figure(figsize=(14, 10))

# #     node_colors = ['skyblue' if G.nodes[node]['type'] == 'city' else 
# #                    'lightgreen' if G.nodes[node]['router_type'] == 'Type 1' else 'orange' 
# #                    for node in G.nodes]
    
# #     nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1000)
# #     nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
    
# #     weights = [d.get('weight', 1) for _, _, d in G.edges(data=True)]
# #     nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.7, edge_color=weights, edge_cmap=plt.cm.Blues)
    
# #     edge_labels = {(u, v): f'{d["upload"]}G' for u, v, d in G.edges(data=True) if 'upload' in d}
# #     nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', font_size=8)
    
# #     plt.title("Network Graph with Router Information")
# #     plt.show()

# # if __name__ == "__main__":
# #     type_1_needed, type_2_needed = calculate_routers(bandwidth_matrix, num_customers)
# #     print("Type 1 Routers Needed:", type_1_needed)
# #     print("Type 2 Routers Needed:", type_2_needed)
    
# #     G = create_network_graph(cities, bandwidth_matrix, type_1_needed, type_2_needed)
# #     visualize_network(G, cities)
# import numpy as np
# import networkx as nx
# import json

# # Define router specifications
# TYPE_1_PORTS_100G = 8
# TYPE_1_PORTS_400G = 2
# TYPE_1_POWER = 250

# TYPE_2_PORTS_100G = 0
# TYPE_2_PORTS_400G = 8
# TYPE_2_POWER = 350

# # Example input data
# cities = ["Bengaluru", "Hyderabad", "Mysuru", "Tumkur"]
# num_customers = [64, 32, 16, 8]  # Example customer data

# # Example bandwidth matrix (in Gbps)
# bandwidth_matrix = np.array([
#     [[0, 0], [6400, 3200], [3200, 2400], [1600, 800]],
#     [[3200, 6400], [0, 0], [1600, 800], [800, 400]],
#     [[1600, 3200], [800, 1600], [0, 0], [400, 200]],
#     [[800, 1600], [400, 800], [200, 400], [0, 0]]
# ])

# def calculate_routers(bandwidth_matrix, num_customers):
#     num_cities = len(bandwidth_matrix)
#     type_1_needed = np.zeros(num_cities, dtype=int)
#     type_2_needed = np.zeros(num_cities, dtype=int)
    
#     for i in range(num_cities):
#         peak_bandwidth = np.max(bandwidth_matrix[i])
#         num_ports_100G = num_customers[i]
#         num_ports_400G = int(peak_bandwidth // 400)
        
#         # Allocate Type 1 routers for 100G ports first
#         while num_ports_100G > 0:
#             type_1_needed[i] += 1
#             num_ports_100G -= TYPE_1_PORTS_100G
#             num_ports_400G -= min(num_ports_400G, TYPE_1_PORTS_400G)

#         # Allocate Type 2 routers for remaining 400G ports
#         while num_ports_400G > 0:
#             type_2_needed[i] += 1
#             num_ports_400G -= TYPE_2_PORTS_400G

#     return type_1_needed, type_2_needed

# def create_network_graph(cities, bandwidth_matrix, type_1_needed, type_2_needed):
#     G = nx.Graph()

#     city_to_routers = {}
#     router_ports = {}

#     # Add city and router nodes
#     for city in cities:
#         G.add_node(city, type='city')
#         city_to_routers[city] = []

#         # Add Type 1 routers
#         for r in range(type_1_needed[city]):
#             router_name = f'{city}_R1_{r+1}'
#             G.add_node(router_name, type='router', router_type='Type 1')
#             G.add_edge(city, router_name)
#             city_to_routers[city].append(router_name)
#             router_ports[router_name] = {
#                 '100G': TYPE_1_PORTS_100G,
#                 '400G': TYPE_1_PORTS_400G
#             }

#         # Add Type 2 routers
#         for r in range(type_2_needed[city]):
#             router_name = f'{city}_R2_{r+1}'
#             G.add_node(router_name, type='router', router_type='Type 2')
#             G.add_edge(city, router_name)
#             city_to_routers[city].append(router_name)
#             router_ports[router_name] = {
#                 '400G': TYPE_2_PORTS_400G
#             }

#     # Create Minimum Spanning Tree (MST) using routers
#     mst = nx.minimum_spanning_tree(G.subgraph([n for n in G if G.nodes[n]['type'] == 'router']))

#     # Allocate bandwidth on MST edges
#     for u, v, edge_data in mst.edges(data=True):
#         total_bandwidth = bandwidth_matrix[cities.index(u)][cities.index(v)].sum()
#         remaining_bandwidth = total_bandwidth

#         # Connect using available 400G ports on routers
#         for router1 in city_to_routers[u]:
#             if remaining_bandwidth <= 0:
#                 break
#             if router_ports[router1]['400G'] > 0:
#                 for router2 in city_to_routers[v]:
#                     if remaining_bandwidth <= 0:
#                         break
#                     if router_ports[router2]['400G'] > 0:
#                         used_ports = min(remaining_bandwidth // 400, router_ports[router1]['400G'], router_ports[router2]['400G'])
#                         if used_ports > 0:
#                             G.add_edge(router1, router2, weight=used_ports * 400, upload=used_ports * 400, download=used_ports * 400)
#                             router_ports[router1]['400G'] -= used_ports
#                             router_ports[router2]['400G'] -= used_ports
#                             remaining_bandwidth -= used_ports * 400

#     return G

# def graph_to_adjacency_matrix(G):
#     nodes = list(G.nodes)
#     adj_matrix = np.zeros((len(nodes), len(nodes)), dtype=int)
    
#     node_index = {node: idx for idx, node in enumerate(nodes)}
    
#     for u, v in G.edges():
#         i, j = node_index[u], node_index[v]
#         adj_matrix[i, j] = 1
#         adj_matrix[j, i] = 1  # Assuming undirected graph
    
#     return adj_matrix, nodes

# def save_graph_to_json(adj_matrix, nodes):
#     graph_data = {
#         "adjacency_matrix": adj_matrix.tolist(),  # Convert numpy array to list
#         "node_names": nodes
#     }
#     graph_json = json.dumps(graph_data, indent=4)
#     print(graph_json)
#     return graph_json

# if __name__ == "__main__":
#     type_1_needed, type_2_needed = calculate_routers(bandwidth_matrix, num_customers)
#     print("Type 1 Routers Needed:", type_1_needed)
#     print("Type 2 Routers Needed:", type_2_needed)
    
#     G = create_network_graph(cities, bandwidth_matrix, type_1_needed, type_2_needed)
    
#     adj_matrix, nodes = graph_to_adjacency_matrix(G)
#     graph_json = save_graph_to_json(adj_matrix, nodes)
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# Define router specifications
TYPE_1_PORTS_100G = 8
TYPE_1_PORTS_400G = 2
TYPE_1_POWER = 250

TYPE_2_PORTS_100G = 0
TYPE_2_PORTS_400G = 8
TYPE_2_POWER = 350

# Example input data
cities = ["Bengaluru", "Hyderabad", "Mysuru", "Tumkur"]
num_customers = [64, 32, 16, 8]  # Example customer data

# Example bandwidth matrix (in Gbps)
bandwidth_matrix = np.array([
    [[0, 0], [6400, 3200], [3200, 2400], [1600, 800]],
    [[3200, 6400], [0, 0], [1600, 800], [800, 400]],
    [[1600, 3200], [800, 1600], [0, 0], [400, 200]],
    [[800, 1600], [400, 800], [200, 400], [0, 0]]
])

# Daily bandwidth patterns
time_patterns = [
    {'time': '1:00AM to 5:30AM', 'low_bandwidth_cities': ["Bengaluru", "Mysuru"]},
    {'time': '5:30AM to 1:00AM', 'low_bandwidth_cities': ["Hyderabad"]},
]

def calculate_routers(bandwidth_matrix, num_customers):
    num_cities = len(bandwidth_matrix)
    type_1_needed = np.zeros(num_cities, dtype=int)
    type_2_needed = np.zeros(num_cities, dtype=int)
    
    for i in range(num_cities):
        peak_bandwidth = np.max(bandwidth_matrix[i])
        num_ports_100G = num_customers[i]
        num_ports_400G = int(peak_bandwidth // 400)
        
        while num_ports_100G > 0 or num_ports_400G > 0:
            if num_ports_100G > 0:
                if num_ports_100G <= TYPE_1_PORTS_100G:
                    type_1_needed[i] += 1
                    num_ports_100G -= TYPE_1_PORTS_100G
                    num_ports_400G -= min(num_ports_400G, TYPE_1_PORTS_400G)
                else:
                    type_2_needed[i] += 1
                    num_ports_400G -= min(num_ports_400G, TYPE_2_PORTS_400G)
                    num_ports_100G -= TYPE_2_PORTS_100G
            elif num_ports_400G > 0:
                if num_ports_400G <= TYPE_1_PORTS_400G:
                    type_1_needed[i] += 1
                    num_ports_100G -= TYPE_1_PORTS_100G
                    num_ports_400G -= TYPE_1_PORTS_400G
                else:
                    type_2_needed[i] += 1
                    num_ports_400G -= TYPE_2_PORTS_400G

    return type_1_needed, type_2_needed

def create_network_graph(cities, bandwidth_matrix, type_1_needed, type_2_needed):
    G = nx.Graph()
    
    for i, city in enumerate(cities):
        G.add_node(city, type='city')
        
        # Add Type 1 routers
        for r in range(type_1_needed[i]):
            router_name = f'{city}_R1_{r+1}'
            G.add_node(router_name, type='router', router_type='Type 1')
            G.add_edge(city, router_name)
        
        # Add Type 2 routers
        for r in range(type_2_needed[i]):
            router_name = f'{city}_R2_{r+1}'
            G.add_node(router_name, type='router', router_type='Type 2')
            G.add_edge(city, router_name)
    
    for i in range(len(cities)):
        for j in range(i + 1, len(cities)):
            if bandwidth_matrix[i, j, 0] > 0 or bandwidth_matrix[i, j, 1] > 0:
                total_bandwidth = bandwidth_matrix[i, j, 0] + bandwidth_matrix[i, j, 1]
                G.add_edge(cities[i], cities[j], weight=total_bandwidth, upload=bandwidth_matrix[i, j, 0], download=bandwidth_matrix[i, j, 1])
    
    return G

def visualize_network(G):
    pos = nx.spring_layout(G, seed=42)  # Fixed layout for consistent visualization
    plt.figure(figsize=(14, 10))

    # Separate positions for cities and routers
    node_colors = []
    node_shapes = []
    for node, data in G.nodes(data=True):
        if data['type'] == 'city':
            node_colors.append('skyblue')
            node_shapes.append('o')
        else:
            if data['router_type'] == 'Type 1':
                node_colors.append('lightgreen')
            else:
                node_colors.append('orange')
            node_shapes.append('s')

    # Draw the nodes and edges
    nx.draw(G, pos, with_labels=True, node_size=1000, node_color=node_colors, font_size=10, font_weight='bold', edge_color='grey')
    
    # Draw edge labels for bandwidth
    edge_labels = {(u, v): f'{d["upload"]}G' for u, v, d in G.edges(data=True) if 'upload' in d}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', font_size=8)
    
    plt.title("Network Graph with Router Information")
    plt.show()
def graph_to_adjacency_matrix(G):
    nodes = list(G.nodes)
    adj_matrix = np.zeros((len(nodes), len(nodes)), dtype=int)
    
    node_index = {node: idx for idx, node in enumerate(nodes)}
    
    for u, v in G.edges():
        i, j = node_index[u], node_index[v]
        adj_matrix[i, j] = 1
        adj_matrix[j, i] = 1  # Assuming undirected graph
    
    return adj_matrix, nodes

def save_graph_to_json(adj_matrix, nodes):
    graph_data = {
        "adjacency_matrix": adj_matrix.tolist(),  # Convert numpy array to list
        "node_names": nodes
    }
    graph_json = json.dumps(graph_data, indent=4)
    print(graph_json)
    return graph_json

if __name__ == "__main__":
    type_1_needed, type_2_needed = calculate_routers(bandwidth_matrix, num_customers)
    print("Type 1 Routers Needed:", type_1_needed)
    print("Type 2 Routers Needed:", type_2_needed)
    
    G = create_network_graph(cities, bandwidth_matrix, type_1_needed, type_2_needed)
    
    adj_matrix, nodes = graph_to_adjacency_matrix(G)
    graph_json = save_graph_to_json(adj_matrix, nodes)

