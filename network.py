import sys
import networkx as nx
from networkx.algorithms import bipartite
import operator
G = nx.Graph()

network_file = sys.argv[1]

with open(network_file, 'r') as f:
    next(f)
    for l in f:
    
        cell, cytokine, _, _, _, confidence, docs, nsentences = l.strip().split('\t')
        G.add_edge(cell, cytokine, nsentences=nsentences, confidence=confidence, docs=docs)
        G.node[cell]['type'] = 'cell'
        G.node[cytokine]['type'] = 'cytokine'
        G.node[cell]['bipartite'] = 0
        G.node[cytokine]['bipartite'] = 1
        
print("============== tolapc network ============")
print(G.number_of_edges())
print(G.number_of_nodes())
#print(nx.clustering(G))
#print(nx.degree(G))
print(nx.is_connected(G))
top_nodes = set(n for n,d in G.nodes(data=True) if d['bipartite']==0)
bottom_nodes = set(G) - top_nodes
print("cells", len(top_nodes))
print("cykines", len(bottom_nodes))
#print(top_nodes, bottom_nodes)
print(bipartite.density(G, top_nodes))
print(bipartite.density(G, bottom_nodes))
#print(bipartite.maximum_matching(G))
#print(bipartite.clustering(G))
#print(bipartite.node_redundancy(G))
print(nx.info(G))
print(nx.density(G))
#print(nx.degree_histogram(G))
print(nx.diameter(G))
print(nx.radius(G))
print(nx.center(G))

print()
print()

network2_file = sys.argv[2]
H = nx.Graph()
with open(network2_file, 'r') as f:
    for l in f:
        cytokine, cell, docs = l.strip().split('\t')
        H.add_edge(cell, cytokine, docs=docs)
        H.node[cell]['type'] = 'cell'
        H.node[cytokine]['type'] = 'cytokine'
        H.node[cell]['bipartite'] = 0
        H.node[cytokine]['bipartite'] = 1
print("============== immuneXpresso network ==================")
print(H.number_of_edges())        
print(H.number_of_nodes())
#print(nx.clustering(H))
#print(nx.degree(H))
print(nx.is_connected(H))
top_nodes = set(n for n,d in H.nodes(data=True) if d['bipartite']==0)
bottom_nodes = set(H) - top_nodes
print("cells", len(top_nodes))
print("cykines", len(bottom_nodes))
#print(top_nodes, bottom_nodes)
print(bipartite.density(H, top_nodes))
print(bipartite.density(H, bottom_nodes))
#print(bipartite.maximum_matching(H))
print(nx.info(H))
print(nx.density(H))
#print(nx.degree_histogram(H))
print(nx.diameter(H))
print(nx.radius(H))
print(nx.center(H))


print()
print()
print("======= intersection of nodes =============")
#print(nx.intersection(G, H))
print("common nodes:")
common = set(G.nodes()) & set(H.nodes())
print(common)
print(len(common))
#print(G.subgraph(common).edges())
#print(H.subgraph(common).edges())


print("============= intersection of nodes and edges ============")
common_graph = nx.intersection(G.subgraph(common), H.subgraph(common))
print(len(common_graph.edges()))
g_with_unique_nodes=G.subgraph(common).copy()
g_with_unique_nodes.remove_nodes_from(n for n in G.subgraph(common) if n not in H.subgraph(common))
#print(common_graph.edges())
node_count = {}
for p in common_graph.edges():
    if p[0] not in node_count:
        node_count[p[0]] = 0
    if p[1] not in node_count:
        node_count[p[1]] = 0
    node_count[p[0]] += 1
    node_count[p[1]] += 1
print(sorted(node_count.items(), key=operator.itemgetter(1)))
#top_nodes = set(n for n,d in common_graph.nodes(data=True) if d['type']=="cell")
#bottom_nodes = set(common_graph) - top_nodes
#print("cells", len(top_nodes))
#print("cykines", len(bottom_nodes))


print("============== nodes on G but not on H ==============0")
new_nodes = set(G.nodes()) - set(H.nodes())
g_edges_g_nodes = G.subgraph(new_nodes)
#for n in new_nodes:
#    print(n)
print("new nodes in ITREL", len(new_nodes))
print("new cell in ITREL", len([n for d,n in g_edges_g_nodes.nodes(data=True) if n['type'] == "cell"]))
print("new cyto in ITREL", len([n for d,n in g_edges_g_nodes.nodes(data=True) if n['type'] == "cytokine"]))

print("g edges using nodes unique to G", len(g_edges_g_nodes.edges()))

g_edges_common_nodes = nx.difference(G.subgraph(common), H.subgraph(common))
print("g edges using common nodes", len(g_edges_common_nodes.edges()))
for e in g_edges_common_nodes.edges():
    print(e, G[e[0]][e[1]])
g_edges_common_nodes.add_nodes_from(G.nodes())
g_edges_g_nodes.add_nodes_from(G.nodes())
common_graph.add_nodes_from(G.nodes())
g_and_common_edges = nx.difference(G, common_graph)
g_and_common_edges = nx.difference(g_and_common_edges, g_edges_common_nodes)
g_and_common_edges = nx.difference(g_and_common_edges, g_edges_g_nodes)
print("g edges using g and common nodes", len(g_and_common_edges.edges()))
node_count = {}
for p in g_and_common_edges.edges():
    if p[0] not in node_count:
        node_count[p[0]] = 0
    if p[1] not in node_count:
        node_count[p[1]] = 0
    node_count[p[0]] += 1
    node_count[p[1]] += 1
for p in g_edges_g_nodes.edges():
    if p[0] not in node_count:
        node_count[p[0]] = 0
    if p[1] not in node_count:
        node_count[p[1]] = 0
    node_count[p[0]] += 1
    node_count[p[1]] += 1
print(sorted(node_count.items(), key=operator.itemgetter(1)))
#for e in g_and_common_edges.edges():
#    print(e)

g_apc = G["professional antigen presenting cell"]
h_apc = H["professional antigen presenting cell"]
print("========== APC ===============")
for e in sorted(g_apc.keys()):
    print(e, g_apc[e]['confidence'], g_apc[e]['nsentences'], g_apc[e]['docs'], e in h_apc)

#for e in sorted(h_apc.keys()):
#    print(e)
    
#common_apc = G["professional antigen presenting cell"].update(H["professional antigen presenting cell"])
#print(common_apc)
print("========== DC ===============")
g_dc = G["dendritic cell"]
h_dc = H["dendritic cell"]
#for e in sorted(g_dc.keys()):
#    print(e, g_dc[e]['confidence'], g_dc[e]['nsentences'], g_dc[e]['docs'], e in h_dc)
print("G graph:", len(g_dc.keys()))
print("unique to g", len([e for e in g_dc if e not in h_dc]))
#for e in sorted(h_apc.keys()):
#    print(e)
    
#common_apc = G["professional antigen presenting cell"].update(H["professional antigen presenting cell"])
#print(common_apc)
#print("common between DC and APC", set(g_dc.keys()) & set(g_apc.keys()))
#print("common between B cells and APC", set(G["B cell"]) & set(g_apc.keys()))
nx.write_graphml(G, './graph.xml')

print("G graph paths")
p = nx.shortest_path_length(G)
g_paths = []
for i in p:
    for j in p[i]:
        if p[i][j] == 7:
            print(i, j)
            g_paths += [x for x in nx.all_shortest_paths(G, i, j)]
#print(g_paths)
sys.exit()
print()
print("H graph paths")
p = nx.shortest_path_length(H)
longest_paths = nx.Graph()
for i in p:
    for j in p[i]:
        if p[i][j] == 6:
            h_paths = [x for x in nx.all_shortest_paths(H, i, j)]
            #print(x)
            for x in h_paths:
                for y in g_paths:
                    xy_intersection = set(x).intersection(set(y))
                    if len(xy_intersection) > 3:
                        #print(x)
                        #print(y)
                        #print(xy_intersection)

                        #print()
                        longest_paths.add_path(y, source="icrel")
                        longest_paths.add_path(x, source="immunexpresso")
for e in longest_paths.edges():
    if e in H.edges() and e in G.edges():
        longest_paths[e[0]][e[1]]['source'] = 'both'
        print("both", e)
nx.write_graphml(longest_paths, './longest_paths.xml')
                        
                        
