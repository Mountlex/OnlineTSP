# Download osm maps

import osmnx as ox
import networkx

def download(place,filename=None):
    if not filename:
        filename = place.split(',')[0]
    G = ox.graph_from_place(place, network_type="drive")
    G = G.to_undirected()
    G = networkx.convert_node_labels_to_integers(G)

    ox.plot_graph(G, show=False, save=True, close=True, filepath=f"data/osm/{filename}.svg")
    ox.save_graphml(G, f"data/osm/{filename}.graphml")
    

download("Manhattan")

