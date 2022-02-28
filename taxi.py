import json
import geojson
from shapely.geometry import Point
from shapely.geometry import shape
import csv
import random
from datetime import datetime
import os

import networkx

def node_in_shape(node, shape):
    x = float(node[1]['x'])
    y = float(node[1]['y'])
    p = Point(x,y)      
    return shape.contains(p)

def load_nodes_in_zones(graph_file, zone_file):
    graph = networkx.read_graphml(graph_file)
    with open(zone_file) as f:
        gj = geojson.load(f)
    
        zone_nodes = dict()
        for zone in gj['features']:
            id = str(zone['properties']['locationid'])
            s = shape(zone['geometry'])
            zone_nodes[id] = [node[0] for node in graph.nodes(data=True) if node_in_shape(node, s)]

    return zone_nodes 

start = datetime.fromisoformat('2021-01-01 00:00:00')

def generate_instance(output_file, i, m, zone_nodes, length=50 ,taxi_file="data/yellow_taxi_data_jan1.csv"):
    with open(taxi_file, newline='') as inputfile:
        with open(output_file, 'w', newline='') as outputfile:
            reader = csv.DictReader(inputfile, delimiter=',')
            writer = csv.writer(outputfile, delimiter=',')
            n = 0
            entries = list(reader)



            for row in entries[i::m]:      
                date_raw = row['tpep_pickup_datetime']
                duration = datetime.fromisoformat(date_raw) - start   

                t = max(0, round(duration.total_seconds() / 60))
                x = row['PULocationID']
        
                if x in zone_nodes and len(zone_nodes[x]) > 0:
                    node = random.choice(zone_nodes[x])
                    writer.writerow([node,t])
                    n += 1
                    if n == length:
                        break
            outputfile.flush()
            outputfile.close()
                    



if not os.path.exists("data/manhattan_zones.json"):
    with open("data/manhattan_zones.json", "w") as file:
        zone_nodes = load_nodes_in_zones("data/osm/Manhattan.graphml", "data/taxi.json")
        json.dump(zone_nodes, file)
else:
    with open("data/manhattan_zones.json", "r") as file:
        zone_nodes = json.load(file)

m = 50
for i in range(m):
    generate_instance(f"data/instances/instance{i}.csv",i, m , zone_nodes, length=50)

