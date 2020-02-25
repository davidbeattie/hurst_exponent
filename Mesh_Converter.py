import numpy as np

def generate_vertices(heightmap, size, max_height):
    vertices = []
    base = (-1, -0.75, -1)

    step_x = size/(heightmap.shape[0] - 1)
    step_y = size/(heightmap.shape[1] - 1)

    for x in range(heightmap.shape[0]):
        for y in range(heightmap.shape[1]):
            x_coord = base[0] + step_x * x 
            y_coord = base[1] + max_height * heightmap[x][y]
            z_coord = base[2] + step_y * y
            vertices.append((x_coord, y_coord, z_coord))
    print("Vertices generated")
    return vertices

def generate_tris(heightmap):
    edges = []
    surfaces = []

    for x in range(heightmap.shape[0] - 1):
        for y in range(heightmap.shape[0] - 1):
            base = x * heightmap.shape[0] + y
            a = base
            b = base + 1
            c = base + heightmap.shape[0] + 1
            d = base + heightmap.shape[0]
            edges.append((a, b))
            edges.append((b, c))
            edges.append((c, a))
            edges.append((c, d))
            edges.append((d, a))
            surfaces.append((a, b, c))
            surfaces.append((a, c, d))
    print("Edges, surfaces generated")
    return edges, surfaces

def export_obj(vertices, tris, filename):
    file = open(filename, "w")
    for vertex in vertices:
        file.write("v " + str(vertex[0]) + " " + str(vertex[1]) + " " + str(vertex[2]) + "\n")
    for tri in tris:
        file.write("f " + str(tri[2]+1) + " " + str(tri[1]+1) + " " + str(tri[0]+1) + "\n")
    file.close()
    print(filename, "saved")
    return