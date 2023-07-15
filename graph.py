import pandas as pd
import networkx as nx
from geopy.distance import distance
from sklearn.metrics import pairwise_distances


def create_distance_matrix(dataframe, lat_col, lon_col, index_col):
    # Создание массива координат на основе указанных столбцов
    coords = dataframe[[lat_col, lon_col]].values

    # Вычисление матрицы расстояний
    distance_matrix = pairwise_distances(coords, metric=lambda x, y: distance(x, y).km)

    # Создание DataFrame distance_df из матрицы расстояний
    distance_df = pd.DataFrame(distance_matrix, index=dataframe[index_col], columns=dataframe[index_col])

    return distance_df


def create_graph(dataframe, distance_df):
    # Создаем пустой граф
    graph = nx.Graph()

    # Добавляем вершины в граф с координатами
    for index, row in distance_df.iterrows():
        vertex = index
        weight = dataframe.loc[dataframe['dst_office_id'] == vertex, 'count'].values[0]
        pickup_latitude = dataframe.loc[dataframe['dst_office_id'] == vertex, 'pickup_latitude'].values[0]
        pickup_longitude = dataframe.loc[dataframe['dst_office_id'] == vertex, 'pickup_longitude'].values[0]
        graph.add_node(vertex, weight=weight, pos=(pickup_latitude, pickup_longitude))

    # Добавляем ребра в граф с соответствующими расстояниями
    for index, row in distance_df.iterrows():
        source_vertex = index
        for column, distance in row.items():
            target_vertex = column
            if source_vertex != target_vertex:
                # Инициализация феромонов на ребре
                pheromone = 1.0
                graph.add_edge(source_vertex, target_vertex, distance=distance, pheromone=pheromone)

    return graph
