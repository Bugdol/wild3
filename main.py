import pandas as pd
import math
from graph import create_distance_matrix, create_graph
from aco import AntColonyOptimization

alpha = 2  # Параметр, определяющий вес феромонов
beta = 5  # Параметр, определяющий вес эвристической информации
evaporation_rate = 0.2  # Скорость испарения феромонов
Q = 100  # Коэффициент для вычисления изменения феромонов при обновлении
max_iterations = 1  # Максимальное количество итераций алгоритма
file_name = "wb_school_task_3.csv"


def main():
    try:
        df = pd.read_csv(file_name)
    except FileNotFoundError:
        print(f"File '{file_name}' not found.")
        return

    distance_df = create_distance_matrix(df, 'pickup_latitude', 'pickup_longitude', 'dst_office_id')
    graph = create_graph(df, distance_df)
    total_weight = sum(graph.nodes[node]['weight'] for node in graph.nodes)  # Вычисляем общий вес всех узлов графа
    truck_capacity = 2000  # Ваша грузоподъемность
    num_ants = max(1, math.ceil(total_weight / truck_capacity))  # Вычисляем количество муравьев в зависимости от общего веса и грузоподъемности

    aco = AntColonyOptimization(alpha, beta)  # Инициализируем класс

    best_solution, best_distance, best_distances = aco.two_opt_ACO(graph, num_ants, truck_capacity, evaporation_rate, Q, max_iterations)

    # Запись результатов в файл
    with open('output.txt', 'w') as f:
        f.write(f"Best Total Distance: {best_distance}\n")
        for idx, ant in enumerate(best_solution):
            ant_weight = sum(graph.nodes[node]['weight'] for node in ant)
            f.write(f"Route {idx + 1}: {ant}, Total Weight: {ant_weight}\n")


if __name__ == "__main__":
    main()





