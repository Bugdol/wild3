import pandas as pd
import math
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import networkx as nx
from graph import create_distance_matrix, create_graph
from aco import AntColonyOptimization

ALPHA = 2  # Параметр, определяющий вес феромонов
BETA = 5  # Параметр, определяющий вес эвристической информации
EVAPORATION_RATE = 0.3  # Скорость испарения феромонов
Q = 100  # Коэффициент для вычисления изменения феромонов при обновлении
MAX_ITERATIONS = 5  # Максимальное количество итераций алгоритма
file_name = "wb_school_task_3.csv"


def plot_convergence(best_distances):
    """
    Визуализирует сходимость алгоритма ACO.
    """
    plt.plot(range(MAX_ITERATIONS), best_distances)
    plt.xlabel('Iteration')
    plt.ylabel('Best Distance')
    plt.title('Сходимость two_opt_ACO')


def plot_routes(graph, best_solution):
    """
    Визуализирует маршруты для лучших решений, найденных алгоритмом ACO.
    """
    for i, ant_route in enumerate(best_solution):
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_title(f'Маршрут {i + 1}')

        pos = nx.get_node_attributes(graph, 'pos')  # Получаем позиции узлов из атрибутов графа
        G = graph.subgraph(ant_route)  # Создаем подграф для текущего маршрута муравья

        nx.draw_networkx_nodes(G, pos, ax=ax, node_color='blue', node_size=20)  # Отображаем узлы на графике

        edges = [(ant_route[j], ant_route[j + 1]) for j in range(len(ant_route) - 1)]  # Создаем ребра для текущего маршрута муравья
        nx.draw_networkx_edges(G, pos, edgelist=edges, ax=ax, arrows=True,arrowstyle='->')  # Отображаем ребра на графике со стрелками

        ax.axis('off')  # Отключаем отображение осей для удаления ненужных линий и меток осей

def main(visualize_convergence=True, visualize_routes=True):
    try:
        df = pd.read_csv(file_name)
    except FileNotFoundError:
        print(f"File '{file_name}' not found.")
        return

    distance_df = create_distance_matrix(df, 'pickup_latitude', 'pickup_longitude', 'dst_office_id')
    graph = create_graph(df, distance_df)
    TRUCK_CAPACITY = 2000  # Грузоподъемность грузовика
    total_weight = sum(graph.nodes[node]['weight'] for node in graph.nodes)  # Вычисляем общий вес всех узлов графа
    num_ants = max(1, math.ceil(total_weight / TRUCK_CAPACITY))  # Вычисляем количество грузовиков в зависимости от общего веса и грузоподъемности

    aco = AntColonyOptimization(ALPHA, BETA)  # Инициализируем класс

    best_solution, best_distance, best_distances = aco.two_opt_ACO(graph, num_ants, TRUCK_CAPACITY, EVAPORATION_RATE, Q, MAX_ITERATIONS)

    # Запись результатов в файл
    with open('output.txt', 'w') as f:
        f.write(f"Best Total Distance: {best_distance}\n")
        for idx, ant in enumerate(best_solution):
            ant_weight = sum(graph.nodes[node]['weight'] for node in ant) #считаем для каждого маршрута вес
            f.write(f"Route {idx + 1}: {ant}, Total Weight: {ant_weight}\n")

    # Визуализация сходимости алгоритма, если параметр visualize_convergence равен True
    if visualize_convergence:
        plot_convergence(best_distances)
        plt.show()

    # Визуализация маршрутов, если параметр visualize_routes равен True
    if visualize_routes:
        plot_routes(graph, best_solution)
        plt.show()


if __name__ == "__main__":
    main(visualize_convergence=True, visualize_routes=False)









