import pandas as pd
import math
from graph import create_distance_matrix, create_graph
from aco import create_initial_population, calculate_total_distance, update_pheromones, local_search, calculate_route_length, two_opt_ACO


alpha = 2  # Параметр, определяющий вес феромонов
beta = 5  # Параметр, определяющий вес эвристической информации
evaporation_rate = 0.2  # Скорость испарения феромонов
Q = 100  # Коэффициент для вычисления изменения феромонов при обновлении
max_iterations = 1  # Максимальное количество итераций алгоритма
fileName = "wb_school_task_3.csv"


def calculate_total_distance(graph, population):
    """Функция вычисляет общее расстояние для совокупности маршрутов в графе"""
    total_distance = 0
    for ant in population:
        route_length = calculate_route_length(graph, ant)  # вычисляет длину маршрута для каждого муравья в популяции
        total_distance += route_length  # суммирует эти расстояния
    return total_distance


def update_pheromones(graph, population, evaporation_rate, Q):
    """Обновление феромонов для каждого муравья в популяции
    Главная идея заключается в том, что муравьи, проходящие более короткие маршруты к цели, будут оставлять больше феромона на ребрах этого маршрута,
    что приведет к укреплению этого маршрута и с большей вероятностью приведет других муравьев к выбору более коротких маршрутов."""
    for ant in population:  # Вычисление изменения феромонов для данного муравья
        delta_pheromones = Q / calculate_route_length(graph, ant)  # Определяет, какое количество феромона должно быть добавлено к каждому ребру на маршруте муравья
        for i in range(len(ant) - 1):  # Обновление феромонов для каждого ребра в маршруте муравья
            current_node = ant[i]
            next_node = ant[i + 1]
            pheromone = graph[current_node][next_node]['pheromone']  # Получение текущего значения феромона для ребра
            updated_pheromone = (1 - evaporation_rate) * pheromone + delta_pheromones  # Обновление значения феромона по формуле (2)
            graph[current_node][next_node]['pheromone'] = updated_pheromone  # Присвоение обновленного значения феромона ребру в графе


def local_search(graph, route):
    """Функция осуществляет локальный поиск для улучшения заданного маршрута"""
    best_route = route
    best_distance = calculate_route_length(graph, route) #Вычисляем длину текущего маршрута и присваиваем ее лучшей длине маршрута.

    for i in range(1, len(route) - 2): #Итерируем по индексам маршрута, начиная с 1 до предпоследнего индекса (не затрагивая склад).
        for j in range(i + 1, len(route) - 1): #Вложенный цикл итерации по индексам маршрута, начиная со следующего индекса i до последнего индекса.
            # Формируем новый маршрут путем обмена ребрами между индексами i и j с помощью реверсии среза маршрута.
            new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
            # Вычисляем длину нового маршрута.
            new_distance = calculate_route_length(graph, new_route)
            if new_distance < best_distance:
                # Если новая длина маршрута меньше лучшей длины маршрута, обновляем лучший маршрут и лучшую длину.
                best_route = new_route
                best_distance = new_distance

    return best_route


def two_opt_ACO(graph, num_ants, truck_capacity, evaporation_rate, Q, max_iterations):
    """Функция, которая создает новые популяции муравьев и выбирает из них лучшую (по минимальному общему расстоянию)"""
    best_distances = []  # Список для хранения лучших расстояний на каждой итерации
    best_distance = float('inf')  # Инициализация лучшего расстояния как бесконечность
    best_solution = None  # Инициализация лучшего решения как None

    for iteration in range(max_iterations):  # создается новая популяция муравьев для поиска оптимальных маршрутов
        population = create_initial_population(graph, num_ants, truck_capacity)
        total_distance = calculate_total_distance(graph, population)

        if total_distance < best_distance:  # если total_distance < best_distance, то обновляются значения best_distance и best_solution.
            best_distance = total_distance
            best_solution = population

        update_pheromones(graph, population, evaporation_rate, Q)  # обновление феромонов на ребрах графа на основе текущей популяции муравьев.

        for ant in population:
            ant = local_search(graph, ant) #для каждого муравья выполняется локальный поиск (2-opt)

        best_distances.append(best_distance)

    return best_solution, best_distance, best_distances


def main():
    try:
        df = pd.read_csv(fileName)
    except FileNotFoundError:
        print(f"File '{fileName}' not found.")
        return

    distance_df = create_distance_matrix(df, 'pickup_latitude', 'pickup_longitude', 'dst_office_id')
    graph = create_graph(df, distance_df)
    total_weight = sum(graph.nodes[node]['weight'] for node in graph.nodes)  # Вычисляем общий вес всех узлов графа
    truck_capacity = 2000  # Ваша грузоподъемность
    num_ants = max(1, math.ceil(
        total_weight / truck_capacity))  # Вычисляем количество муравьев в зависимости от общего веса и грузоподъемности
    best_solution, best_distance, best_distances = two_opt_ACO(graph, num_ants, truck_capacity, evaporation_rate, Q, max_iterations)

    # Запись результатов в файл
    with open('output.txt', 'w') as f:
        f.write(f"Best Total Distance: {best_distance}\n")
        for idx, ant in enumerate(best_solution):
            ant_weight = sum(graph.nodes[node]['weight'] for node in ant)
            f.write(f"Route {idx + 1}: {ant}, Total Weight: {ant_weight}\n")


if __name__ == "__main__":
    main()



