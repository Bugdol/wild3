import random


class AntColonyOptimization:
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def create_initial_population(self, graph, num_ants, truck_capacity):
        """Функция создает начальную популяцию муравьев"""
        population = []
        for _ in range(num_ants):
            ant = self.create_ant(graph, truck_capacity, population)
            population.append(ant)
        return population

    def create_ant(self, graph, truck_capacity, population):
        """Функция создает муравья и формирует его маршрут в заданном графе с учетом ограничений по вместимости грузовика"""
        ant = []
        start_node = self.get_start_node(graph, 507)  # стартовый узел - вершина 507
        ant.append(start_node)
        visited_nodes = set([start_node])  # множество содержит посещенные вершины
        current_node = start_node  # текущая вершина равняется стартовой вершине
        current_weight = graph.nodes[start_node]['weight']  # устанавливаем вес вершины

        while len(ant) < len(graph):  # проходим по всему графу
            next_node = self.select_next_node(graph, ant, visited_nodes, current_node, truck_capacity - current_weight, population)  # вычисляется следующая вершина
            if next_node is None:  # если нет доступных вершин (это может быть по причине того, что не удовлетворяются ограничения по вместимости грузовика)
                break
            ant.append(next_node)
            visited_nodes.add(next_node)
            current_node = next_node  # current_node устанавливается равным next_node, чтобы следующая итерация цикла начиналась с выбранной вершины
            current_weight += graph.nodes[next_node]['weight']  # увеличиваем вес маршрута на следующую вершину

        ant.append(start_node)  # добавляем стартовый узел в конец списка(т.к. грузовик вернулся обратно)
        return ant

    def get_start_node(self, graph, label):
        """Функция находит стартовую вершину в графе по заданному метке (label)"""
        for node in graph.nodes(data=True):
            if str(node[0]) == str(label):  # Сравниваем метку вершины с заданной меткой
                return node[0]
        raise ValueError("Start node with label '{}' does not exist in the graph.".format(label))

    def select_next_node(self, graph, ant, visited_nodes, current_node, remaining_capacity, population):
        """Функция выбирает следующую вершину для муравья из списка соседних вершин
        Функция использует список соседних вершин от текущей вершины, исключая вершины, которые уже посещены текущим муравьем и не удовлетворяют ограничениям вместимости грузовика.
        - Если нет доступных соседних вершин, функция возвращает None.
        - Функция вычисляет вероятности выбора соседних вершин и использует их в соответствии с весами для выбора следующей вершины"""
        neighboring_nodes = list(graph.neighbors(current_node))  # Получаем список соседних вершин от текущей вершины
        neighboring_nodes = [node for node in neighboring_nodes if graph.nodes[node]['weight'] <= remaining_capacity and node not in visited_nodes and not any(
            node in ant for ant in population)]  # Фильтруем список соседних вершин, исключая уже посещенные вершины, вершины не удовлетворяющие ограничениям вместимости грузовика и вершины, которые уже присутствуют в маршрутах других муравьев
        if not neighboring_nodes:  # Если нет доступных соседних вершин, возвращаем None
            return None
        probabilities = self.calculate_probabilities(graph, ant, current_node, neighboring_nodes, remaining_capacity, self.alpha, self.beta)  # Вычисляем вероятности выбора соседних вершин
        next_node = random.choices(neighboring_nodes, weights=probabilities)[0]  # Выбираем следующую вершину на основе вероятностей. Если у нас есть несколько вершин с одинаковыми вероятностями, то из них выбирается одна случайным образом с использованием функции random.choices()
        return next_node

    def calculate_probabilities(self, graph, ant, current_node, neighboring_nodes, remaining_capacity, alpha, beta):
        """Функция вычисляет вероятности выбора соседних вершин на основе значения феромона, расстояния и ограничений на вес в графе
        Чем больше вероятность, тем больше шанс, что выберут эту вершину и наоборот"""
        total = 0.0
        probabilities = [
            (graph[current_node][node]['pheromone'] ** alpha) * ((1.0 / graph[current_node][node]['distance']) ** beta)
            for node in neighboring_nodes
            if graph.nodes[node]['weight'] <= remaining_capacity  # Вычисляется вероятность выбора соседней вершины на основе формулы (1),  alpha отвечает за влияние феромонов, а beta за влияние расстояния на вероятность выбора следующей вершины.
        ]
        total = sum(probabilities)
        probabilities = [p / total for p in probabilities] if total > 0 else [1.0 / len(neighboring_nodes)] * len(neighboring_nodes)  # Если total больше 0 (есть доступные соседние вершины), то все вероятности в списке probabilities делятся на total для нормализации, что сумма вероятностей была равна 1.
        # Если total равно 0 (нет доступных соседних вершин), то все вероятности в списке probabilities устанавливаются равными 1.0 / len(neighboring_nodes) для того, чтобы сумма вероятностей была равна 1.
        return probabilities

    def calculate_route_length(self, graph, route):
        """Функция вычисляет общую длину заданного маршрута в графе"""
        length = 0
        for i in range(len(route) - 1):
            current_node = route[i]
            next_node = route[i + 1]
            length += graph[current_node][next_node]['distance']  # вычисляет расстояние между каждой последовательной парой узлов и суммирует
        return length

    def calculate_total_distance(self, graph, population):
        """Функция вычисляет общее расстояние для совокупности маршрутов в графе"""
        total_distance = 0
        for ant in population:
            route_length = self.calculate_route_length(graph, ant)  # вычисляет длину маршрута для каждого муравья в популяции
            total_distance += route_length  # суммирует эти расстояния
        return total_distance

    def update_pheromones(self, graph, population, evaporation_rate, Q):
        """Обновление феромонов для каждого муравья в популяции
        Главная идея заключается в том, что муравьи, проходящие более короткие маршруты к цели, будут оставлять больше феромона на ребрах этого маршрута,
        что приведет к укреплению этого маршрута и с большей вероятностью приведет других муравьев к выбору более коротких маршрутов."""
        for ant in population:  # Вычисление изменения феромонов для данного муравья
            delta_pheromones = Q / self.calculate_route_length(graph, ant)  # Определяет, какое количество феромона должно быть добавлено к каждому ребру на маршруте муравья
            for i in range(len(ant) - 1):  # Обновление феромонов для каждого ребра в маршруте муравья
                current_node = ant[i]
                next_node = ant[i + 1]
                pheromone = graph[current_node][next_node]['pheromone']  # Получение текущего значения феромона для ребра
                updated_pheromone = (1 - evaporation_rate) * pheromone + delta_pheromones  # Обновление значения феромона по формуле (2)
                graph[current_node][next_node]['pheromone'] = updated_pheromone  # Присвоение обновленного значения феромона ребру в графе

    def local_search(self, graph, route):
        """Функция осуществляет локальный поиск для улучшения заданного маршрута"""
        best_route = route
        best_distance = self.calculate_route_length(graph, route)  # Вычисляем длину текущего маршрута и присваиваем ее лучшей длине маршрута.

        for i in range(1, len(route) - 2):  # Итерируем по индексам маршрута, начиная с 1 до предпоследнего индекса (не затрагивая склад).
            for j in range(i + 1, len(route) - 1):  # Вложенный цикл итерации по индексам маршрута, начиная со следующего индекса i до последнего индекса.
                # Формируем новый маршрут путем обмена ребрами между индексами i и j с помощью реверсии среза маршрута.
                new_route = route[:i] + route[i:j + 1][::-1] + route[j + 1:]
                # Вычисляем длину нового маршрута.
                new_distance = self.calculate_route_length(graph, new_route)
                if new_distance < best_distance:
                    # Если новая длина маршрута меньше лучшей длины маршрута, обновляем лучший маршрут и лучшую длину.
                    best_route = new_route
                    best_distance = new_distance

        return best_route

    def two_opt_ACO(self, graph, num_ants, truck_capacity, evaporation_rate, Q, max_iterations):
        """Функция, которая создает новые популяции муравьев и выбирает из них лучшую (по минимальному общему расстоянию)"""
        best_distances = []  # Список для хранения лучших расстояний на каждой итерации
        best_distance = float('inf')  # Инициализация лучшего расстояния как бесконечность
        best_solution = None  # Инициализация лучшего решения как None

        for iteration in range(max_iterations):  # создается новая популяция муравьев для поиска оптимальных маршрутов
            population = self.create_initial_population(graph, num_ants, truck_capacity)
            total_distance = self.calculate_total_distance(graph, population)

            if total_distance < best_distance:  # если total_distance < best_distance, то обновляются значения best_distance и best_solution.
                best_distance = total_distance
                best_solution = population

            self.update_pheromones(graph, population, evaporation_rate, Q)  # обновление феромонов на ребрах графа на основе текущей популяции муравьев.

            for ant in population:
                ant = self.local_search(graph, ant)  # для каждого муравья выполняется локальный поиск (2-opt)

            best_distances.append(best_distance)

        return best_solution, best_distance, best_distances
