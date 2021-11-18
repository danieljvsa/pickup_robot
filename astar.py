import pygame
import math
from queue import PriorityQueue
import csv
import pandas as pd

WIDTH = 800
WIN = pygame.display.set_mode((WIDTH, WIDTH))
pygame.display.set_caption("A* Path Finding Algorithm")

RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 255, 0)
YELLOW = (255, 255, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
PURPLE = (128, 0, 128)
ORANGE = (255, 165 ,0)
GREY = (128, 128, 128)
TURQUOISE = (64, 224, 208)

class Spot:
    def __init__(self, row, col, width, total_rows):
        self.row = row
        self.col = col
        self.x = row * width
        self.y = col * width
        self.color = WHITE
        self.neighbors = []
        self.width = width
        self.total_rows = total_rows
        self.value = 0

    def get_pos(self):
        return self.row, self.col

    def is_closed(self):
        return self.color == RED

    def is_open(self):
        return self.color == GREEN

    def is_barrier(self):
        return self.color == BLACK

    def is_start(self):
        return self.color == ORANGE

    def is_end(self):
        return self.color == TURQUOISE

    def is_package(self):
        return self.color == YELLOW, self.value == 1

    def reset(self):
        self.color = WHITE

    def make_start(self):
        self.color = ORANGE

    def make_closed(self):
        self.color = RED

    def make_open(self):
        self.color = GREEN

    def make_barrier(self):
        self.color = BLACK

    def make_end(self):
        self.color = TURQUOISE

    def make_package(self):
        self.color = YELLOW
        self.value = 1

    def make_path(self):
        self.color = PURPLE

    def draw(self, win):
        pygame.draw.rect(win, self.color, (self.x, self.y, self.width, self.width))

    def update_neighbors(self, grid):
        self.neighbors = []
        if self.row < self.total_rows - 1 and not grid[self.row + 1][self.col].is_barrier(): # DOWN
            self.neighbors.append(grid[self.row + 1][self.col])

        if self.row > 0 and not grid[self.row - 1][self.col].is_barrier(): # UP
            self.neighbors.append(grid[self.row - 1][self.col])

        if self.col < self.total_rows - 1 and not grid[self.row][self.col + 1].is_barrier(): # RIGHT
            self.neighbors.append(grid[self.row][self.col + 1])

        if self.col > 0 and not grid[self.row][self.col - 1].is_barrier(): # LEFT
            self.neighbors.append(grid[self.row][self.col - 1])

    def __lt__(self, other):
        return False

def calc_pit(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    a = (x2 - x1)**2 + (y2 - y1)**2
    b = math.sqrt(a)
    return b

def h(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return abs(x1 - x2) + abs(y1 - y2)

#funcao para calcular a distancia do inicio aos pacotes (alterar para fazer para todos os pontos)
def calc_points(grid, start, end, packages):
    points = packages
    count = 0
    count_data = 1
    scores = []
    n_scores = []
    custom_path = []
    custom_path.append(start)
    temp1 = 0
    temp2 = 0
    data = []
    custom_path_header = ['Ponto', 'Coordenadas', 'Custo']

    #print(points)
    data.append('Ponto de partida')
    data.append(start.get_pos())
    data.append(0)
    with open('robot_path.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(custom_path_header)
        writer.writerow(data)

    data = []

    for i in range(0, len(packages)):
        score = calc_pit(start.get_pos(), packages[i].get_pos()) + h(start.get_pos(), packages[i].get_pos())
        #print(calc_pit(start.get_pos(), packages[i].get_pos()))
        #print(h(start.get_pos(), packages[i].get_pos()))
        scores.append(score)
    for k in range(0, len(packages)):
        for l in range(0, len(packages)):
            if(l < len(packages) - 1):
                if(scores[k] < scores[l]):
                    temp1 = scores[k]
                    temp2 = points[k]
                    scores[k] = scores[l]
                    points[k] = points[l]
                    scores[l] = temp1
                    points[l] = temp2
    last = points[0]
    data.append('Pacote 1')
    data.append(points[0].get_pos())
    data.append(round(scores[0], 2))
    with open('robot_path.csv', 'a', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(data)
        #writer.close()
    #print(data)
    data = []
    points.pop(0)
    custom_path.append(last)
    #print(points)
    count = count + 1
    count_data = count_data + 1

    for i in range(0, len(packages)):
        #print(points)
        #print(count)
        if(last == end):
            print('Its end')
        for o in range(0, len(packages)):
            score = calc_pit(last.get_pos(), points[o].get_pos()) + h(last.get_pos(), points[o].get_pos())
            n_scores.append(score) 
        for k in range(0, len(packages)):
            for l in range(0, len(packages)):
                if(l < len(packages) - 1):
                    if(n_scores[k] < n_scores[l]):
                        temp1 = n_scores[k]
                        temp2 = points[k]
                        n_scores[k] = n_scores[l]
                        points[k] = points[l]
                        n_scores[l] = temp1
                        points[l] = temp2
        if(len(points) > 0):
            last = points[0]
            custom_path.append(points[0])
            data.append('Pacote ' + str(count_data))
            data.append(points[0].get_pos())
            data.append(round(n_scores[0], 2))
            with open('robot_path.csv', 'a', encoding='UTF8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(data)
            data = []
            count = count + 1
            count_data = count_data + 1
            
            points.pop(0)
            #print(points)
            #print(last)
            n_scores = []
            #print(count)
        if(count == 2 or len(points) == 0):
            if(custom_path[-1] == end):
                count == 0
            else:
                custom_path.append(end)
                count = 0
                data.append('Ponto de entrega')
                data.append(end.get_pos())
                data.append('---')
                with open('robot_path.csv', 'a', encoding='UTF8', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(data) 
                data = [] 
            if(len(points) > 0):
                #print(len(points))
                for o in range(0, len(packages)):
                    score = calc_pit(end.get_pos(), points[o].get_pos()) + h(end.get_pos(), points[o].get_pos())
                    n_scores.append(score) 
                for k in range(0, len(packages)):
                    for l in range(0, len(packages)):
                        if(l < len(packages) - 1):
                            if(n_scores[k] < n_scores[l]):
                                temp1 = n_scores[k]
                                temp2 = points[k]
                                n_scores[k] = n_scores[l]
                                points[k] = points[l]
                                n_scores[l] = temp1
                                points[l] = temp2
                last = points[0]
                custom_path.append(points[0])
                data.append('Pacote ' + str(count_data))
                data.append(points[0].get_pos())
                data.append(round(n_scores[0], 2))
                with open('robot_path.csv', 'a', encoding='UTF8', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(data)
                data = []
                points.pop(0)
                n_scores = []
                count = count + 1
                count_data = count_data + 1
                #print(count)
                    
            
    
    #print(scores)
    #print(custom_path)
    return points, custom_path


def reconstruct_path(came_from, current, draw):
    while current in came_from:
        current = came_from[current]
        current.make_path()
        draw()

#algoritmo que calcula caminho com menor custo para um dado inicio e fim
def algorithm(draw, grid, start, end):
    count = 0
    total_value = 0
    open_set = PriorityQueue()
    open_set.put((0, count, start))
    came_from = {}
    new_came_from = {}
    g_score = {spot: float("inf") for row in grid for spot in row}
    g_score[start] = 0
    f_score = {spot: float("inf") for row in grid for spot in row}
    f_score[start] = h(start.get_pos(), end.get_pos())	
    open_set_hash = {start}

    #print(start_packages)

    while not open_set.empty():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        current = open_set.get()[2]
        open_set_hash.remove(current)
        

        #print(read_packages)

        if (current == end ):
            reconstruct_path(came_from, end, draw)
            end.make_end()
            return True

        
        for neighbor in current.neighbors:
            temp_g_score = g_score[current] + 1

            if temp_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = temp_g_score
                f_score[neighbor] = temp_g_score + h(neighbor.get_pos(), end.get_pos())
                if neighbor not in open_set_hash:
                    count += 1
                    open_set.put((f_score[neighbor], count, neighbor))
                    open_set_hash.add(neighbor)
                    #neighbor.make_open()
                

        draw()

        #if current != start:
            #current.make_closed()

    return False


def make_grid(rows, width):
    grid = []
    gap = width // rows
    for i in range(rows):
        grid.append([])
        for j in range(rows):
            spot = Spot(i, j, gap, rows)
            grid[i].append(spot)

    return grid


def draw_grid(win, rows, width):
    gap = width // rows
    for i in range(rows):
        pygame.draw.line(win, GREY, (0, i * gap), (width, i * gap))
        for j in range(rows):
            pygame.draw.line(win, GREY, (j * gap, 0), (j * gap, width))


def draw(win, grid, rows, width):
    win.fill(WHITE)

    for row in grid:
        for spot in row:
            spot.draw(win)

    draw_grid(win, rows, width)
    pygame.display.update()


def get_clicked_pos(pos, rows, width):
    gap = width // rows
    y, x = pos

    row = y // gap
    col = x // gap

    return row, col


def main(win, width):
    ROWS = 50
    grid = make_grid(ROWS, width)

    start = None
    end = None
    #array para os pacotes
    packages = []

    with open('packages.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        csv_reader.__next__()
        for row in csv_reader:
            #print(row[0] + "," + row[1])
            x = int(row[0])
            y = int(row[1])
            package = grid[x][y]
            packages.append(package)
            package.make_package()

    run = True
    while run:
        draw(win, grid, ROWS, width)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

            if pygame.mouse.get_pressed()[0]: # LEFT
                pos = pygame.mouse.get_pos()
                row, col = get_clicked_pos(pos, ROWS, width)
                spot = grid[row][col]
                if not start and spot != end and spot != package:
                    start = spot
                    start.make_start()

                elif not end and spot != start and spot != package:
                    end = spot
                    end.make_end()
                
                elif spot != end and spot != start and spot != package:
                    spot.make_barrier()

            elif pygame.mouse.get_pressed()[2]: # RIGHT
                pos = pygame.mouse.get_pos()
                row, col = get_clicked_pos(pos, ROWS, width)
                spot = grid[row][col]
                spot.reset()
                if spot == start:
                    start = None
                elif spot == end:
                    end = None
                elif spot == package:
                    package = None

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and start and end and package:
                    for row in grid:
                        for spot in row:
                            spot.update_neighbors(grid)
                    #solucao mais proxima para calcular a melhor rota possivel
                    points, custom_path = calc_points(grid, start, end, packages)
                    #print(packages)
                    #df_csv = pd.read_csv('robot_path.csv', header=[0])
                    #print(df_csv)
                    for i in range(0, len(custom_path)):
                        j = i + 1
                        if(j < len(custom_path)):
                            algorithm(lambda: draw(win, grid, ROWS, width), grid, custom_path[i], custom_path[j])
                    #	j = i + 1
                    #	if(i == 0):
                    #		algorithm(lambda: draw(win, grid, ROWS, width), grid, start, points[0])
                    #		algorithm(lambda: draw(win, grid, ROWS, width), grid, points[0], points[1])
                    #	else:
                    #		if(j < len(packages) and i >= 1):
                    #			algorithm(lambda: draw(win, grid, ROWS, width), grid, points[i], points[j])
                    #fin_pack = len(packages) - 1
                    #algorithm(lambda: draw(win, grid, ROWS, width), grid, packages[fin_pack], end)
                    df_csv = pd.read_csv('robot_path.csv', header=[0])
                    print(df_csv) 

                    for i in range(0, len(packages)): 
                        packages[i].make_package()

                if event.key == pygame.K_c:
                    start = None
                    end = None
                    package = None
                    grid = make_grid(ROWS, width)

    pygame.quit()

main(WIN, WIDTH)