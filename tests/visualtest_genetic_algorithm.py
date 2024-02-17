import numpy as np
import matplotlib.pyplot as plt

def display_population_generation() -> None: 
    def population_generation(population_size, dim, threshold=0.5):
        initial_population = np.zeros((population_size, dim))
        initial_population[0, :] = np.random.uniform(-1, 1, dim)    
        
        for row in range(1, population_size):
            while True:
                population =  np.random.uniform(-1, 1, dim) 
                differences = initial_population[:row, :] - population[np.newaxis, :]
                distances = np.linalg.norm(differences, axis=1)
                if np.all(distances > threshold):
                    initial_population[row, :] = population
                    break   
        return initial_population   

    population_size = 10
    dim = 2
    threshold = 0.5 
    initial_population = population_generation(population_size, dim, threshold)    

    # 반지름
    radius = threshold
    # 그래프 생성
    plt.figure(figsize=(10, 10))

    # 각 좌표에 원 그리기
    for x, y in initial_population:
        circle = plt.Circle((x, y), radius, color='red', fill=False)
        plt.gca().add_patch(circle)

    # 점 표시
    plt.scatter(initial_population[:, 0], initial_population[:, 1], color='blue')

    # 축 설정
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title('Points with Circular Areas')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()
    return

display_population_generation()