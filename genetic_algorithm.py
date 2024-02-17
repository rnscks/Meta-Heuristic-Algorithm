from abc import ABC, abstractmethod
import numpy as np
from multipledispatch import dispatch

class MetaHeuristic(ABC):
    def __init__(self) -> None:
        super().__init__()
    
    @abstractmethod
    def optimization(self):
        pass    


class GeneticAlgorithm(MetaHeuristic):
    def __init__(self,
                f: callable,
                pop_size: int = 10,
                dim: int = 2,
                ub: float = 10.0,
                lb: float = -10.0) -> None:
        super().__init__()
        self.pop_size: int = pop_size
        self.dim: int = dim
        self.ub: float = ub
        self.lb: float = lb
        self.f = f
        
        threshold = (self.ub + self.lb)/ 3.0
        self.pop: np.ndarray = self.population_generation(threshold)   
        fitness = self.evaluation() 
        fitness = fitness[:, np.newaxis]
        self.pop = np.hstack((self.pop, fitness))
        self.pop = self.pop[self.pop[:, -1].argsort()]
        self.pop[:, -1] = self.pop[:, -1] - self.pop[0, -1]
        self.pop[:, -1] = np.cumsum(self.pop[:, -1])
    

    def optimization(self):
        return
    
    def population_generation(self, threshold=0.5):
        initial_population = np.zeros((self.pop_size, self.dim))
        initial_population[0, :] = np.random.uniform(self.lb, self.ub, self.dim)    
        
        for row in range(1, self.pop_size):
            
            while True:
                population =  np.random.uniform(self.lb, self.ub, self.dim)
                differences = initial_population[:row, :] - population[np.newaxis, :]
                distances = np.linalg.norm(differences, axis=1)
                
                if np.all(distances > threshold):
                    initial_population[row, :] = population
                    break
                
        return initial_population   

    @dispatch()
    def evaluation(self):
        return np.apply_along_axis(func1d=self.f, axis=1, arr=self.pop[:, :self.dim])    
    
    @dispatch(np.ndarray)
    def evaluation(self, pop: np.ndarray):
        return self.f(pop)
    
if __name__ == "__main__":
    def f(x: np.ndarray) -> float:
        return np.sum(x**2)
    
    ga = GeneticAlgorithm(f = f)
