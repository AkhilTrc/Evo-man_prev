# -- coding: utf-8 --
"""
Created on Fri Sep 11 12:00:29 2020

@author: Ayuub Hussein
"""
# Import evoman framework

import sys, os
sys.path.insert(0, 'evoman') 
from environment import Environment
from demo_controller import player_controller
import numpy as np
import glob
from deap import base, creator, tools, algorithms
import operator, random
import math


experiment_name = 'algo_v_test_builtin' #naming convention
#tuan enemy 4 , fajjaz enemy 3 , ayuub enemy 2
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)
    
env = Environment(experiment_name=experiment_name,
                  enemies=[3],
				  playermode="ai",
				  player_controller=player_controller(100),
		  		  speed="fastest",
				  enemymode="static",
				  level=2)

run_mode = 'test'

# Standard variables
pop_size = 100 # if it works it works
total_generations = 5
n_weights = (env.get_num_sensors()+1)*100 + (100+1)*5
upperbound = 1
lowerbound = -1

sigma = 1
tau = math.pow(np.sqrt(n_weights),-1)
mut_prob = 0.001

average_pops = []
std_pops = []
best_per_gen = []
player_means = []

best_overall = 0
noimprove = 0

log = tools.Logbook()
tlbx = base.Toolbox()
env.state_to_log()



# Register and create deap functions and classes
def register_deap_functions():
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", np.ndarray, fitness=creator.FitnessMax, lifepoints=1)

    tlbx.register("atrr_float", np.random.uniform, low=lowerbound, high=upperbound)
    tlbx.register("individual", tools.initRepeat, creator.Individual, tlbx.atrr_float, n=n_weights)
    tlbx.register("population", tools.initRepeat, list, tlbx.individual, n=pop_size)
    
    tlbx.register("evaluate", evaluate)
    tlbx.register("mate", tools.cxTwoPoint)
    tlbx.register("mutate", tools.mutFlipBit, indpb=0.05)
    tlbx.register("select", tools.selTournament, tournsize=3)

# Evaluate individual
def evaluate(individual):
    a,b,c,d = env.play(pcont=individual)
    #individual.lifepoints = b
    return a,

def simulation(env,x):
    f,p,e,t = env.play(pcont=x)
    return f


def eval2(x):
    return np.array(list(map(lambda y: simulation(env,y), x)))


register_deap_functions()

def results(pop_fit, player_life):
    best = np.argmax(pop_fit)
    std = np.std(pop_fit)
    mean = np.mean(pop_fit)
    mean_life = np.mean(player_life)

    player_means.append(mean_life)
    average_pops.append(mean)
    std_pops.append(std)
    best_per_gen.append(pop_fit[best])
    return best, std, mean

def offspring_mutation(offspring):
    for mutant in offspring:
        if  random.random() < mut_prob:
            tlbx.mutate(mutant)
            del mutant.fitness.values



if run_mode =='train':
    for n_sim in range(3):
    
        if not os.path.exists(experiment_name+'/sim {}'.format(n_sim+1)):
            os.makedirs(experiment_name+'/sim {}'.format(n_sim+1))
        print("-------------Simulation {}-------------------".format(n_sim+1))
        # initializes population at random
        pop = tlbx.population()
        pop_fit = [tlbx.evaluate(ind) for ind in pop]
    
        for ind, fit in zip(pop, pop_fit):
            ind.fitness.values = fit
    
        pop_fit = [ind.fitness.values[0] for ind in pop]
        player_life = [ind.lifepoints for ind in pop]
    
        best, std, mean = results(pop_fit, player_life)
    
        file_aux  = open(experiment_name+'/results.txt','a')
        file_aux.write('\n\ngen best mean std')
        print( '\n GENERATION '+str(n_sim)+' '+str(round(pop_fit[best],6))+' '+str(round(mean,6))+' '+str(round(std,6)))
        file_aux.write('\n'+str(n_sim)+' '+str(round(pop_fit[best],6))+' '+str(round(mean,6))+' '+str(round(std,6))   )
        file_aux.close()
    
        for n_gen in range(total_generations):
            print("------------Generation {}-------------".format(n_gen + 1))
    
            offspring = tlbx.select(pop, len(pop))
            offspring = list(map(tlbx.clone, offspring))
                    
            # Apply crossover on the offspring
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < 0.5:
                    tlbx.mate(child1 ,child2)
                    del child1.fitness.values
                    del child2.fitness.values
    
    
            # Apply mutation on the offspring
            offspring_mutation(offspring)
    
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(tlbx.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
                
            pop[:] = offspring
            
            pop_fit = [ind.fitness.values[0] for ind in pop]
            player_life = [ind.lifepoints for ind in pop]
    
    
            best, std, mean = results(pop_fit, player_life)
            
            # saves results
            file_aux  = open(experiment_name+'/results.txt','a')
            print( '\n GENERATION '+str(total_generations)+' '+str(round(pop_fit[best],6))+' '+str(round(mean,6))+' '+str(round(std,6)))
            file_aux.write('\n'+str(total_generations)+' '+str(round(pop_fit[best],6))+' '+str(round(mean,6))+' '+str(round(std,6))   )
            file_aux.close()
        
            # saves generation number
            file_aux  = open(experiment_name+'/gen.txt','w')
            file_aux.write(str(total_generations))
            file_aux.close()
        
            # saves file with the best solution
            np.savetxt(experiment_name+'/best.txt',pop[best])
    
    
            print("Pop:", pop_fit)
        
        
        
# loads file with the best solution for testing
if run_mode =='test':
    print( '\n RUNNING SAVED BEST SOLUTION \n')
    env.update_parameter('speed','fastest')
    for i in range(5):
        bsol = np.loadtxt(experiment_name+'/best.txt')
        eval2([bsol])
    sys.exit(0)
