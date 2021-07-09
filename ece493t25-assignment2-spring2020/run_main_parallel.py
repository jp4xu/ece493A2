from maze_no_gui import Maze
from RL_brainsample_hacky_PI import rlalgorithm as rlalg1
import numpy as np
import sys
import matplotlib.pyplot as plt
from collections import deque
import time

from RL_brainsample_qlearning import rlalgorithm as rlalg2
from RL_brainsample_doubqlearning import rlalgorithm as rlalg3
# from RL_brainsample_sarsa import rlalgorithm as rlalg2
import multiprocessing

#Example Short Fast for Debugging
# showRender=True
# renderEveryNth=10
do_plot_rewards=True
episodes=2000
# printEveryNth=1

#Example Full Run, you may need to run longer
showRender=False
# episodes=2000
renderEveryNth=10000
printEveryNth=10000
# do_plot_rewards=True

DEBUG=0
def debug(debuglevel, msg, **kwargs):
    if debuglevel <= DEBUG:
        if 'printNow' in kwargs:
            if kwargs['printNow']:
                print(msg)
        else:
            print(msg)


def plot_rewards(experiments, save = False, title = None):
    color_list=['blue','green','red','black','magenta']
    label_list=[]
    plt.clf()
    plt.figure(figsize=[12,9])
    for i, (name, data) in enumerate(experiments):
        x_values=range(len(data['global_reward']))
        label_list.append(name)
        y_values=data['global_reward']
        plt.plot(x_values, y_values,label=label_list[-1])
        plt.legend(label_list)
    title_suffix =  ": " + str(title) if title else ""
    plt.title("Reward Progress" + title_suffix, fontsize=18)
    plt.xlabel("Episode", fontsize=18)
    plt.ylabel("Return", fontsize=18)
    plt.tick_params(axis='both', which='major',
                    labelsize=14)
    if save:
        fname = str(title) if title else '_'.join(label_list)
        print(fname)
        plt.savefig(fname, dpi = 200)
    else:
        plt.show()

def update(env, RL, data, episodes=50):
    global_reward = np.zeros(episodes)
    data['global_reward']=global_reward

    for episode in range(episodes):
        t=0
        # initial state
        if episode == 0:
            state = env.reset(value = 0)
        else:
            state = env.reset()

        debug(2,'state(ep:{},t:{})={}'.format(episode, t, state))

        # RL choose action based on state
        action = RL.choose_action(str(state))
        while True:
            # fresh env
            # RL take action and get next state and reward
            state_, reward, done = env.step(action)
            global_reward[episode] += reward


            # RL learn from this transition
            # and determine next state and action
            state, action =  RL.learn(str(state), action, reward, str(state_))

            # break while loop when end of this episode
            if done:
                break
            else:
                t=t+1

    env.destroy()

def multi_wrap(env, RL, episodes, out_q, exp_name, task_name, algo_ind=0):
    """
    out_q is an instance of multiprocessing.Queue
    """
    data = {}

    start_time = time.time()
    update(env, RL, data, episodes)
    time_taken = time.time() - start_time


    out_q.put({"name": exp_name, "data": data, "ind": algo_ind})
    out_string1 = "Task {} Experiment {} completed in {} seconds".format(task_name, exp_name, round(time_taken, 4))
    out_string2 = "max reward = {} medLast100={} varLast100={}".format(
        np.max(data['global_reward']),
        np.median(data['global_reward'][-100:]), 
        np.var(data['global_reward'][-100:])
    )
    print(out_string1,'\n',out_string2,'\n\n')

if __name__ == "__main__":
    sim_speed = 0.01

    ref_q = multiprocessing.Queue()
    exp_start_time = time.time()
    DO_HACKY = False

    if(len(sys.argv)>1):
        episodes = int(sys.argv[1])
    if(len(sys.argv)>2):
        showRender = sys.argv[2] in ['true','True','T','t']
    if(len(sys.argv)>3):
        datafile = sys.argv[3]


    # Task Specifications
    agentXY=[0,0]
    tasks = [
        # Task 1 - E
        {
            "name": '1',
            "goal": [4,5],
            "walls": np.array([[2,2],[3, 2], [4, 2], [5, 2], 
                         [2,3], [2, 4], [2,5], [2,6], [2, 7],
                         [2,8],[3, 8], [4, 8], [5, 8]]),
            "pits": np.array([[6,3],[1,4]])
        }
        ,
        # Task 2 - C
        {
            "name": '2',
            "goal": [2,5],
            "walls": np.array([[2,2],[3, 2], [4, 2], 
                         [2,3], [2, 4], [2,6], [2, 7],
                         [3, 8], [4, 8], [5, 8]]),
            "pits": np.array([[5,2],[2,8]])
        }
        ,
        # Task 3 - E
        {
            "name": '3',
            "goal": [4,5],
            "walls": np.array([[2,2],[3, 2], [4, 2], [5, 2], 
                         [2,3], [2, 4], [2,5], [2,6], [2, 7],
                         [2,8],[3, 8], [4, 8], [5, 8]]),
            "pits": np.array([[4,4],[4,6], [5,5], [5, 4], [5,6]])
        }
    ]

    # RL_algorithms
    algos = [
        rlalg2,
        rlalg3
    ]

    # Params
    initial_vals = [ 0]
    learning_rates = [ 0.1, 0.2, 0.5, 0.7]
    epsilons = [0.1]
    decays = [ 0.8, 0.9, 0.99]

    num_charts = len(tasks)
    num_algos = len(algos)
    all_qs = [multiprocessing.Queue() for _ in range(num_charts) ]
    all_experiments = []
    algo_names = [algo(actions=[0]).display_name for algo in algos]
    exp_num = 0
    
    exp_name_template =  "(v={},a={},g={},e={})"
    seen_ref = num_charts if DO_HACKY else 0
    hacky_inds = {}


    
    for i, task in enumerate(tasks):
        goalXY = task["goal"]
        wall_shape = task["walls"]
        pits = task["pits"]

        env1 = Maze(agentXY,goalXY,wall_shape, pits)
        possible_actions = list(range(len(env1.action_space)))

        # Reference
        if DO_HACKY:
            hacky_name = "Hacky_RL_{}".format(i)
            hacky_inds[hacky_name] = i
            ref_process = multiprocessing.Process(
                target=multi_wrap, 
                args=(
                    Maze(agentXY,goalXY,wall_shape, pits), 
                    rlalg1(actions=possible_actions), 
                    episodes, ref_q, hacky_name, task["name"]
                )
            )
            ref_process.start()

        # Values
        for j, algo in enumerate(algos):
            ex_class = algo(actions=[0])
            ex_display_name = ex_class.display_name
            for val in initial_vals:
                for lr in learning_rates:
                    for e in epsilons:
                        for gam in decays:
                            exp_num += 1
                            name = exp_name_template.format(val,lr,gam,e)
                            p = multiprocessing.Process (
                                target=multi_wrap,
                                args = (
                                    Maze(agentXY,goalXY,wall_shape,pits),   # Environment
                                    algo(
                                        actions=possible_actions,
                                        learning_rate=lr,
                                        reward_decay=gam,
                                        e_greedy=e,
                                        init_val=val,
                                        name=name
                                    ),
                                    episodes, all_qs[i], ex_display_name + name, task["name"], j
                                )
                            )
                            p.start()

    
            all_experiments.append( [deque() for _ in range(num_charts)])

    print("Num experiments:", exp_num,'\n\n')

    # Busy waiting because deadlocks occur if multiprocessing queues fill up
    # and I don't want to write consumer functions
    while exp_num > 0:
        for i in range(num_charts):
            while not all_qs[i].empty():
                exp_num -= 1
                out_dict = all_qs[i].get()
                j = out_dict["ind"]
                all_experiments[j][i].append((out_dict["name"], out_dict["data"]))
        
        if not ref_q.empty():
            seen_ref -= 1
            ref_dict = ref_q.get()
            ref_tup = (ref_dict["name"], ref_dict["data"])
            h_ind = hacky_inds[ref_dict["name"]]

            all_experiments[ref_dict["ind"]][-h_ind].appendleft((ref_tup))
    # More busy waiting
    while seen_ref:
        if not ref_q.empty():
            seen_ref -= 1
            ref_dict = ref_q.get()
            ref_tup = (ref_dict["name"], ref_dict["data"])
            h_ind = hacky_inds[ref_dict["name"]]

            all_experiments[ref_dict["ind"]][-h_ind].appendleft((ref_tup))
    
    total_time = time.time() - exp_start_time

    if(do_plot_rewards):
        #Simple plot of return for each episode and algorithm, you can make more informative plots
        for i in range(num_charts):
            for j in range(num_algos):
                plot_rewards(all_experiments[j][i], save=True, title="Task_{}_{}".format(tasks[i]["name"], algo_names[j]))

    print("\nAll experiments complete in {} s".format(total_time))

    #Not implemented yet
    #if(do_save_data):
    #    for env, RL, data in experiments:
    #        saveData(env,RL,data)

