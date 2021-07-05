import gym
import numpy as np
import matplotlib.pyplot as plt
import random

x_t_space = np.linspace(-1.2, 0.6, 25)
x_t_dot_space = np.linspace(-0.07, 0.07, 25)

def random_episode(env):
    """ This is an example performing random actions on the environment"""
    while True:
        env.render()
        action = env.action_space.sample()
        print("do action: ", action)
        observation, reward, done, info = env.step(action)
        print("observation: ", observation)
        print("reward: ", reward)
        print("")
        if done:
            break

def convDiscrete(observation,bins):
    return np.digitize(observation, bins)

    #x_t = int(observation[0]*100) +12 #car state [-1.2, 0.6] -> [0, 18]
    #x_t_dot = int(observation[0]*100) +7 #car state [-0.07,0.07]-> [0, 14]

def epsilon_greedy(Q, env, idx, epsilon):
    if random.random() < epsilon: 
        #explore
        a = random.randint(0, env.action_space.n-1)
    else:
        #exploit
        #calc env state to discrete space
        a = np.argmax(Q[idx[0],idx[1], :])
    return a


def qlearning(env, alpha=0.1, gamma=0.9, epsilon=1, num_ep=int(5e3), rendering=True):
    Q = np.random.rand(len(x_t_space), len(x_t_dot_space),  env.action_space.n)
    #Q = np.zeros((len(x_t_space), len(x_t_dot_space), env.action_space.n))

    # TODO: implement the qlearning algorithm
    nr_goals = 0
    ep_steps_list = []
    goal_list = []
    isRenderOn = False
    for i in range(num_ep):
        if rendering:
            if i > num_ep-5:
                isRenderOn = True
            else:
                isRenderOn = False

        
        if i % 100 == 0:
            print("Ep. Nr. {}".format(i))
            print("Goals: {}/100 | epsilon decay = {}".format(nr_goals,epsilon))
            goal_list.append(nr_goals)
            nr_goals = 0


        s = env.reset()
        done = False
        a = 2
        ep_steps = 0
        while not done:
            if isRenderOn:
                env.render()
            s_, r, done, _ = env.step(a)
            xt_idx = np.digitize(s[0],x_t_space)
            xtd_idx = np.digitize(s[1],x_t_dot_space)
            xt_idx_ = np.digitize(s_[0],x_t_space)
            xtd_idx_ = np.digitize(s_[1],x_t_dot_space)

            a_ = epsilon_greedy(Q,env, [xt_idx_, xtd_idx_],epsilon)
            a_maximize = np.argmax(Q[xt_idx,xtd_idx,:])

            Q[xt_idx, xtd_idx, a] = Q[xt_idx, xtd_idx, a] + alpha*(r + gamma*Q[xt_idx_,xtd_idx_,a_maximize] - Q[xt_idx, xtd_idx, a])
            
            s = s_
            a = a_
            ep_steps+=1

        if done and ep_steps <200:
            nr_goals +=1

        epsilon -= 0.00055* (((num_ep-i)/num_ep)**2)
        if epsilon < 0:
            epsilon = 0

        ep_steps_list.append(ep_steps)

    return Q, ep_steps_list, goal_list

def sarsa(env, alpha=0.1, gamma=0.9, epsilon=1, num_ep=int(5e3), rendering=True):
    Q = np.random.rand(len(x_t_space), len(x_t_dot_space),  env.action_space.n)
    #Q = np.zeros((len(x_t_space), len(x_t_dot_space), env.action_space.n))
    w = np.random.rand(len(x_t_space), len(x_t_dot_space),  env.action_space.n)
    goal_list = []
    ep_steps_list = []
    isRenderOn = False
    nr_goals = 0
    for i in range(num_ep):
        if rendering:
            if i > num_ep-5:
                isRenderOn = True
            else:
                isRenderOn = False

        
        if i % 100 == 0:
            print("Ep. Nr. {}".format(i))
            print("Goals: {}/100 | epsilon decay = {}".format(nr_goals,epsilon))
            goal_list.append(nr_goals)
            nr_goals = 0



        s = env.reset()
        done = False
        ep_steps = 0
        #a = np.random.randint(env.action_space.n)
        #s, r, done, _ = env.step(a)
        xt_idx = np.digitize(s[0],x_t_space)
        xtd_idx = np.digitize(s[1],x_t_dot_space)

        a = epsilon_greedy(Q, env, [xt_idx,xtd_idx], epsilon)
        while not done:
            s_, r, done, _ = env.step(a)
            xt_idx = np.digitize(s[0],x_t_space)
            xtd_idx = np.digitize(s[1],x_t_dot_space)
            xt_idx_ = np.digitize(s_[0],x_t_space)
            xtd_idx_ = np.digitize(s_[1],x_t_dot_space)
            a_ = epsilon_greedy(Q, env, [xt_idx_,xtd_idx_], epsilon)
            #s_, r, done, _ = env.step(a_)
            #Q[s,a] = Q[s,a] + alpha*(r + gamma*Q[s_,a_] - Q[s,a])
            Q[xt_idx, xtd_idx, a] = Q[xt_idx, xtd_idx, a] + alpha*(r + gamma*Q[xt_idx_,xtd_idx_,a_] - Q[xt_idx, xtd_idx, a])
            s = s_
            a = a_
            ep_steps +=1

        if done and ep_steps <200:
            nr_goals +=1

        epsilon -= 0.00055* (((num_ep-i)/num_ep)**2)
        if epsilon < 0:
            epsilon = 0

        ep_steps_list.append(ep_steps)
    
    return Q, ep_steps_list, goal_list

def Q_w(w, idx, a):
    #return w[0,0,:] + w[idx[0],idx[1], :] * np.array([x_t_space[idx[0]], x_t_dot_space[idx[1]], np.array([0,1,2])])
    return w[0,0,0:2] + w[idx[0],idx[1], 0:2] @ np.array([x_t_space[idx[0]], x_t_dot_space[idx[1]]])

def sarsa_lfa(env, alpha=0.1, gamma=0.9, epsilon=1, num_ep=int(5e3), rendering=True):
    Q = np.random.rand(len(x_t_space), len(x_t_dot_space),  env.action_space.n)
    #Q = np.zeros((len(x_t_space), len(x_t_dot_space), env.action_space.n))

    #

    # init weights
    w = np.random.rand(len(x_t_space)+1, len(x_t_dot_space)+1,  env.action_space.n)
    w[0,0,:] = [1,1,1]
    goal_list = []
    ep_steps_list = []
    isRenderOn = False
    nr_goals = 0
    for i in range(num_ep):
        if rendering:
            if i > num_ep-5:
                isRenderOn = True
            else:
                isRenderOn = False

        
        if i % 100 == 0:
            print("Ep. Nr. {}".format(i))
            print("Goals: {}/100 | epsilon decay = {}".format(nr_goals,epsilon))
            goal_list.append(nr_goals)
            nr_goals = 0



        s = env.reset()
        done = False
        ep_steps = 0
        #a = np.random.randint(env.action_space.n)
        #s, r, done, _ = env.step(a)
        xt_idx = np.digitize(s[0],x_t_space)
        xtd_idx = np.digitize(s[1],x_t_dot_space)

        a = epsilon_greedy(Q, env, [xt_idx,xtd_idx], epsilon)
        while not done:
            s_, r, done, _ = env.step(a)
            xt_idx = np.digitize(s[0],x_t_space)
            xtd_idx = np.digitize(s[1],x_t_dot_space)
            xt_idx_ = np.digitize(s_[0],x_t_space)
            xtd_idx_ = np.digitize(s_[1],x_t_dot_space)
            a_ = epsilon_greedy(Q, env, [xt_idx_,xtd_idx_], epsilon)
            #s_, r, done, _ = env.step(a_)
            #Q[s,a] = Q[s,a] + alpha*(r + gamma*Q[s_,a_] - Q[s,a])
            delta = r + gamma*Q_w(w,[xt_idx_,xtd_idx_], a) - Q_w(w, [xt_idx,xtd_idx], a)
            #Q[xt_idx, xtd_idx, a] = Q[xt_idx, xtd_idx, a] + alpha*(r + gamma*Q[xt_idx_,xtd_idx_,a_] - Q[xt_idx, xtd_idx, a])
            
            # update w
            for i in range(len(x_t_space)+1):
                for j in range(len(x_t_dot_space)+1):
                    if i==xt_idx and j ==xtd_idx:
                        w[i,j,0:2]  = w[i,j,0:2] + alpha * delta * np.array([x_t_space[xt_idx], x_t_dot_space[xtd_idx]])
                    else:
                        w[i,j,:]  = w[i,j,:]   #w[idx[0],idx[1], :] * [x_t_space[idx[0]], x_t_space[idx[1]], [0,1,2]]

            s = s_
            a = a_
            ep_steps +=1

        if done and ep_steps <200:
            nr_goals +=1

        epsilon -= 0.00055* (((num_ep-i)/num_ep)**2)
        if epsilon < 0:
            epsilon = 0

        ep_steps_list.append(ep_steps)
    
    return Q, ep_steps_list, goal_list

def plotData(data, name):
    plt.figure()
    plt.plot(data)
    plt.title(name)
    plt.savefig(name +".png")

def main():
    num_ep = int(5e3)
    nr_overall_steps = 10
    env = gym.make('MountainCar-v0')
    env.reset()
    #random_episode(env)
    # overall_steps = np.zeros((num_ep))
    # overall_goals = np.zeros((num_ep))
    # for i in range(nr_overall_steps):
    #     _, step_list, goal_list = qlearning(env, num_ep=num_ep, rendering=False)
    #     overall_steps += np.array(step_list)
    #     overall_goals += np.array(np.repeat(goal_list, 100))

    # plotData(overall_steps/nr_overall_steps, "QLearning Mean steps over episodes")
    # plotData(overall_goals/nr_overall_steps, "QLearning Mean goals over episodes")

    # overall_steps = np.zeros((num_ep))
    # overall_goals = np.zeros((num_ep))
    # for i in range(nr_overall_steps):
    #     _, step_list, goal_list = sarsa(env, num_ep=num_ep, rendering=False)
    #     overall_steps += np.array(step_list)
    #     overall_goals += np.array(np.repeat(goal_list, 100))

    # plotData(overall_steps/nr_overall_steps, "Sarsa Mean steps over episodes")
    # plotData(overall_goals/nr_overall_steps, "Sarsa Mean goals over episodes")
    sarsa_lfa(env, num_ep=num_ep, rendering=True)
    env.close()


if __name__ == "__main__":
    main()
