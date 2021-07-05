import gym
import numpy as np
import matplotlib.pyplot as plt
import sys

def epsilon_greedy(Q, s, env, epsilon):
    if np.random.rand() < epsilon: 
        #explore
        a = np.random.randint(env.action_space.n)
    else:
        #exploit
        a = np.random.choice(np.argwhere(Q[s,:]==np.max(Q[s,:])).reshape(-1))
    return a

def mc(env, num_ep, epsilon=0.1):
    returns = np.zeros((env.observation_space.n,  env.action_space.n))
    Q = np.zeros((env.observation_space.n,  env.action_space.n))  
    # loop over episodes
    for i_episode in range(1, num_ep + 1):
        episode = []
        s = env.reset()
        done = False
        while not done:
            a = epsilon_greedy(Q, s, env, epsilon)
            s_, r, done, _ = env.step(a)
            episode.append((s, a, r))
            s = s_
        N = np.zeros((env.observation_space.n,  env.action_space.n))
        states, actions, rewards = zip(*episode)
        for i, state in enumerate(states):
            N[state, actions[i]] += 1
            if N[state, actions[i]] == 1:
                returns[state, actions[i]] += np.sum(rewards[i:])
                Q[state, actions[i]] = np.mean(returns[state, actions[i]])
    return Q

def nstep_sarsa(env, n=1, alpha=0.1, gamma=0.9, epsilon=0.1, num_ep=int(1e4)):
    Q = np.zeros((env.observation_space.n,  env.action_space.n))
    len_list = []
    avg_len_list = []
    
    for i in range(num_ep):
        
        done = False
        len = 0
        rewards = [0]
        states = []
        actions = []
        T = int(99999999)
        t = 0
        tau = 0
        s = env.reset()
        states.append(s)
        actions.append(epsilon_greedy(Q, s, env, epsilon))
        while tau != T-1:
            if t < T: #if not over termination state
                s, r, done, _ = env.step(actions[t])
                states.append(s)
                rewards.append(r)
                if done:
                    T = t+1
                else:
                    actions.append(epsilon_greedy(Q, s, env, epsilon))
            
            tau = t-n+1
            if tau >= 0: # use n-step if possible
                G = 0
                for i in range(int(tau+1), np.minimum(tau+n, T) +1):
                    G += gamma**(i-tau-1) * rewards[i]
                if tau+n < T:
                    G = G + gamma**n * Q[states[tau+n],actions[tau+n]]
                
                Q[states[tau],actions[tau]] += alpha*(G - Q[states[tau],actions[tau]])
            t = t+1
        len_list.append(len)
        avg_len_list.append(np.mean(len_list))
    
    #plt.plot(avg_len_list)
    #plt.show()
    return Q


env=gym.make('FrozenLake-v0', map_name="8x8")
# TODO: run multiple times, evaluate the performance for different n and alpha
#nstep_sarsa(env)


Q_std = mc(env, num_ep=10000)
print("MC Finished: ", Q_std)
n_idx = 0
n_test = np.array([1, 5, 10, 25, 50, 75, 100])
alpha_test = np.arange(0.25, 0.85, 0.1)
RMS = np.zeros((len(n_test), len(alpha_test)))
for n in n_test:
    a_idx = 0
    for alpha in alpha_test:
        print("Run Sarsa with n={}, a={}".format(n,alpha))
        Q_hat = nstep_sarsa(env, n=n, alpha=alpha, num_ep=1000)
        print("Sarsa Finished: ", Q_hat)
        RMS[n_idx,a_idx] = np.sqrt(((Q_hat - Q_std) ** 2).mean())
        a_idx += 1
    n_idx +=1

for n in range(len(n_test)):
    plt.plot(alpha_test, RMS[n,:], label='n=' + str(n_test[n]))
plt.title("res")
plt.xlabel("alpha")
plt.ylabel("Root mean squared error")
plt.legend()
plt.savefig('ex08' + '.png')
plt.show()

