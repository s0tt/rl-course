import gym
import numpy as np
import matplotlib.pyplot as plt


def policy(state, theta):
    """ TODO: return probabilities for actions under softmax action selection """
    h = state @ theta
    return np.exp(h)/np.sum(np.exp(h))


def generate_episode(env, theta, display=False):
    """ enerates one episode and returns the list of states, the list of rewards and the list of actions of that episode """
    state = env.reset()
    states = [state]
    actions = []
    rewards = []
    for t in range(500):
        if display:
            env.render()
        p = policy(state, theta)
        action = np.random.choice(len(p), p=p)

        state, reward, done, info = env.step(action)
        rewards.append(reward)
        actions.append(action)
        if done:
            break
        states.append(state)

    return states, rewards, actions


def REINFORCE(env, gamma=0.99, alpha=0.05):
    theta = np.random.rand(4, 2)  # policy parameters
    ep_len_list = []
    mean_ep_len = []

    for e in range(1000):
        if e % 300 == 0:
            states, rewards, actions = generate_episode(env, theta, False)  # display the policy every 300 episodes
        else:
            states, rewards, actions = generate_episode(env, theta, False)

        
        # TODO: keep track of previous 100 episode lengths and compute mean
        if len(ep_len_list) >= 100:
            ep_len_list.pop(0) #remove last item
        ep_len_list.append(len(states))
        mean = sum(ep_len_list) / len(ep_len_list)
        mean_ep_len.append(mean)

        print("episode:\t" + str(e) + " length:\t" + str(len(states)) + " mean len:\t" + str(mean))

        # TODO: implement the reinforce algorithm to improve the policy weights
        nr_steps = len(states)
        G = np.zeros([nr_steps])
        for t in range(nr_steps):
            for k in range(t+1,nr_steps+1):
                G[t] += (gamma**(k-t-1)) * rewards[k-1]
            action = actions[t]
            theta[:,action] = theta[:,action] + alpha * (gamma**t) * G[t] * (states[t] * (1 - policy(states[t], theta)[action]))
    return mean_ep_len




def main():
    env = gym.make('CartPole-v1')
    mean_ep_len = REINFORCE(env)
    plt.plot(mean_ep_len)
    plt.title("Mean Ep length over time")
    plt.xlabel("Episodes")
    plt.ylabel("Mean Episode Length")
    plt.legend()
    plt.savefig('ex09' + '.png')
    plt.show()
    env.close()


if __name__ == "__main__":
    main()
