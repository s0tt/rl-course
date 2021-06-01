import gym
import numpy as np

# Init environment
env = gym.make("FrozenLake-v0")
# you can set it to deterministic with:
# env = gym.make("FrozenLake-v0", is_slippery=False)

# If you want to try larger maps you can do this using:
#random_map = gym.envs.toy_text.frozen_lake.generate_random_map(size=5, p=0.8)
#env = gym.make("FrozenLake-v0", desc=random_map)


# Init some useful variables:
n_states = env.observation_space.n
n_actions = env.action_space.n

def func_v(state, gamma):
    v_max = -1
    for action in range(n_actions):
        for tup in env.P[state][action]:
            v = tup[0]*(tup[2] + gamma*func_v(tup[1], gamma))
            if v >= v_max:
                v_max = v
    return v_max


def value_iteration():
    V_states = np.zeros(n_states)  # init values as zero
    theta = 1e-8
    gamma = 0.8
    delta = 1
    # TODO: implement the value iteration algorithm and return the policy
    # Hint: env.P[state][action] gives you tuples (p, n_state, r, is_terminal), which tell you the probability p that you end up in the next state n_state and receive reward r
    policy = np.zeros(n_states, dtype=int)
    steps = 0
    while delta > theta:
        steps+=1
        delta = 0
        for state in range(n_states):
            v = V_states[state]
            v_action = np.zeros(n_actions)
            for action in range(n_actions):
                for tup in env.P[state][action]:
                    v_action[action] += tup[0]*(tup[2] + gamma*V_states[tup[1]])
            policy[state] = np.argmax(v_action).astype(int)
            V_states[state] = np.amax(v_action)
            delta = max(delta, np.abs(v - V_states[state]))
    print("Steps: ", steps)
    print("Optimal value: ", V_states)
    return policy
    

def main():
    # print the environment
    print("current environment: ")
    env.render()
    print("")

    # run the value iteration
    policy = value_iteration()
    print("Computed policy:")
    print(policy)

    # This code can be used to "rollout" a policy in the environment:
    print ("rollout policy:")
    maxiter = 100
    state = env.reset()
    for i in range(maxiter):
        new_state, reward, done, info = env.step(policy[state])
        env.render()
        state=new_state
        if done:
            print ("Finished episode")
            break


if __name__ == "__main__":
    main()
