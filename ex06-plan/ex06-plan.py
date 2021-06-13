import gym
import copy
import random
import numpy as np
import matplotlib.pyplot as plt

class Node:
    def __init__(self, parent=None, action=None):
        self.parent = parent  # parent of this node
        self.action = action  # action leading from parent to this node
        self.children = []
        self.sum_value = 0.  # sum of values observed for this node, use sum_value/visits for the mean
        self.visits = 0


def rollout(env, maxsteps=100):
    """ Random policy for rollouts """
    G = 0
    for i in range(maxsteps):
        action = env.action_space.sample()
        _, reward, terminal, _ = env.step(action)
        G += reward
        if terminal:
            return G
    return G


def epsilon_greedy(epsilon, node):
    if np.random.rand() < epsilon: 
        #explore
        new_node = random.choice(node.children)
    else:
        #exploit
        values = [c.sum_value/(c.visits if c.visits > 0 else 1) for c in node.children]  # calculate values for child actions
        #if len(values) != 0:
        new_node = node.children[np.argmax(values)]  # select the best child
        #else:
            #new_node = random.choice(node.children)
    return new_node

def mcts(env, root, maxiter=500, epsilon=0.1):
    """ TODO: Use this function as a starting point for implementing Monte Carlo Tree Search
    """
    mod_print = 1000
    mean_return = np.zeros((500))
    tree_length = np.zeros((500))
    # this is an example of how to add nodes to the root for all possible actions:
    if not root.children:
        root.children = [Node(root, a) for a in range(env.action_space.n)]

    for i in range(maxiter):
        state = copy.deepcopy(env)
        G = 0.

        node = root

        #1. TODO: traverse the tree using an epsilon greedy tree policy
        # This is an example howto randomly choose a node and perform the action:
        # node = random.choice(root.children)
        # _, reward, terminal, _ = state.step(node.action)
        # G += reward
        while node.children:
            node = epsilon_greedy(epsilon, node)
            _, reward, terminal, _ = state.step(node.action)
            G += reward

        #2. TODO: Expansion of tree
        if not terminal:
            node.children = [Node(node, a) for a in range(env.action_space.n)]
            random.shuffle(node.children)

        #3. This performs a rollout (Simulation):
        if not terminal:
            G += rollout(state)

        #4. TODO: update all visited nodes in the tree
        # This updates values for the current node:
        while node:
            tree_length[i] += 1
            node.visits += 1
            node.sum_value += G
            node = node.parent
        
        mean_return[i] = G

    return mean_return, tree_length


def main():
    env = gym.make("Taxi-v3")
    env.seed(0)  # use seed to make results better comparable
    # run the algorithm 10 times:
    rewards = []
    mean_return = np.zeros((500))
    tree_length = np.zeros((500))

    mean_return_cnt = 0
    for i in range(10):
        env.reset()
        terminal = False
        root = Node()  # Initialize empty tree
        sum_reward = 0.
        while not terminal:
            env.render()
            mean_ret, tree_len = mcts(env, root)  # expand tree from root node using mcts
            mean_return+=mean_ret
            tree_length+=tree_len
            mean_return_cnt+= 1
            print("Mean return:\t{} | Tree length:\t{}".format(np.mean(mean_ret).round(2), np.mean(tree_len).round(2)))
            print("Avg. return:\t{} | Avg. length:\t{}".format(np.mean(mean_return/mean_return_cnt).round(2), np.mean(tree_length/mean_return_cnt).round(2)))
            values = [c.sum_value/(c.visits if c.visits > 0 else 1) for c in root.children]  # calculate values for child actions
            bestchild = root.children[np.argmax(values)]  # select the best child
            _, reward, terminal, _ = env.step(bestchild.action) # perform action for child
            root = bestchild  # use the best child as next root
            root.parent = None
            sum_reward += reward
        rewards.append(sum_reward)
        print("finished run " + str(i+1) + " with reward: " + str(sum_reward))
    print("mean reward: ", np.mean(rewards))
    plt.figure()
    plt.plot(mean_return/mean_return_cnt)
    plt.title("Mean return over episodes")
    plt.savefig("mean_return.png")
    plt.figure()
    plt.plot(tree_length/mean_return_cnt)
    plt.title("Tree length mean over episodes")
    plt.savefig("tree_length.png")
    plt.show()

if __name__ == "__main__":
    main()
