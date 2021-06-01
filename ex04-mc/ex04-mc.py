import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('Blackjack-v0')

def mc_prediction(policy="sum20"):
    # This example shows how to perform a single run with the policy that hits for player_sum >= 20

    obs = env.reset()  # obs is a tuple: (player_sum, dealer_card, useable_ace)
    done = False
    v_val = np.zeros((2,10,10,2)) #Value for (action, player_sum, dealer_card, useable_ace)
    counts = np.zeros((2,10,10,2)) #counter for (action, player_sum, dealer_card, useable_ace)

    while not done:
        #print("observation:", obs)
        player_sum_idx = obs[0]-12
        dealer_card_idx = obs[1]-1
        ace_idx = 1 if obs[2] else 0

        action_taken = None
        if policy == "sum20":
            if obs[0] >= 20:
                #print("stick")
                action_taken = 0
                obs, reward, done, _ = env.step(0)
            else:
                #print("hit")
                action_taken = 1
                obs, reward, done, _ = env.step(1)
        else:
            action_taken = policy[player_sum_idx, dealer_card_idx, ace_idx]
            obs, reward, done, _ = env.step(action_taken)
        #print("reward:", reward)
        #print("observation after action:", obs[0])
        if player_sum_idx >= 0:
            v_val[action_taken, player_sum_idx, dealer_card_idx, ace_idx] += reward
            counts[action_taken, player_sum_idx, dealer_card_idx, ace_idx] += 1

    #V = V/counts
    return v_val, counts

def plt_fig(V, usable_ace, ax):
    x_range = np.arange(12, 22)
    y_range = np.arange(1, 11)
    X, Y = np.meshgrid(x_range, y_range)

    Z = np.array([np.sum(V[:,x-12,y-1, usable_ace]) for x, y in zip(np.ravel(X), np.ravel(Y))]).reshape(X.shape)

    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, vmin=-1.0, vmax=1.0)
    ax.set_xlabel('player_sum')
    ax.set_ylabel('dealer_card')
    ax.set_zlabel('reward')
    ax.set_zlim(-1.01, 1.01)
    ax.view_init(ax.elev, -120)

def plot_v(V):
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(211, projection='3d')
    ax.set_title('Usable Ace')
    plt_fig(V, 1, ax)
    ax = fig.add_subplot(212, projection='3d')
    ax.set_title('No Usable Ace')
    plt_fig(V, 0, ax)
    plt.savefig("blackjack.png")
    plt.show()

def mc(episodes=500000):
    v_val = np.zeros((2,10,10,2)) #Value for (action, player_sum, dealer_card, useable_ace)
    V_avg = np.zeros((2,10,10,2))
    counts = np.zeros((2,10,10,2)) #counter for (action, player_sum, dealer_card, useable_ace)
    for episode in range(episodes):
        V_ep, counts_ep = mc_prediction("sum20")
        #v_val[counts_ep != 0] = V_ep[counts_ep != 0]
        v_val += V_ep
        counts += counts_ep
        #V_avg[V_avg==0] = V_ep[V_avg==0]/counts_ep[V_avg==0]
        if episode % 100000 == 0:
            counts_ep_new = counts
            counts_ep_new[counts_ep_new == 0] = 1 # add 1's to avoid divison by 0
            V_avg_ep = v_val/counts_ep_new
            actions_no_ace = np.argmax(V_avg_ep[:,:,:,0], axis=0)
            actions_ace = np.argmax(V_avg_ep[:,:,:,1], axis=0)
            print("Best policy (no ace) at ep.{}:\n {}".format(episode, actions_no_ace))
            print("Best policy (with ace) at ep.{}:\n {}".format(episode, actions_ace))

    counts[counts == 0] = 1 # add 1's to avoid divison by 0
    V_avg = v_val/counts
    plot_v(V_avg)
    #actions = np.amax(V_avg)

def mc_es(episodes=500000):
    v_val = np.zeros((2,10,10,2)) #Value for (action, player_sum, dealer_card, useable_ace)
    counts = np.zeros((2,10,10,2)) #counter for (action, player_sum, dealer_card, useable_ace)
    for episode in range(episodes):
        V_ep, counts_ep = mc_prediction(np.random.choice([0,1],size=(10,10,2)))
        #v_val[counts_ep != 0] = V_ep[counts_ep != 0]
        v_val += V_ep
        counts += counts_ep
        #V_avg[V_avg==0] = V_ep[V_avg==0]/counts_ep[V_avg==0]
        if episode % 100000 == 0:
            counts_ep_new = counts
            counts_ep_new[counts_ep_new == 0] = 1 # add 1's to avoid divison by 0
            V_avg_ep = v_val/counts_ep_new
            actions_no_ace = np.argmax(V_avg_ep[:,:,:,0], axis=0)
            actions_ace = np.argmax(V_avg_ep[:,:,:,1], axis=0)
            print("Best policy (no ace) at ep.{}:\n {}".format(episode, actions_no_ace))
            print("Best policy (with ace) at ep.{}:\n {}".format(episode, actions_ace))

    counts[counts == 0] = 1 # add 1's to avoid divison by 0
    V_avg = v_val/counts
    #plot_v(V_avg)
    actions_no_ace = np.argmax(V_avg[:,:,:,0], axis=0)
    actions_ace = np.argmax(V_avg[:,:,:,1], axis=0)
    print("Best policy (no ace):\n {}".format(actions_no_ace))
    print("Best policy (with ace):\n {}".format(actions_ace))
    #actions = np.amax(V_avg)


if __name__ == "__main__":
    mc()
    #mc_es()
