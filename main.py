import numpy as np
import torch
import gym
from TD3 import TD3
import matplotlib.pyplot as plt
import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(seed):
    env_with_Dead = True  # Whether the Env has dead state. True for Env like BipedalWalkerHardcore-v3, CartPole-v0. False for Env like Pendulum-v0
    env = gym.make('BipedalWalkerHardcore-v3')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    expl_noise = 0.25
    print('  state_dim:', state_dim, '  action_dim:', action_dim, '  max_a:', max_action, '  min_a:', env.action_space.low[0])

    render = True
    Loadmodel = True
    ModelIdex =3600 #which model to load
    random_seed = seed

    Max_episode = 2000000
    save_interval = 400 #interval to save model

    if random_seed:
        print("Random Seed: {}".format(random_seed))
        torch.manual_seed(random_seed)
        env.seed(random_seed)
        np.random.seed(random_seed)

    writer = SummaryWriter(log_dir='runs/exp')

    kwargs = {
        "env_with_Dead":env_with_Dead,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "gamma": 0.99,
        "net_width": 200,
        "a_lr": 1e-4,
        "c_lr": 1e-4,
        "Q_batchsize":256,
    }
    model = TD3(**kwargs)
    if Loadmodel: model.load(ModelIdex)
    replay_buffer = ReplayBuffer.ReplayBuffer(state_dim, action_dim, max_size=int(1e6))

    all_ep_r = []

    for episode in range(Max_episode):
        s, done = env.reset(), False
        ep_r = 0
        steps = 0
        expl_noise *= 0.999

        '''Interact & trian'''
        while not done:
            steps+=1
            if render:
                a = model.select_action(s)
                s_prime, r, done, info = env.step(a)
                env.render()
            else:
                a = ( model.select_action(s) + np.random.normal(0, max_action * expl_noise, size=action_dim)
                ).clip(-max_action, max_action)
                s_prime, r, done, info = env.step(a)

                # Tricks for BipedalWalker
                if r <= -100:
                    r = -1
                    replay_buffer.add(s, a, r, s_prime, True)
                else:
                    replay_buffer.add(s, a, r, s_prime, False)

                if replay_buffer.size > 2000: model.train(replay_buffer)

            s = s_prime
            ep_r += r

        '''plot & save'''
        if (episode+1)%save_interval==0:
            model.save(episode + 1)
            # plt.plot(all_ep_r)
            # plt.savefig('seed{}-ep{}.png'.format(random_seed,episode+1))
            # plt.clf()


        '''record & log'''
        # all_ep_r.append(ep_r)
        if episode == 0: all_ep_r.append(ep_r)
        else: all_ep_r.append(all_ep_r[-1]*0.9 + ep_r*0.1)
        writer.add_scalar('s_ep_r', all_ep_r[-1], global_step=episode)
        writer.add_scalar('ep_r', ep_r, global_step=episode)
        writer.add_scalar('exploare', expl_noise, global_step=episode)
        print('seed:',random_seed,'episode:', episode,'score:', ep_r, 'step:',steps , 'max:', max(all_ep_r))

    env.close()




if __name__ == '__main__':
    main(seed=1)





