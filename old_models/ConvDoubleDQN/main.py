from old_models.ConvDoubleDQN.run import run

if __name__ == '__main__':
    iter = [1]

    for i in iter:
        run(canv_chck=True, chckpt=True, netname='variations',
            episodes=100000, lr=0.001, gamma=0.9999, batch_size=64,
            epsilon=0.7, epsilon_min=0.05, ep_dec=0.001,
            beta_inc=0.0005)