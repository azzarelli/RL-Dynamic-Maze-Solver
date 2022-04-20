from ConvDDQN.run import run

if __name__ == '__main__':
    iter = [1]

    for i in iter:
        run(canv_chck=True, chckpt=False, netname='variations',
            episodes=100000, lr=0.0001, gamma=0.999, batch_size=128,
            epsilon=0.9, epsilon_min=0.001, ep_dec=0.005,
            beta_inc=0.001)