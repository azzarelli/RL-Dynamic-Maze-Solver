from ConvDDQN.run import run

if __name__ == '__main__':
    iter = [1]

    for i in iter:
        run(canv_chck=True, chckpt=False, netname='DDvariations',
            episodes=100000, lr=0.001, gamma=0.9999, batch_size=64,
            epsilon=0.9, epsilon_min=0.05, ep_dec=0.001,
            beta_inc=0.0005)