from run import run

if __name__ == '__main__':
    iter = [1]

    for i in iter:
        run(canv_chck=True, chckpt=False, episodes=10000,
            lr=0.001, epsilon=0.8, gamma=0.999,
           netname='variations', batch_size=64, epsilon_min=0.00033, beta_inc=0.01, ep_dec=0.005)