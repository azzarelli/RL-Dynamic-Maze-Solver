from run import run

if __name__ == '__main__':
    iter = [1]

    for i in iter:
        run(canv_chck=True, chckpt=False, episodes=10000,
            lr=0.001, epsilon=0.9, gamma=0.9999,
           netname='variations', batch_size=64, epsilon_min=0.05, beta_inc=0.1, ep_dec=0.1)