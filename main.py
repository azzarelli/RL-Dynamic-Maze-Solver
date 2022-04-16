from run import run

if __name__ == '__main__':
    iter = [1]

    for i in iter:
        run(canv_chck=True, chckpt=False, episodes=1000,
            lr=0.001, epsilon=0.9, gamma=0.99,
           netname='variations', batch_size=16, epsilon_min=0.5, beta_inc=0.1, ep_dec=0.01)