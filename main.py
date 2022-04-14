from run import run

if __name__ == '__main__':
    iter = [0.1,0.01,0.001,0.0001]

    for i in iter:
        run(canv_chck=True, chckpt=False, episodes=1000,
            lr=0.001, epsilon=0.9, gamma=0.99,
           netname='variations', batch_size=64, epsilon_min=0.2, ep_dec=0.01)