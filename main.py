from run import run

if __name__ == '__main__':
    iter = [1]

    for i in iter:
        run(canv_chck=True, chckpt=False, episodes=1000,
            lr=0.0001, epsilon=0.9, gamma=0.7,
           netname='variations', batch_size=64, epsilon_min=0.05)