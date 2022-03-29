from run import run

if __name__ == '__main__':
    run(train_chck=False, chckpt=True, episodes=1000,
        lr=0.01, epsilon=0.5, gamma=0.99,
       netname='dynamic-new')

