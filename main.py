from run import run

if __name__ == '__main__':
    run(train_chck=False, chckpt=False, episodes=1000,
        lr=0.01, epsilon=0.9, gamma=0.99,
       netname='dynamic-new')

