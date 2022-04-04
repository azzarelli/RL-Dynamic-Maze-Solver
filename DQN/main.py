from run import run

if __name__ == '__main__':
    run(train_chck=False, chckpt=False, episodes=300,
        lr=0.001, epsilon=0.3, gamma=0.99,
       netname='dynamic-test1')

