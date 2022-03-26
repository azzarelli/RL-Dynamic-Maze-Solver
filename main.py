from run import run

if __name__ == '__main__':
    run(train_chck=False, chckpt=False, episodes=100, lr=0.001, epsilon=0.9, gamma=0.99, netname='DDQN-na18-lr0001ep09gam099.pt')
