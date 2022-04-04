from run import run

if __name__ == '__main__':
    run(canv_chck=True, chckpt=True, episodes=10000,
        lr=0.001, epsilon=0.1, gamma=0.9,
       netname='dynamic-lr0001-walkintowalls-conv-DeadVisitor-v2')

