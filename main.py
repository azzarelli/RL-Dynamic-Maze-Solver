from run import run

if __name__ == '__main__':
    run(canv_chck=True, chckpt=False, episodes=10000,
        lr=0.001, epsilon=0.9, gamma=0.9,
       netname='deadvision-per-l1loss-lr0001-gam09')

