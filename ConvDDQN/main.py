from ConvDDQN.run import run

if __name__ == '__main__':
    iter = [1]

    for i in iter:
        run(canv_chck=1, chckpt=False, netname='warmup',
            episodes=100000, lr=0.001, gamma=0.9999, batch_size=128,
            epsilon=0.4, ep_dec=0.0001,
            beta_inc=0.00005)
