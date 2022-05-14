from ConvDDQN.run import run

if __name__ == '__main__':
    iter = [1]

    for i in iter:
        run(canv_chck=0, chckpt=False, netname='warmup',
            episodes=100000, lr=0.001, gamma=0.9995, batch_size=128,
            epsilon=0.6, ep_dec=0.01,
            beta_inc=0.001)
