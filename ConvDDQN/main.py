from ConvDDQN.run import run

if __name__ == '__main__':
    # Batch 128 or 256
    # gamma 0.9990-0.9999

    iter = [0.0001]
    for j in range(1):
        for i in iter:
            NAME = 'lr' + str(i) + '_' + str(j)
            run(canv_chck=0, chckpt=False, netname=NAME,
                episodes=10000, lr=0.0001, gamma=0.99, batch_size=64,
                epsilon=0.8, ep_dec=0.00001,
                beta_inc=0.0005)

