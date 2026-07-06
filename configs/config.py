class DataConfig:
    dataset     = 'cifar10'   # 'cifar10' or 'mnist'
    save_root   = './data'
    train_stop  = 20000
    test_stop   = 2000


class DPConfig:
    epsilons    = [10]
    dp_c        = 1.0
    lr          = 0.001   # 对齐 Mjölnir
    dp_delta    = 1e-5
    mechanism   = 'gaussian'  


class SPEConfig:
    mode        = 'spe'       # 'spe' or 'direct'
    fc_dim      = 65536       # 16 * 64 * 64
    reshape     = (16, 64, 64)
    wavelet     = 'db4'
    wavelet_level = 2
    save_root   = './train_data'
    save_vis    = False  
    seed        = 42
    max_samples = None


class TrainConfig:
    dataset        = 'cifar10'
    epsilon        = 10
    batch_size     = 364
    num_epochs     = 200
    lr             = 3e-4
    weight_decay   = 5e-4
    max_lr         = 1e-3
    use_adaptive_t = False  
    num_workers    = 4
    save_root      = './results'
