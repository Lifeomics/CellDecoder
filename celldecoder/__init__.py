
from .config import get_args
from .utils import *
from .data import *
from .model.bnn import Net
from .utils.train import Trainer
from .utils.train import Tester
from argparse import Namespace


def train(**kwargs):
    args = get_args(" ".join(f"--{k} {v}" for k,v in kwargs.items()).split())
    
    # seed and log
    seed = random_state = args.seed
    seed_everything(seed)
    log_dir = args.log_dir
    os.makedirs(log_dir, exist_ok=True)
    info_dict = get_arg_dict(args)
    json.dump(info_dict, open(os.path.join(log_dir, "args.json"), "w"), indent=2)
    
    # data
    dataset, args, info = load_data(args)
    num_features, num_classes, sample_num = (
        args.num_features,
        args.num_classes,
        args.num_samples,
    )
    y = info["y"]
    dataset = load_train_split(args, dataset, y, random_state=random_state)
    
    # hyper-param 
    device = f"cuda:{args.device_id}"
    hid_dim = args.nhid
    lr = args.lr

    # net
    model = Net(num_features, hid_dim, num_classes, dataset, args, device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # train
    metrics = "accuracy f1_macro f1_micro".split()
    trainer = Trainer(model, optimizer, metrics, args)
    val_metric_dict = trainer.train_till_end(dataset, device)
    print('Validation metrics: ', val_metric_dict)
    
    return val_metric_dict 

def test(**kwargs):
    args = Namespace(**kwargs)
    
    # load original configs
    print("Loading config from {}".format(os.path.join(args.log_dir, "args.json")))
    cfg = json.load(open(os.path.join(args.log_dir, "args.json")))
    cfg.update(args.__dict__)
    args = Namespace(**cfg)
    args.ckpt_path = os.path.join(args.log_dir, args.ckpt_model)
    print(args)
    
    # seed and log
    seed = random_state = args.seed
    seed_everything(seed)
    
    # data
    dataset, args, info = load_data(args)
    num_features, num_classes, sample_num = (
        args.num_features,
        args.num_classes,
        args.num_samples,
    )
    
    # hyper-param 
    device = f"cuda:{args.device_id}"
    hid_dim = args.nhid

    # net
    model = Net(num_features, hid_dim, num_classes, dataset, args, device).to(device)
    model.load_state_dict(
        torch.load(osp.join(args.ckpt_path), map_location=torch.device("cpu"))
    )
    model = model.to(device)
    
    # test
    metrics = "accuracy f1_macro f1_micro".split()
    tester = Tester(model, metrics, args)
    test_metric_dict = tester.test(dataset, device)
    print('Test metrics: ', test_metric_dict)
    
    return test_metric_dict


def predict(**kwargs):
    args = Namespace(**kwargs)
    
    # load original configs
    print("Loading config from {}".format(os.path.join(args.log_dir, "args.json")))
    cfg = json.load(open(os.path.join(args.log_dir, "args.json")))
    cfg.update(args.__dict__)
    args = Namespace(**cfg)
    args.ckpt_path = os.path.join(args.log_dir, args.ckpt_model)
    print(args)
    
    # seed and log
    seed = random_state = args.seed
    seed_everything(seed)
    
    # data
    dataset, args, info = load_data(args)
    num_features, num_classes, sample_num = (
        args.num_features,
        args.num_classes,
        args.num_samples,
    )
    
    # hyper-param 
    device = f"cuda:{args.device_id}"
    hid_dim = args.nhid

    # net
    model = Net(num_features, hid_dim, num_classes, dataset, args, device).to(device)
    model.load_state_dict(
        torch.load(osp.join(args.ckpt_path), map_location=torch.device("cpu"))
    )
    model = model.to(device)
    
    # test
    metrics = "accuracy f1_macro f1_micro".split()
    tester = Tester(model, metrics, args)
    if args.predict_type == 'id':
        preds = tester.pred(dataset, device)
        return preds
    elif args.predict_type == 'prob':
        preds = tester.pred(dataset, device, prob = True)
        return preds
    elif args.predict_type == 'cell':
        preds = tester.pred(dataset, device)
        id2cls = list(dataset.raw_data['cls2id'].keys())
        preds = [id2cls[x] for x in preds]
        return preds
    else:
        raise NotImplementedError(f"{args.predict_type} not implemented")
    
def embed(**kwargs):
    from .model.bnn import NetFeat
    from torch_geometric.loader import DataLoader
    
    # load original configs
    args = Namespace(**kwargs)
    print("Loading config from {}".format(os.path.join(args.log_dir, "args.json")))
    cfg = json.load(open(os.path.join(args.log_dir, "args.json")))
    cfg.update(args.__dict__)
    args = Namespace(**cfg)
    args.ckpt_path = os.path.join(args.log_dir, args.ckpt_model)
    print(args)

    # seed and log
    seed = random_state = args.seed
    seed_everything(seed)

    # data
    dataset, args, info = load_data(args)
    num_features, num_classes, sample_num = (
        args.num_features,
        args.num_classes,
        args.num_samples,
    )

    # hyper-param 
    device = f"cuda:{args.device_id}"
    hid_dim = args.nhid

    # net
    model = NetFeat(num_features, hid_dim, num_classes, dataset, args, device).to(device)
    model.load_state_dict(
        torch.load(osp.join(args.ckpt_path), map_location=torch.device("cpu"))
    )
    model = model.to(device)

    batch_size = args.batch_size
    loader = DataLoader(dataset, batch_size, shuffle = False)
    model.eval()

    preds = []
    internal_embeddings = {}
    for data in tqdm(loader):
        data = data.to(device)
        out = model(data.x, data.batch).cpu()
        ie = model.internal_embeddings
        for k,v in ie.items():
            if k not in internal_embeddings:
                internal_embeddings[k] = []
            internal_embeddings[k].append(v)

    if args.out_embed == 'output':
        internal_embeddings = {'output':internal_embeddings['output']}

    for k,v in internal_embeddings.items():
        internal_embeddings[k] = torch.cat(v, dim=0)

    save_dir = os.path.join(args.log_dir, f"{args.dataset}-embed.pt")
    torch.save(internal_embeddings, save_dir)

    print(f"saving to {save_dir}")
    print("# keys")
    print(internal_embeddings.keys())
    print("# Tensor shapes")
    for k in internal_embeddings.keys():
        print(k, internal_embeddings[k].shape)
        
    return internal_embeddings

def explain_feature(**kwargs):
    from .model.bnn import NetFeat
    from torch_geometric.loader import DataLoader
    
    # load original configs
    args = Namespace(**kwargs)
    print("Loading config from {}".format(os.path.join(args.log_dir, "args.json")))
    cfg = json.load(open(os.path.join(args.log_dir, "args.json")))
    cfg.update(args.__dict__)
    args = Namespace(**cfg)
    args.ckpt_path = os.path.join(args.log_dir, args.ckpt_model)
    print(args)

    # seed and log
    seed = random_state = args.seed
    seed_everything(seed)

    # data
    dataset, args, info = load_data(args)
    num_features, num_classes, sample_num = (
        args.num_features,
        args.num_classes,
        args.num_samples,
    )

    # hyper-param 
    device = f"cuda:{args.device_id}"
    hid_dim = args.nhid
    log_dir = args.log_dir

    explain_file = osp.join(log_dir, f"{args.dataset}-explain-{args.explain_method}.pth")

    if args.explain_method in "grad grad_cam".split():
        from .model.bnn_explain_feature import Net
        from .utils.explain import get_grads, get_id2nodes, FeatExplainer

        model = Net(num_features, hid_dim, num_classes, dataset, args, device)
        model.load_state_dict(torch.load(args.ckpt_path, map_location=torch.device("cpu")))
        model = model.to(device)

        batch_size = args.batch_size
        dataloader = DataLoader(dataset, batch_size, shuffle=False)
        model.eval()

        grads, fmaps, ys = get_grads(model, dataloader, device)  # [layer,num_samples,node,dim]

        id2nodes = [get_id2nodes(hid2id) for hid2id in dataset.raw_data["hid2ids"]]

        # calculate explains
        exps = FeatExplainer(device, fmaps, grads, args.explain_method, batch_size = args.batch_size).gen_explains_gpu()

        # save
        explain_dict = {"exp_samples": exps, "ys": ys, "id2nodes": id2nodes, "hid2ids": dataset.raw_data["hid2ids"]}
        print("explainations saved to ", explain_file)
        torch.save(explain_dict, explain_file)
        
    else:
        from ppi.model.bnn_explain import Net
        from ppi.utils.explainer import get_model_att, process_attention

        model = Net(num_features, hid_dim, num_classes, dataset, args, device)
        model.load_state_dict(torch.load(args.ckpt_path, map_location=torch.device("cpu")))
        model = model.to(device)

        batch_size = 1 # must be 1 so that we can calculate attention for each graph
        dataloader = DataLoader(dataset, batch_size, shuffle=False)
        model.eval()

        # attention
        assert args.encoder == "gat", "only support gat"
        best_attentions = get_model_att(dataset.datas, model, device)
        attention_dict = process_attention(dataset, best_attentions, hierarchy=args.hierarchy, cross=False, return_sample = args.return_sample)
        print("explainations saved to ", explain_file)
        torch.save(attention_dict, explain_file)
        
        if args.prod_value:
            assert args.return_sample == 1, "return_sample must be 1"
            name2id = dataset.raw_data['hid2ids'][0]
            # explain_dict['0']['LILRB2,C1orf162'] [S, K]
            
            edge_attns = attention_dict['0'] # {edge_name: [S, K]}
            edge_names = list(edge_attns.keys()) # [E]
            edge_values = np.stack(list(edge_attns.values())) # [E, S, K]
            source_names = [name.split(',')[0] for name in edge_names]
            source_ids = np.array([name2id[name] for name in source_names]) 
            source_x = np.stack([dataset.datas[i].x[source_ids] for i in range(len(dataset.datas))]) # [S, E, 1]
            source_x = source_x.transpose(1, 0, 2) # [E, S, 1]
            edge_attns_mul = edge_values * source_x
            attn_mul = dict(zip(edge_names, edge_attns_mul))
            
            explain_file_mul = explain_file+"_mul"
            print("explainations(with mul) saved to ", explain_file_mul)
            torch.save(attn_mul, explain_file_mul)
            

def explain_ppi(**kwargs):
    args = Namespace(**kwargs)
    
    # load original configs
    print("Loading config from {}".format(os.path.join(args.log_dir, "args.json")))
    cfg = json.load(open(os.path.join(args.log_dir, "args.json")))
    cfg.update(args.__dict__)
    args = Namespace(**cfg)
    args.ckpt_path = os.path.join(args.log_dir, args.ckpt_model)
    print(args)
    
    # seed and log
    seed = random_state = args.seed
    seed_everything(seed)
    
    # data
    dataset, args, info = load_data(args)
    num_features, num_classes, sample_num = (
        args.num_features,
        args.num_classes,
        args.num_samples,
    )
    
    # hyper-param 
    device = f"cuda:{args.device_id}"
    hid_dim = args.nhid

    # net
    from .model.bnn_explain import Net
    from .utils.explainer import PPIExplainer

    model = Net(num_features,args.nhid,num_classes,dataset,args,device)
    model.load_state_dict(torch.load(args.ckpt_path, map_location=torch.device("cpu")))
    model = model.to(device)

    explainer = PPIExplainer(
            model, dataset=dataset, explain_cross=False, 
            device=device, epochs=args.exp_train_epochs, lr=args.exp_lr,
            train_sample_gt=args.train_sample_gt, ce_loss_gt=args.ce_loss_gt)
    out_put_dict = explainer.explain(args)
    # save mask dict to dir
    file_name = args.ckpt_path.split('/')[-1].split('.')[0]+ '_' + args.encoder+str(args.multi_atten) + '_explainer_edge_mask'
    os.makedirs(os.path.join(args.log_dir,'ppi'))
    with open(os.path.join(args.log_dir,'ppi', file_name), 'w') as f:
        json.dump(out_put_dict, f, indent=4)