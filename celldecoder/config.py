import argparse
import os

def get_args(args=None):
    parser = argparse.ArgumentParser()

    # logs
    parser.add_argument("--log_dir", type=str, default="../../logs/tmp/", help = "log directory")
    parser.add_argument("--device_id", type=int, default=1, help = "device id")
    parser.add_argument("--ckpt_model", default="best.pth", type=str, help="checkpoint model name")
    parser.add_argument("--ckpt_path", default=None, type=str, help="checkpoint path")

    # dataset
    parser.add_argument("--dataset", type=str, default="Tumor", help="dataset name")
    parser.add_argument("--hierarchy", type=int, default=3, choices=[3, 5], help="hierarchy level")
    parser.add_argument("--num_nodes", type=int, default=-1, help="number of nodes")
    parser.add_argument("--num_features", type=int, default=-1, help="number of features")
    parser.add_argument("--num_classes", type=int, default=-1, help="number of classes")
    parser.add_argument("--num_samples", type=int, default=-1, help="number of samples")
    parser.add_argument("--fn_process", type=str, default="processed-train", help="processed data file")
    

    # setting
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--add_self_loop", type=int, default=1, help="whether add self loop")
    parser.add_argument("--val_ratio", type=float, default=0.3, help="validation ratio")
    parser.add_argument("--max_epochs", type=int, default=1000, help="max training epochs")
    parser.add_argument("--early_metric", type=str, default="f1_macro", help="early stopping metric")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--patience", type=int, default=50, help="early stopping patience")
    parser.add_argument("--bootstrap_num", type=int, default=-1, help="bootstrap number")

    # searchable args
    parser.add_argument(
        "--nhid", type=int, default=16, help="dim of hidden embedding"
    )
    parser.add_argument("--n_layers", type=int, default=3, help="number of layers")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--dropout", type=float, default=0, help="dropout rate")
    parser.add_argument("--skip_raw", type=int, default=0, help="whether skip raw features")
    parser.add_argument(
        "--add_one_hot",
        type=int,
        default=0,
        choices=[0, 1],
        help="whether add one-hot features",
    )
    parser.add_argument("--encoder", type=str, default="gin", choices=["gin", "gat"], help="encoder type")
    parser.add_argument("--heads", type=int, default=1, help="number of heads")
    parser.add_argument("--pool", type=str, default="mean", choices=["mean", "gat"], help="pooling type")
    parser.add_argument("--pool_act", type=str, default="", help="pooling activation function")

    # mode
    parser.add_argument(
        "--train", type=int, default=1, choices=[0, 1], help="1 train 0 test"
    )
    parser.add_argument(
        "--predict_type", type=str, default="cell", choices = ["cell","prob","id"], help="type of predictions"
    )
    
    # explain_feature
    parser.add_argument(
        "--explain_method", type=str, default="grad", choices=["grad", "grad_cam", "attention"], help="explanation method"
    )
    parser.add_argument("--return_sample", type=int, default=0, help="whether return sample-wise explaination")
    parser.add_argument("--prod_value", type=int, default=0, help="whether return the product of explaination and value")
    parser.add_argument("--out_embed", default="output", type=str, help="output embedding name")
    
    # explain_ppi
    parser.add_argument('--correlation', default=0, type=int, choices=[0,1], help = "whether outputing the correlation between edge masks of different labels")
    parser.add_argument('--multi_atten', default=1, type=int, choices=[0,1], help = "whether calculating product of edge mask and edge attention in gat-encoder in explain_ppi")
    parser.add_argument('--train_sample_gt', default=0, type=int, choices=[0,1],help = "For calculating cmask in training explain_ppi, 0 use ground-truth label, 1 use predicted label")
    parser.add_argument('--ce_loss_gt', default=0, type=int, choices=[0,1],help="For calculating mask loss in explain_ppi, 0 use ground-truth label, 1 use predicted label") 
    parser.add_argument('--exp_train_epochs',type=int,default=100, help='number of epochs for training explain_ppi')
    parser.add_argument('--exp_lr',type=float,default=0.01, help='learning rate for training explain_ppi')

    args = parser.parse_args(args)
    args.ckpt_path = os.path.join(args.log_dir, args.ckpt_model)
    return args