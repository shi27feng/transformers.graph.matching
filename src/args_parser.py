import argparse


def parameter_parser():
    """
    A method to parse up command line parameters.
    The default hyper-parameters give a high performance model without grid search.
    """
    parser = argparse.ArgumentParser(description="Run SimGNN.")

    parser.add_argument("--metric",
                        default="ged",
                        help="Metric. Default is (ged), another option is (mcs)")

    parser.add_argument("--dataset_name",
                        nargs="?",
                        default="linux",
                        help="Dataset name. Default is LINUX")
    
    parser.add_argument("--dataset_root",
                        nargs="?",
                        default="../dataset/",
                        help="Dataset name. Default is LINUX")

    parser.add_argument("--gnn_operator",
                        nargs="?",
                        default="gcn",
                        help="Type of GNN-Operator. Default is gcn")

    parser.add_argument("--regularization",
                        default="mean",  # 'sigmoid', 'sum'
                        help="regularization type")

    parser.add_argument("--epochs",
                        type=int,
                        default=200,  # 350,
                        help="Number of training epochs. Default is 350.")

    parser.add_argument("--gnn_dims",
                        type=str,
                        default='32,64',
                        help="Filters (neurons) in primal graph convolutions. Default is (64, 32).")

    parser.add_argument("--mha_dim",
                        type=int,
                        default=64,
                        help="The dimension of multi-head attention.")
    
    parser.add_argument("--fc_hidden",
                        type=int,
                        default=64,
                        help="The hidden dimension of fully-connected layer.")

    parser.add_argument("--kernel-size",
                        type=int,
                        default=3,
                        help="kernel dimensions of 2D convolutions. Default is 4.")

    parser.add_argument("--use-residual",
                        type=bool,
                        default=True,
                        help="Whether to use residual adjacency matrix. Default is False.")

    parser.add_argument("--stride",
                        type=int,
                        default=1,
                        help="kernel dimensions of 2D convolutions. Default is 1.")

    parser.add_argument("--batch-size",
                        type=int,
                        default=128,
                        help="Number of graph pairs per batch. Default is 128.")

    parser.add_argument("--valid-batch-factor",
                        type=int,
                        default=2,
                        help="Number of graph pairs per batch. Default is 128.")

    parser.add_argument("--dropout",
                        type=float,
                        default=0.3,
                        help="Dropout probability. Default is 0.")

    parser.add_argument("--learning-rate",
                        type=float,
                        default=0.001,
                        help="Learning rate. Default is 0.001.")

    parser.add_argument("--weight-decay",
                        type=float,
                        default=5e-4,
                        help="Adam weight decay. Default is 5*10^-4.")

    parser.add_argument("--plot",
                        dest="plot",
                        action="store_true")

    parser.add_argument("--synth",
                        dest="synth",
                        action="store_true")

    parser.add_argument("--measure-time",
                        action="store_true",
                        help="Measure average calculation time for one graph pair. Learning is not performed.")

    parser.add_argument("--notify",
                        dest="notify",
                        action="store_true",
                        help="Send notification message when the code is finished (only Linux & Mac OS support).")

    parser.add_argument("--save-model",
                        type=bool,
                        default=False)

    parser.add_argument("--checkpoint-path",
                        type=str,
                        default='checkpoint')

    parser.set_defaults(plot=True)
    parser.set_defaults(measure_time=False)
    parser.set_defaults(notify=False)
    parser.set_defaults(synth=False)

    return parser.parse_args()