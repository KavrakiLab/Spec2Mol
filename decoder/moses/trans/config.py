import argparse


def get_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()

    # Model
    model_arg = parser.add_argument_group("Model")
    model_arg.add_argument(
        "--q_cell",
        type=str,
        default="gru",
        choices=["gru"],
        help="Encoder rnn cell type",
    )
    model_arg.add_argument(
        "--q_bidir",
        default=False,
        action="store_true",
        help="If to add second direction to encoder",
    )
    model_arg.add_argument(
        "--q_d_h", type=int, default=256, help="Encoder h dimensionality"
    )
    model_arg.add_argument(
        "--q_n_layers", type=int, default=1, help="Encoder number of layers"
    )
    model_arg.add_argument(
        "--q_dropout", type=float, default=0.5, help="Encoder layers dropout"
    )
    model_arg.add_argument(
        "--d_cell",
        type=str,
        default="gru",
        choices=["gru"],
        help="Decoder rnn cell type",
    )
    model_arg.add_argument(
        "--d_n_layers", type=int, default=3, help="Decoder number of layers"
    )
    model_arg.add_argument(
        "--d_dropout", type=float, default=0, help="Decoder layers dropout"
    )
    model_arg.add_argument(
        "--d_z", type=int, default=128, help="Latent vector dimensionality"
    )
    model_arg.add_argument(
        "--d_d_h", type=int, default=512, help="Decoder hidden dimensionality"
    )
    model_arg.add_argument(
        "--freeze_embeddings",
        default=False,
        action="store_true",
        help="If to freeze embeddings while training",
    )
    model_arg.add_argument(
        "--fc_h",
        type=int,
        default=512,
        help="Fully connected hidden dimensionality",
    )
    model_arg.add_argument(
        "--fc_use_relu",
        default=False,
        action="store_true",
        help="should use relu in fc layer",
    )
    model_arg.add_argument(
        "--fc_use_bn",
        default=False,
        action="store_true",
        help="should use bn in fc layer",
    )
    model_arg.add_argument(
        "--regression_loss_weight",
        type=float,
        default=1.0, #Equal weight
        help="Weight to be used for regression loss",
    )
    model_arg.add_argument(
        "--regression_field_names",
        type=str,
        default="MolLogP,MolMR,BalabanJ,NumHAcceptors,NumHDonors,NumValenceElectrons,TPSA,QED,SA,logP",
        help="Fields used for regression",
    )

    model_arg.add_argument(
        "--ignore_vae",
        action="store_true",
        help="Should use VAE reparameterization?",
    )

    model_arg.add_argument(
        "--use_tanh",
        action="store_true",
        help="Should use tanh in the final of the model",
    )


    # Train
    train_arg = parser.add_argument_group("Train")
    train_arg.add_argument(
        "--n_batch", type=int, default=512, help="Batch size"
    )
    train_arg.add_argument(
        "--tb_log_interval", type=int, default=100, help="Batch size"
    )

    train_arg.add_argument(
        "--clip_grad",
        type=int,
        default=50,
        help="Clip gradients to this value",
    )
    train_arg.add_argument(
        "--kl_start",
        type=int,
        default=0,
        help="Epoch to start change kl weight from",
    )
    train_arg.add_argument(
        "--kl_w_start", type=float, default=0, help="Initial kl weight value"
    )
    train_arg.add_argument(
        "--kl_w_end", type=float, default=0.05, help="Maximum kl weight value"
    )
    train_arg.add_argument(
        "--lr_start", type=float, default=3 * 1e-4, help="Initial lr value"
    )
    train_arg.add_argument(
        "--lr_n_period",
        type=int,
        default=10,
        help="Epochs before first restart in SGDR",
    )
    train_arg.add_argument(
        "--lr_n_restarts",
        type=int,
        default=10,
        help="Number of restarts in SGDR",
    )
    train_arg.add_argument(
        "--lr_n_mult",
        type=int,
        default=1,
        help="Mult coefficient after restart in SGDR",
    )
    train_arg.add_argument(
        "--lr_end",
        type=float,
        default=3 * 1e-4,
        help="Maximum lr weight value",
    )
    train_arg.add_argument(
        "--n_last",
        type=int,
        default=1000,
        help="Number of iters to smooth loss calc",
    )
    train_arg.add_argument(
        "--n_jobs", type=int, default=1, help="Number of threads"
    )
    train_arg.add_argument(
        "--n_workers",
        type=int,
        default=1,
        help="Number of workers for DataLoaders",
    )

    resume_arg = parser.add_argument_group("Resume")
    resume_arg.add_argument(
        "--resume",
        default=False,
        action="store_true",
        help="Resume from a saved model",
    )
    resume_arg.add_argument(
        "--model_load",
        type=str,
        required=False,
        help="Where to load the model",
    )
    resume_arg.add_argument(
        "--config_load",
        type=str,
        required=False,
        help="Where to load the config",
    )
    resume_arg.add_argument(
        "--start_epoch",
        type=int,
        required=False,
        default=0,
        help="Where to load the config",
    )


    return parser
