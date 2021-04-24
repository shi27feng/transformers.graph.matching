from utils import tab_printer
from trainer import GraphMatchTrainer
from parser import parameter_parser


def main():
    """
    Parsing command line parameters, reading data, fitting and scoring a SimGNN model.
    """
    args = parameter_parser()
    tab_printer(args)
    trainer = GraphMatchTrainer(args)

    if args.measure_time:
        trainer.measure_time()
    else:
        trainer.fit()
        trainer.score()

    if args.notify:
        import os, sys
        if sys.platform == 'linux':
            os.system('notify-send WaveSim "Program is finished."')
        elif sys.platform == 'posix':
            os.system("""
                      osascript -e 'display notification "WaveSim" with title "Program is finished."'
                      """)
        else:
            raise NotImplementedError('No support for this OS.')


if __name__ == "__main__":
    main()