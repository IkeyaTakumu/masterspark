import argparse

def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--camera_id',
        default = 0,
        type=int
    )
    parser.add_argument(
        '--window_height',
        default = int(9*80),
        type = int
    )
    parser.add_argument(
        '--window_width',
        default = int(16*80),
        type = int
    )
    parser.add_argument(
        '--bdbox_xp',
        default = 1,
        type = int
    )
    parser.add_argument(
        '--bdbox_yp',
        default = 1,
        type = int
    )
    parser.add_argument(
        '--bdbox_width',
        default = 200,
        type = int
    )
    parser.add_argument(
        '--bdbox_height',
        default = 200,
        type = int
    )
    parser.add_argument(
        '--track_mode',
        default = "cam",
        type = str
    )
    args = parser.parse_args()

    return args
