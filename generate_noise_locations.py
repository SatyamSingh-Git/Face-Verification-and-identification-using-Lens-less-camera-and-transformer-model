import argparse
import numpy as np
import random


def get_noise_locations(num_pos):
    positions = []
    for chan in range(1, 5):
        if chan == 4:
            positions += [(random.randint(32, 64 - 1), random.randint(32, 64 - 1), chan) for _ in
                          range(num_pos * 3)]
        else:
            positions += [(random.randint(0, 64 - 1), random.randint(0, 64 - 1), chan)
                          for _ in
                          range(num_pos)]
    return positions

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate pseudo-random noise locations."
    )

    parser.add_argument(
        "--loc_per_pixel",
        type=int,
        required=True,
        help="Number of locations per pixel."
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    loc_per_pixel = args.loc_per_pixel

    positions = get_noise_locations(loc_per_pixel)
    np.save(f'noise_{loc_per_pixel}_pixels_per_block.npy', positions)

    