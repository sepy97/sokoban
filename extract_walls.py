from glob import glob
from os import makedirs
from argparse import ArgumentParser

import numpy as np

def main(input_directory: str, output_directory: str, verbose: bool = False):
    iteration = 0
    files = glob(f"{input_directory}/*")
    makedirs(output_directory, exist_ok=True)

    for filepath in files:
        with open(filepath, 'r') as file:
            # Extract the entire board from the file
            board = np.array([list(line[:-1]) for line in file.readlines()])
            
            if verbose:
                print('\n'.join(map(''.join, board)))
            
            # Add the outside walls to fill the gaps until the actual stage begins
            for i, first_wall in enumerate(np.argmax(board == '#', 1)):
                board[i, :first_wall] = "#"
        
            for i, last_wall in enumerate(np.argmax((board == '#')[:, ::-1], 1)):
                board[i, -(last_wall + 1):] = "#"
                
            for i, first_wall in enumerate(np.argmax(board == '#', 0)):
                board[:first_wall, i] = "#"
                
            for i, last_wall in enumerate(np.argmax((board == '#')[:, ::-1], 0)):
                board[-(last_wall + 1):, i] = "#"
                
            if verbose:
                print('\n'.join(map(''.join, board)))
                print('-' * 80)
            
            # Extract just the walls
            walls = np.array(np.where(board == '#')).T + 1
            walls = np.ascontiguousarray(walls)
            size = board.shape[::-1]

            # Output the target filetype
            with open(f"{output_directory}/{iteration:04d}.txt", 'w') as output_file:
                print(" ".join(map(str, size)), file=output_file)
                print(len(walls), end = ' ', file=output_file)
                print(' '.join(map(str, walls.ravel())), file=output_file)
                
                print(0, file=output_file)
                print(0, file=output_file)
                print("0 0", end="", file=output_file)
            
            iteration += 1

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("input_directory", type=str, help="Location with the map files to parse.")
    parser.add_argument("output_directory", type=str, help="Location to place the output txt files.")
    parser.add_argument('-v', "--verbose", action='store_true', help="Print the boards while creating them.")

    main(**parser.parse_args().__dict__)