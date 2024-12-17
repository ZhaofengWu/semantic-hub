from multiprocessing import Pool, cpu_count
import os
import subprocess
import sys

from tqdm import tqdm


def convert_to_wav(input_file):
    output_file = f"{os.path.splitext(input_file)[0]}.wav"
    command = ["ffmpeg", "-i", input_file, "-ar", "16000", output_file]

    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        print(f"Error converting {input_file} to {output_file}: {e}")


def main(input_directory):
    input_directory = os.path.join(
        input_directory, "scratch/shared/beegfs/hchen/train_data/VGGSound_final/video/"
    )

    # List all .mp4 files in the specified directory
    input_files = [
        os.path.join(input_directory, f) for f in os.listdir(input_directory) if f.endswith(".mp4")
    ]

    # Estimate the number of worker processes to allow (based on CPU count)
    num_workers = min(len(input_files), cpu_count())

    # Create a pool of worker processes
    with Pool(num_workers) as pool:
        #       r = list(tqdm.tqdm(p.imap(_foo, range(30)), total=30))
        # pool.map(convert_to_wav, input_files)
        list(tqdm(pool.imap(convert_to_wav, input_files), total=len(input_files)))


if __name__ == "__main__":
    try:
        main(*sys.argv[1:])  # pylint: disable=no-value-for-parameter,too-many-function-args
    except Exception as e:
        import pdb
        import traceback

        if not isinstance(e, (pdb.bdb.BdbQuit, KeyboardInterrupt)):
            print("\n" + ">" * 100 + "\n")
            traceback.print_exc()
            print()
            pdb.post_mortem()
