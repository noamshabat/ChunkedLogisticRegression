import os.path
import os
import math
import pprint
import platform
import random
import json
import pandas as pd

pp = pprint.PrettyPrinter(indent=4)


def verify_csv(input_file):
    if type(input_file) != str:
        raise TypeError("Expected 'input_file' to be a string representing the file name. Received: " + str(type(input)))

    if not input_file.endswith('.csv'):
        raise AttributeError("Expected the input file to be a csv file (*.csv). File is " + input_file)

    if not os.path.exists(input_file):
        raise FileNotFoundError("The input file does not exist: " + input_file)


def get_titles(fd, separator):
    # TODO: is there any way to make sure this is a title row?
    return fd.readline().replace("\n","").split(separator)


def create_target_dir(target_dir):
    if os.path.isfile(target_dir):
        raise AttributeError("Target dir already exists and is a file: " + target_dir)

    if os.path.exists(target_dir):
        if len(os.listdir(target_dir)):
            raise AttributeError("Target dir already exists and is not empty: " + target_dir)
        return target_dir

    os.makedirs(target_dir)
    return target_dir


def get_line_count(fd):
    with fd:
        for i, l in enumerate(fd):
            pass
    return i + 1


def get_configuration(input_file, separator, max_chunk_size_bytes, num_shuffles, target_dir, test_percent):
    verify_csv(input_file)

    # open file and get first row - assume titles
    fd = open(input_file, 'r')

    # initialize configuration object
    configuration = {
        "input": input_file,
        "separator": separator,
        "chunk_size_bytes": max_chunk_size_bytes,
        "shuffles": num_shuffles,
        "titles": get_titles(fd, separator),
        "record_size": len(fd.readline()),
        "target_dir": create_target_dir(target_dir),
        "test_percent": test_percent,
        "test_lines": 0,  # will be filled with count of train lines.
        "line_count": get_line_count(fd) + 1  # we already read 1 line.
    }
    configuration["lines_per_chunk"] = int(max_chunk_size_bytes / configuration["record_size"])
    configuration["data_size"] = os.path.getsize(input_file) - len(separator.join(configuration["titles"]))
    configuration["num_cols"] = len(configuration["titles"])
    configuration["num_chunks"] = math.ceil(configuration["line_count"] / configuration["lines_per_chunk"])

    fd.close()
    return configuration


def preallocate_windows(file_name, size):
    os.system(f"fsutil file createnew {file_name} {size}")


def preallocate_linux(file_name, size):
    os.system(f"fallocate -l  {size} {file_name}")


def preallocate(file_name, size):
    os_preallocate = {
        "Windows": preallocate_windows,
        "Linux": preallocate_linux
    }
    os_preallocate[platform.system()](file_name, size)


def create_shuffle_folders_and_files(configuration):
    for shuffle in range(configuration['shuffles']):
        os.makedirs(configuration["target_dir"] + f"/shuffle{shuffle}")
        for chunk in range(configuration["num_chunks"]):
            preallocate(configuration["target_dir"] + f"/shuffle{shuffle}/chunk{chunk}", configuration["chunk_size_bytes"])


def shuffle_chunk(configuration, chunk, shuffle_index, shuffle_metadata):
    curr_shuffle_md = shuffle_metadata[shuffle_index]

    # get available chunk indexes based on written lines
    available_chunks = [chunki for chunki in curr_shuffle_md.keys() if chunki != 'written' and curr_shuffle_md[chunki]["lines"] < configuration["lines_per_chunk"]]

    for line in chunk:
        if len(available_chunks) == 0:
            print("warning - all chunks filled and still writing data! is input inconsistent?")
            available_chunks = curr_shuffle_md.keys()
        # random out chunk
        out_chunk = random.choice(available_chunks)
        curr_shuffle_md[out_chunk]["file"].write(f"{line}")
        curr_shuffle_md[out_chunk]["lines"] += 1
        curr_shuffle_md["written"] += 1
        if curr_shuffle_md[out_chunk]["lines"] >= configuration["lines_per_chunk"]:
            available_chunks.remove(out_chunk)


def create_shuffle_metadata(configuration):
    shuffle_metadata = {}
    for i in range(configuration["shuffles"]):
        shuffle_metadata[i] = {}
        for c in range(configuration["num_chunks"]):
            shuffle_metadata[i][c] = {
                "lines": 0,      # will hold number of lines written per chunk.
                "file": open(configuration["target_dir"] + f"/shuffle{i}/chunk{c}", 'w'),
            }
            shuffle_metadata[i]["written"] = 0

    return shuffle_metadata


# TODO: make this function pure. don't change the input.
def extract_train_data(configuration, chunk, size, train_file):
    for i in range(size):
        index = random.randint(0, len(chunk)-1)
        train_file.write(chunk[index])
        chunk.pop(index)
        configuration['test_lines'] += 1
    return chunk


def create_chunks(input_file, separator, max_chunk_size_bytes, num_shuffles, target_dir, test_percent):
    # first analyze the input file to create a configuration object.
    configuration = get_configuration(input_file, separator, max_chunk_size_bytes, num_shuffles, target_dir,
                                      test_percent)

    create_shuffle_folders_and_files(configuration)

    shuffle_metadata = create_shuffle_metadata(configuration)

    remaining_lines = configuration["line_count"]
    test_file = f"{configuration['target_dir']}/test.csv"
    with open(test_file, 'w') as tf:
        with open(input_file, 'r') as fd:
            next(fd)  # skip the title
            for chunk_index in range(configuration["num_chunks"]):
                # get lines for current chunk
                lines_to_get = min(configuration["lines_per_chunk"], remaining_lines)
                chunk = [next(fd) for x in range(lines_to_get)]
                remaining_lines -= lines_to_get

                post_train = extract_train_data(configuration, chunk, int(lines_to_get*0.25), tf)
                # write lines on all shuffles
                for shuffle_index in range(configuration["shuffles"]):
                    shuffle_chunk(configuration, post_train, shuffle_index, shuffle_metadata)
                    print(f"Finished shuffling chunk {chunk_index} shuffle {shuffle_index}")

    for shuffle_index in range(configuration["shuffles"]):
        if shuffle_metadata[shuffle_index]["written"] < (configuration["line_count"] - configuration["test_lines"]):
            raise ArithmeticError(f"Expected all shuffles to write all lines. Shuffle {shuffle_index} wrote " +
                                  f"{shuffle_metadata[shuffle_index]['written']}/{configuration['line_count']}")

    if remaining_lines > 0:
        raise ArithmeticError("Expected remaining lines to be 0. After all shuffles done it is now" + remaining_lines)

    with open(f"{configuration['target_dir']}/configuration.json", 'w') as fp:
        json.dump(configuration, fp)
    pp.pprint(configuration)


def get_shuffle_chunks(configuration, shuffle_index):
    curr_chunk_index = 0

    def get_chunks():
        nonlocal curr_chunk_index
        while curr_chunk_index < configuration["num_chunks"]:
            curr_chunk_index += 1
            yield pd.read_csv(f"{configuration['target_dir']}/shuffle{shuffle_index}/chunk{curr_chunk_index-1}",
                              header=None, names=configuration['titles'])

    return get_chunks


def get_shuffles(configuration):
    curr_shuffle_index = 0

    def shuffle_generator():
        nonlocal curr_shuffle_index
        while curr_shuffle_index < configuration["shuffles"]:
            curr_shuffle_index += 1
            yield get_shuffle_chunks(configuration, curr_shuffle_index - 1)

    return shuffle_generator


def get_split_data(target_dir):
    with open(f"{target_dir}/configuration.json") as json_file:
        configuration = json.load(json_file)

    return get_shuffles(configuration)


def get_test_data(target_dir):
    with open(f"{target_dir}/configuration.json") as json_file:
        configuration = json.load(json_file)

    return pd.read_csv(f"{target_dir}/test.csv", header=None, names=configuration['titles'])


# create_chunks('./diabetes.csv', ',', 5000, 3, './tmp', 0.25)
