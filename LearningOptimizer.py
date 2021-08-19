from sklearn.linear_model import SGDClassifier

from SplitDataRunner import run_learner
import datetime
import csv

def get_permutations(dict_in, args_in, result):
    if len(args_in):
        curr_key = list(args_in.keys())[0]
        # get the options array for one arg
        curr_options_array = args_in[curr_key]
        # delete the fetched arg from the input object
        new_args = args_in.copy()
        del new_args[curr_key]
        # iterate over existing options
        for option in curr_options_array:
            c_dict = dict_in.copy()
            c_dict[curr_key] = option
            result = get_permutations(c_dict, new_args, result)
    else:
        result.append(dict_in)
    return result


def log(instr):
    time = datetime.datetime.now().isoformat()
    print(time, instr)


def to_csv(dict_array, csv_path):
    keys = dict_array[0].keys()
    with open(csv_path, 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(dict_array)


def test_all(data_folder, config_in, out_file_name):
    log('starting')
    for learnerName, learner_config in config_in.items():
        print(f'start working on learner: {learnerName}')
        print(learner_config)
        permutations = get_permutations({}, learner_config['args'], [])
        print(len(permutations))
        results_array = []
        for perm in permutations:
            log('running with args' + str(perm))
            result = run_learner(learner_config["learner"], 30, data_folder, perm)
            log('results: ' + str(result))
            perm.update(result)
            results_array.append(perm)

        log('done')

    if out_file_name is not None:
        to_csv(results_array, out_file_name)


if __name__ == "__main__":
    config = {
        "SGDClassifier": {
            "learner": SGDClassifier,
            "args": {
                "alpha": [0.0001, 0.001, 0.01],
                "penalty": ['l2', 'l1', 'elasticnet'],
                "loss": ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron', 'squared_loss', 'huber',
                         'epsilon_insensitive', 'squared_epsilon_insensitive'],
                "l1_ratio": [0.15, 0.3, 0.45, 0.6, 0.75],
                "warm_start": [False, True]
                # "class_weight": [None, "balanced"]
            }
        }
    }
    test_all('./tmp', config, './optimizer_out.csv')
