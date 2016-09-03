import collections
import sys
import numpy as np
import time
import pyspark
import random
import math

def get_block_dim(num_ids, num_blocks, block_id):
    cut_off_block = num_ids % num_blocks
    if block_id < cut_off_block:
        block_dim = num_ids / num_blocks + 1
    else:
        block_dim = num_ids / num_blocks
    return block_dim

def get_block_offset(num_ids, num_blocks, block_id):
    cut_off_block = num_ids % num_blocks
    if block_id <= cut_off_block:
        block_offset = (num_ids / num_blocks + 1) * block_id
    else:
        block_offset = (num_ids / num_blocks + 1) * cut_off_block \
          + (num_ids / num_blocks) * (block_id - cut_off_block)
    return block_offset

def get_block_id(num_ids, num_blocks, my_id):
    cut_off_block = num_ids % num_blocks
    cut_off_block_offset = (num_ids / num_blocks + 1) * cut_off_block

    if my_id >= cut_off_block_offset:
        block_id = (my_id - cut_off_block_offset) / (num_ids / num_blocks) + cut_off_block
    else:
        block_id = my_id / (num_ids / num_blocks + 1)
    return block_id

def parse_line(line):
    tokens = line.split(",")
    return int(tokens[0]), int(tokens[1]), float(tokens[2])

def blockify_matrix(partition_id, partition, num_y_ids, num_blocks):
    blocks = collections.defaultdict(lambda: collections.defaultdict(list))
    for entry in partition:
        _, (x_id, y_id, rating) = entry

        block_y_id = get_block_id(num_y_ids, num_blocks, y_id)
        blocks[partition_id][block_y_id].append(entry[1])

    for item in blocks.items():
        yield item

def initialize_factor_matrix(block_id, K, block_dim):
    factor_block = np.random.rand(block_dim, K)
    return block_id, factor_block

def sgd_on_one_partition(partition, sub_epoch, num_x_ids, num_y_ids, num_workers, step_size):
    # partition is of this form: (block_id, ((data, H_block), W_block))
    # data is a list of tuples
    # H_block and W_block are numpy arrays
    block_x_id = partition[0]
    block_y_id = (block_x_id + sub_epoch) % num_workers
    W_block = partition[1][0][1]
    H_block = partition[1][1]
    if not block_y_id in partition[1][0][0]:
        return block_x_id, (partition[1][0][0], W_block, H_block)
    data_block = partition[1][0][0][block_y_id]

    x_id_offset = get_block_offset(num_x_ids, num_workers, block_x_id)
    y_id_offset = get_block_offset(num_y_ids, num_workers, block_y_id)
    num_outrange = 0
    for sample in data_block:
        (x_id, y_id, rating) = sample
        W_id = x_id - x_id_offset
        H_id = y_id - y_id_offset

        diff = rating - np.dot(W_block[W_id], H_block[H_id])

        W_gradient = -2 * diff * H_block[H_id]
        W_block[W_id] -= step_size * W_gradient

        H_gradient = -2 * diff * W_block[W_id]
        H_block[H_id] -= step_size * H_gradient
    return block_x_id, (partition[1][0][0], W_block, H_block)

def evaluate_error(cart_prod, num_workers, num_x_ids, num_y_ids, error_accum, count_accum):
    block_x_id = cart_prod[0][0]
    block_y_id = cart_prod[1][0] % num_workers
    W_block = cart_prod[0][1][1]
    H_block = cart_prod[1][1]

    if not block_y_id in cart_prod[0][1][0]:
        return 0, 0
    data_block = cart_prod[0][1][0][block_y_id]

    x_id_offset = get_block_offset(num_x_ids, num_workers, block_x_id)
    y_id_offset = get_block_offset(num_y_ids, num_workers, block_y_id)
    error = .0
    n = 0
    for sample in data_block:
        (x_id, y_id, rating) = sample
        W_id = x_id - x_id_offset
        H_id = y_id - y_id_offset

        diff = rating - np.dot(W_block[W_id], H_block[H_id])
        error += diff ** 2
        n += 1

    error_accum.add(error)
    count_accum.add(n)

if __name__ == "__main__":
    csv_file = sys.argv[1]
    K = int(sys.argv[2]) #rank
    num_workers = int(sys.argv[3])
    num_iterations = int(sys.argv[4])
    step_size = 0.001
    step_size_decay = 0.99

    conf = pyspark.SparkConf().setAppName("DSGD-MF")
    sc = pyspark.SparkContext(conf=conf)

    ratings = sc.textFile(csv_file).map(parse_line)

    num_x_ids = ratings.map(lambda x: x[0]).max() + 1
    num_y_ids = ratings.map(lambda x: x[1]).max() + 1

    print 'num is ', num_x_ids, num_y_ids

    blockified_ratings = ratings.map( \
        lambda triple: (get_block_id(num_x_ids, num_workers, triple[0]), triple))\
      .partitionBy(num_workers) \
      .mapPartitionsWithIndex( \
        lambda partition_id, partition: blockify_matrix(partition_id, partition, num_y_ids, num_workers), \
        preservesPartitioning=True)

    print "blockified ratings"

    block_ids = sc.parallelize([x for x in range(num_workers)]).persist()

    # <block_id, W_block_mat>
    W_matrix = block_ids.map(lambda x: initialize_factor_matrix(x, K, \
                                                                get_block_dim(num_x_ids, num_workers, x)))

    # {idx, H_mat}
    H_matrices = {}
    H_matrices[0] = block_ids.map(lambda x: initialize_factor_matrix(x, K, \
                                                                     get_block_dim(num_y_ids, num_workers, x)))
    # {idx, <block_id, <block_ratings, W_block_mat>>}
    partitioned_data_with_Ws = {}
    partitioned_data_with_Ws[0] = blockified_ratings.join(W_matrix, numPartitions=num_workers)

    def perform_sgd_on_one_partition(sub_epoch, num_x_ids, num_y_ids, num_workers, step_size):
        return (lambda x: sgd_on_one_partition(x, sub_epoch, num_x_ids, num_y_ids, num_workers, step_size))

    def update_H_matrix_ids(num_workers):
        return (lambda x: ((x[0] - 1) % num_workers, x[1][2]))

    start_time = time.time()
    print("iteration\ttime (sec)\terror\trmse\tcount")
    rmses = []
    for iteration in range(num_iterations):
        error_accum = sc.accumulator(0.0)
        count_accum = sc.accumulator(0)
        for sub_epoch in range(num_workers):
            updated_joined_data = partitioned_data_with_Ws[sub_epoch]\
              .join(H_matrices[sub_epoch], num_workers)\
              .map(perform_sgd_on_one_partition(sub_epoch, num_x_ids, num_y_ids, \
                                            num_workers, step_size),\
                   preservesPartitioning=True)

            next_index = (sub_epoch + 1) % num_workers
            partitioned_data_with_Ws[next_index] = updated_joined_data\
              .map(lambda x: (x[0], (x[1][0], x[1][1])),\
                   preservesPartitioning=True)

            H_matrices[next_index] = updated_joined_data\
              .map(update_H_matrix_ids(num_workers))

            if next_index == 0:
               partitioned_data_with_Ws[0]\
                       .cartesian(H_matrices[0])\
                       .foreach(lambda x: evaluate_error(x, num_workers, num_x_ids, \
                                                    num_y_ids, error_accum, count_accum))

        error_total = error_accum.value
        n_total = count_accum.value
        print "%d\t%.4f\t%.4f\t%.4f\t%d" \
        % (iteration, time.time() - start_time,\
           error_total, math.sqrt(error_total / n_total), n_total)
        rmses.append(math.sqrt(error_total / n_total))

    step_size *= step_size_decay

    sc.stop()
    print rmses
