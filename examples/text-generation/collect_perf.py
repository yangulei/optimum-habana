import os
import pandas as pd

def get_stats(fname, name):
    if not os.path.exists(fname):
        return None
    
    res = None
    with open(fname, 'r') as f:
        for line in f.readlines():
            if name in line:
                res = float(line.split(' ')[-2])
                break
    return res


model = "llama2-13b"
config = "1c"
list_in = [3072]
list_out = [1,5120]
list_bs = [1,2,4,8,16,32,64,128,256,512]

for input_tokens in list_in:
    df_throughput = pd.DataFrame(index=list_bs, columns=list_out)
    df_memory = pd.DataFrame(index=list_bs, columns=list_out)
    df_compilation = pd.DataFrame(index=list_bs, columns=list_out)
    for output_tokens in list_out:
        for bs in list_bs:
            res_dir="res-{}-{}-in{}-out{}-bs{}".format(model, config, input_tokens, output_tokens, bs)
            res_path = os.path.join(res_dir, 'run.log')
            print("processing {}".format(res_path))
            df_throughput.loc[bs,output_tokens] = get_stats(res_path, "Throughput (including tokenization)")
            df_memory.loc[bs,output_tokens] = get_stats(res_path, "Max memory allocated")
            df_compilation.loc[bs,output_tokens] = get_stats(res_path, "Graph compilation duration")
    df_throughput.to_csv("res-{}-in{}_throughput.csv".format(config, input_tokens))
    df_memory.to_csv("res-{}-in{}_memory.csv".format(config, input_tokens))
    df_compilation.to_csv("res-{}-in{}_compilation.csv".format(config, input_tokens))


config = "tp"
list_in = [3072]
list_out = [1,5120]
list_bs = [1,2,4,8,16,32,64,128]

for input_tokens in list_in:
    df_throughput = pd.DataFrame(index=list_bs, columns=list_out)
    df_memory = pd.DataFrame(index=list_bs, columns=list_out)
    df_compilation = pd.DataFrame(index=list_bs, columns=list_out)
    for output_tokens in list_out:
        for bs in list_bs:
            res_dir="res-{}-{}-in{}-out{}-bs{}".format(model, config, input_tokens, output_tokens, bs)
            res_path = os.path.join(res_dir, 'run.log')
            print("processing {}".format(res_path))
            df_throughput.loc[bs,output_tokens] = get_stats(res_path, "Throughput (including tokenization)")
            df_memory.loc[bs,output_tokens] = get_stats(res_path, "Max memory allocated")
            df_compilation.loc[bs,output_tokens] = get_stats(res_path, "Graph compilation duration")
    df_throughput.to_csv("res-{}-in{}_throughput.csv".format(config, input_tokens))
    df_memory.to_csv("res-{}-in{}_memory.csv".format(config, input_tokens))
    df_compilation.to_csv("res-{}-in{}_compilation.csv".format(config, input_tokens))