import torch
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def weight(adapter_model):
    all_params = []
    all_params_flattened = []
    if isinstance(adapter_model, dict):
        all_params = []
        for key, value in adapter_model.items():
            if isinstance(value, torch.Tensor):
                all_params.append(value.flatten().cpu().numpy())  # 展平为一维并转为 numpy 数组
                # print(key)
        # 将所有展平后的参数合并成一个一维向量
        # all_params_flattened = np.concatenate(all_params)
    #all_params = [item for sublist in all_params for item in sublist]
    return all_params

def compute(model_name):
    adapter_model1 = torch.load(f'/data1/jinni/first/OpenFedLLM-main/output/{model_name}/client_grad-0.pt')
    adapter_model2 = torch.load(f'/data1/jinni/first/OpenFedLLM-main/output/{model_name}/client_grad-1.pt')
    adapter_model3 = torch.load(f'/data1/jinni/first/OpenFedLLM-main/output/{model_name}/client_grad-2.pt')
    adapter_model4 = torch.load(f'/data1/jinni/first/OpenFedLLM-main/output/{model_name}/client_grad-3.pt')
    adapter_model5 = torch.load(f'/data1/jinni/first/OpenFedLLM-main/output/{model_name}/client_grad-4.pt')
    log_abs_parameters1 = weight(adapter_model1)  # np.log10(np.abs(model1p) + 1e-8)
    log_abs_parameters2 = weight(adapter_model2)  # np.log10(np.abs(model2p) + 1e-8)
    log_abs_parameters3 = weight(adapter_model3)  # np.log10(np.abs(model3p) + 1e-8)
    log_abs_parameters4 = weight(adapter_model4)  # np.log10(np.abs(model4p) + 1e-8)
    log_abs_parameters5 = weight(adapter_model5)  # np.log10(np.abs(model5p) + 1e-8)

    all_lists = [log_abs_parameters1, log_abs_parameters2, log_abs_parameters3, log_abs_parameters4,
                 log_abs_parameters5]
    all_sim_max = []
    for i in range(len(log_abs_parameters1)):  # 遍历32个子列表
        # 从每个列表中提取第i个子列表
        sublist_vectors = [lst[i] for lst in all_lists]

        # 计算所有子列表之间的余弦相似度
        similarity_matrix = cosine_similarity(sublist_vectors)
        all_sim_max.append(similarity_matrix)
    result = np.sum(all_sim_max, axis=0)

    print(result)

    log_abs_parameters1 = [item for sublist in log_abs_parameters1 for item in sublist]
    log_abs_parameters2 = [item for sublist in log_abs_parameters2 for item in sublist]
    log_abs_parameters3 = [item for sublist in log_abs_parameters3 for item in sublist]
    log_abs_parameters4 = [item for sublist in log_abs_parameters4 for item in sublist]
    log_abs_parameters5 = [item for sublist in log_abs_parameters5 for item in sublist]
    all_lists = [log_abs_parameters1, log_abs_parameters2, log_abs_parameters3, log_abs_parameters4,
                 log_abs_parameters5]
    similarity_matrix = cosine_similarity(all_lists)  # 计算余弦相似度
    print(similarity_matrix)


def julei(model_name):
    adapter_model1 = torch.load(f'/data1/jinni/first/OpenFedLLM-main/output/{model_name}/client_grad-0.pt')
    adapter_model2 = torch.load(f'/data1/jinni/first/OpenFedLLM-main/output/{model_name}/client_grad-1.pt')
    adapter_model3 = torch.load(f'/data1/jinni/first/OpenFedLLM-main/output/{model_name}/client_grad-2.pt')
    adapter_model4 = torch.load(f'/data1/jinni/first/OpenFedLLM-main/output/{model_name}/client_grad-3.pt')
    adapter_model5 = torch.load(f'/data1/jinni/first/OpenFedLLM-main/output/{model_name}/client_grad-4.pt')
    adapter_model6 = torch.load(f'/data1/jinni/first/OpenFedLLM-main/output/{model_name}/client_grad-5.pt')
    log_abs_parameters1 = weight(adapter_model1)  # np.log10(np.abs(model1p) + 1e-8)
    log_abs_parameters2 = weight(adapter_model2)  # np.log10(np.abs(model2p) + 1e-8)
    log_abs_parameters3 = weight(adapter_model3)  # np.log10(np.abs(model3p) + 1e-8)
    log_abs_parameters4 = weight(adapter_model4)  # np.log10(np.abs(model4p) + 1e-8)
    log_abs_parameters5 = weight(adapter_model5)  # np.log10(np.abs(model5p) + 1e-8)
    log_abs_parameters6 = weight(adapter_model6)
    log_abs_parameters1 = [item for sublist in log_abs_parameters1 for item in sublist]
    log_abs_parameters2 = [item for sublist in log_abs_parameters2 for item in sublist]
    log_abs_parameters3 = [item for sublist in log_abs_parameters3 for item in sublist]
    log_abs_parameters4 = [item for sublist in log_abs_parameters4 for item in sublist]
    log_abs_parameters5 = [item for sublist in log_abs_parameters5 for item in sublist]
    log_abs_parameters6 = [item for sublist in log_abs_parameters6 for item in sublist]

    all_lists = [log_abs_parameters1, log_abs_parameters2, log_abs_parameters3, log_abs_parameters4,
                 log_abs_parameters5,log_abs_parameters6]
    agg_clust = AgglomerativeClustering(n_clusters=2)

    # 使用数据进行聚类
    labels = agg_clust.fit_predict(all_lists)

    # 查看聚类结果
    print("Cluster labels:", labels)
    return labels
