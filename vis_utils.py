import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from kaokore_ds import *
from sampler_utils import *
import wandb

def class_distro_chart(classes_dict, class_names, file_name):
    plt.figure(figsize=(15,8))
    class_distro = {class_names[i]: classes_dict[i] for i in classes_dict}
    chart = sns.barplot(data = pd.DataFrame.from_dict([class_distro]).melt(), x = "variable", y="value", hue="variable").set_title('Natural Images Class Distribution')
    fig = chart.get_figure()
    fig.savefig(f'figures/{file_name}.png')
    
if __name__ == '__main__':
    kaokore_status_class_names = {0: 'noble', 1: 'warrior', 2: 'incarnation', 3: 'commoner'}
    # class_distro_chart(train_dataset.count_dict,
    #                    kaokore_status_class_names,
    #                    'original_train_status_classdistro')
    # class_distro_chart(get_class_distribution(train_loader_ws, train_dataset, 'train'),
    #                    kaokore_status_class_names,
    #                    'classweighted_sampling_status_classdistro')
    
    # class_distro_chart(batch_sampler.count_dict_new,
    #                    kaokore_status_class_names,
    #                    'odin_sampling_status_classdistro')
    
    wrun = wandb.init(project='ood_art_visualization_tests')
    
    columns = ['epoch', 'index', 'class', 'confidence score', 'image']
    epoch = 1
    least_conf = wandb.Table(columns = columns)
    most_conf = wandb.Table(columns = columns)
    num_rows = 4
    
    
    
    
    for cls in list(kaokore_status_class_names.keys()):
        cls_indices = (batch_sampler.all_y == cls).nonzero()
        og_indices = torch.tensor(batch_sampler.old_indices).to(device)
        sampling_probs_cls = batch_sampler.sampling_probs[cls_indices]
        
        sampling_scores, indices=torch.sort(sampling_probs_cls, dim = 0)
        rev_sampling_scores, rev_indices=torch.sort(sampling_probs_cls, dim = 0, descending = True)
        sampler_indices, rev_sampler_indices = cls_indices[indices], cls_indices[rev_indices]
        og_indices, rev_og_indices = og_indices[sampler_indices], og_indices[rev_sampler_indices]
        indices, rev_indices = og_indices.squeeze().tolist(), rev_og_indices.squeeze().tolist()
        sampling_scores, rev_sampling_scores = sampling_scores.squeeze(), rev_sampling_scores.squeeze()
        
        #careful about the mapping, both the sampler and the train dataset needed to be mapped
        print(cls, indices, batch_sampler.all_y[sampler_indices].squeeze().tolist(),
              [train_dataset[i][2] for i in indices], sampling_scores)
    
        most_confident_vals = [[i, None] for i in sampling_scores[:num_rows]]
        for i,ind in enumerate(indices[:num_rows]):
            most_confident_vals[i][1] = train_dataset[ind][0]
            
        rev_indices = indices[:len(indices)-num_rows-1:-1] #reverse the last nrows indices to not sort a second time
        # print(len(rev_indices))
        # print(sampling_scores[:num_rows])
        least_confident_vals = [[i, None] for i in rev_sampling_scores[:num_rows]]
        for i,ind in enumerate(rev_indices):
            least_confident_vals[i][1] = train_dataset[ind][0]
        
        #sanity check
        for i in range(len(most_confident_vals)):
            if most_confident_vals[i][1] is not None and least_confident_vals[i][1] is not None:
                print(most_confident_vals[i][0], least_confident_vals[i][0])
            else:
                assert False, "Confident images not indexed"
                
        for i in range(num_rows):
            l_img = wandb.Image(least_confident_vals[i][1])
            m_img = wandb.Image(most_confident_vals[i][1])
            row_tab1 = [epoch, 
                        indices[i],
                   kaokore_status_class_names[train_dataset[indices[i]][2]], 
                   least_confident_vals[i][0].item(),
                   l_img]
            row_tab2 = [epoch,
                        rev_indices[i],
                   kaokore_status_class_names[train_dataset[rev_indices[i]][2]], 
                   most_confident_vals[i][0].item(),
                   m_img]
            least_conf.add_data(*row_tab1)
            most_conf.add_data(*row_tab2)
        
    # Create a wandb Artifact
    artifact = wandb.Artifact(
        name="confidence_images",
        type="kaokore_dataset",
    )
    
    artifact.add(least_conf, "least_confident")
    artifact.add(most_conf, "most_confident")
    wrun.log_artifact(artifact)
    
    wrun.finish()
     
           