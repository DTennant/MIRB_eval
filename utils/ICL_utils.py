import random
import copy


def select_demonstration(support_meta, n_shot, dataset, query=None):
    if 'operator_induction' in dataset:
        operator_index = {'+': 0, '-': 1, 'x': 2}
        n_shot_support_raw = random.sample(support_meta, n_shot)
        n_shot_support = copy.deepcopy(n_shot_support_raw)
        operator = query['operator']
        operator_idx = operator_index[operator]
        for support in n_shot_support:    
            support['answer'] = support['answer'][operator_idx]
    elif dataset == 'open_mi':
        # use two classes for now
        query_class = query['answer']
        other_class = random.choice([cls for cls in query['classes'] if cls != query_class])
        order_keys = [query_class, other_class] if random.choice([True, False]) else [other_class, query_class]
        answers = {query_class: query_class, other_class: other_class}
        
        n_shot_support = []
        for i in range(n_shot):
            for key in order_keys:
                # For each key, add one shot
                support = {
                    'image': [query['support'][key]['images'][i]], 
                    'answer': answers[key],
                    'question': "This is a"
                }
                n_shot_support.append(support)
    elif dataset == 'open_mi_5':
        # 5 classes
        classes = query['classes']
        random.shuffle(classes)
        answers = {cls: cls for cls in classes} 
        
        n_shot_support = []
        for i in range(n_shot):
            for key in classes:
                # For each key, add one shot
                support = {
                    'image': [query['support'][key]['images'][i]], 
                    'answer': answers[key],
                    'question': "This is a"
                }
                n_shot_support.append(support)
    
    elif dataset == 'matching_mi':
        n_shot_support_raw = copy.deepcopy(random.sample(support_meta, n_shot))
        n_shot_support = []
        for i in range(n_shot):
            n_shot_support.append(n_shot_support_raw[i]['same'])
            n_shot_support.append(n_shot_support_raw[i]['diff'])
    
    elif dataset == 'open_t2i_mi':
        query_class = query['task_label']
        other_class = random.choice([cls for cls in query['classes'] if cls != query_class])
        order_keys = [query_class, other_class] if random.choice([True, False]) else [other_class, query_class]
        answers = {query_class: query_class, other_class: other_class}
        n_shot_support = []
        for i in range(n_shot):
            for key in order_keys:
                # For each key, add one shot
                if dataset == 'open_t2i_mi':
                    support = {
                        'image': query['support'][key]['images'][i], 
                        'question': f'Generate a {key}'
                    }
                else:
                    support = {
                        'answer': query['support'][key]['images'][i], 
                        'question': f'Generate a {key}'
                    }
                n_shot_support.append(support)
    elif dataset == 'cobsat':
        latent_var = query['latent']
        latent = query[latent_var]
        task = query['task']
        # get support set with same latents
        n_shot_support = [x for x in support_meta if (x[latent_var] == latent and x['latent'] == latent_var and x['task'] == task)]
        n_shot_support = copy.deepcopy(random.sample(n_shot_support, n_shot))
    else:
        n_shot_support = random.sample(support_meta, n_shot)
    return n_shot_support

def get_task_instruction(args, dataset):
    if dataset in ['analogy', 'domain', 'plot', 'image_needles', 'plot_text', 'places', 'image_needles_concat']:
        instr = 'Answer with a single word.'
    elif dataset in ['codeu', 'foods', 'image_jigsaw', 'codeu_text']:
        instr = 'Answer with the option symbol.'
    elif dataset in ['arxiv', 'arxiv_text']:
        instr = 'Answer with the paper title.'
    elif dataset in ['count', 'count_concat']:
        instr = 'Answer with a single number.'
    elif dataset in ['3d_scene', '3d_scene_concat']:
        instr = 'The following images are different views of the same 3D scene. Answer with a single number.'
    
    return instr

def format_answer(answer, dataset, query=None):
    if dataset in ['count']:
        answer = str(answer)
    return answer