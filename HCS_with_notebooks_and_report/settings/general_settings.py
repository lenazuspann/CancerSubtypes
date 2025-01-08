# set parameters for __main__.py
perform_initial_analysis = False
use_saved_results = True
plot_graph = False
evaluate_results = True


# set parameters for the construction of the similarity graph
epsilon = 0.6
sigma = 150
weighted = True
load_K = False


# initialization of important values
dict_class_description = {'BRCA': 'Breast invasive carcinoma',
                          'KIRC': 'Kidney renal clear cell carcinoma',
                          'COAD': 'Colon adenocarcinoma',
                          'LUAD': 'Lung adenocarcinoma',
                          'PRAD': 'Prostate adenocarcinoma'}

classes = ['KIRC', 'LUAD', 'PRAD']