import pandas as pd

# Note: The following functions (get_embedding_weight, add_hooks, evaluate_batch, 
# get_average_grad, get_accuracy, get_best_candidates, get_loss_per_candidate) 
# were originally from allennlp but are not currently used in the codebase.
# They are kept for reference but can be removed if not needed.

def get_goals_and_targets(params):

	train_goals = getattr(params, 'goals', [])
	train_targets = getattr(params, 'targets', [])
	test_goals = getattr(params, 'test_goals', [])
	test_targets = getattr(params, 'test_targets', [])
	offset = getattr(params, 'data_offset', 0)

	if params.train_data:
		train_data = pd.read_csv(params.train_data)
		train_targets = train_data['target'].tolist()[offset:offset+params.n_train_data]
		if 'goal' in train_data.columns:
			train_goals = train_data['goal'].tolist()[offset:offset+params.n_train_data]
		else:
			train_goals = [""] * len(train_targets)
		if params.test_data and params.n_test_data > 0:
			test_data = pd.read_csv(params.test_data)
			test_targets = test_data['target'].tolist()[offset:offset+params.n_test_data]
			if 'goal' in test_data.columns:
				test_goals = test_data['goal'].tolist()[offset:offset+params.n_test_data]
			else:
				test_goals = [""] * len(test_targets)
		elif params.n_test_data > 0:
			test_targets = train_data['target'].tolist()[offset+params.n_train_data:offset+params.n_train_data+params.n_test_data]
			if 'goal' in train_data.columns:
				test_goals = train_data['goal'].tolist()[offset+params.n_train_data:offset+params.n_train_data+params.n_test_data]
			else:
				test_goals = [""] * len(test_targets)

	assert len(train_goals) == len(train_targets)
	assert len(test_goals) == len(test_targets)
	print('Loaded {} train goals'.format(len(train_goals)))
	print('Loaded {} test goals'.format(len(test_goals)))

	return train_goals, train_targets, test_goals, test_targets

# Unused functions removed - they were using allennlp which is not needed
# If you need these functions, they can be reimplemented without allennlp dependency