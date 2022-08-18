import numpy as np
import functools
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, ShuffleSplit
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import normalize, OneHotEncoder
import importlib
import torch
from torch_sparse import SparseTensor
from model import LogReg


# borrow from BGRL [https://github.com/nerdslab/bgrl/blob/main/bgrl/logistic_regression_eval.py]
def fit_logistic_regression(X, y, data_random_seed=1, repeat=1):
    # transfrom targets to one-hot vector
    one_hot_encoder = OneHotEncoder(categories='auto', sparse=False)

    y = one_hot_encoder.fit_transform(y.reshape(-1, 1)).astype(np.bool)

    # normalize x
    X = normalize(X, norm='l2')

    # set random state
    rng = np.random.RandomState(data_random_seed)  # this will ensure the dataset will be split exactly the same
                                                   # throughout training

    accuracies = []
    for _ in range(repeat):
        # different random split after each repeat
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=rng)

        # grid search with one-vs-rest classifiers
        logreg = LogisticRegression(solver='liblinear')
        c = 2.0 ** np.arange(-10, 11)
        cv = ShuffleSplit(n_splits=5, test_size=0.5)
        clf = GridSearchCV(estimator=OneVsRestClassifier(logreg), param_grid=dict(estimator__C=c),
                           n_jobs=5, cv=cv, verbose=0)
        clf.fit(X_train, y_train)

        y_pred = clf.predict_proba(X_test)
        y_pred = np.argmax(y_pred, axis=1)
        y_pred = one_hot_encoder.transform(y_pred.reshape(-1, 1)).astype(np.bool)

        test_acc = accuracy_score(y_test, y_pred)
        accuracies.append(test_acc)
    return accuracies


# borrow from BGRL [https://github.com/nerdslab/bgrl/blob/main/bgrl/logistic_regression_eval.py]
def fit_logistic_regression_preset_splits(X, y, train_masks, val_masks, test_mask):
    # transfrom targets to one-hot vector
    one_hot_encoder = OneHotEncoder(categories='auto', sparse=False)
    y = one_hot_encoder.fit_transform(y.reshape(-1, 1)).astype(np.bool)

    # normalize x
    X = normalize(X, norm='l2')

    accuracies = []
    for split_id in range(train_masks.shape[1]):
        # get train/val/test masks
        train_mask, val_mask = train_masks[:, split_id], val_masks[:, split_id]

        # make custom cv
        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        # grid search with one-vs-rest classifiers
        best_test_acc, best_acc = 0, 0
        for c in 2.0 ** np.arange(-10, 11):
            clf = OneVsRestClassifier(LogisticRegression(solver='liblinear', C=c))
            clf.fit(X_train, y_train)

            y_pred = clf.predict_proba(X_val)
            y_pred = np.argmax(y_pred, axis=1)
            y_pred = one_hot_encoder.transform(y_pred.reshape(-1, 1)).astype(np.bool)
            val_acc = accuracy_score(y_val, y_pred)
            if val_acc > best_acc:
                best_acc = val_acc
                y_pred = clf.predict_proba(X_test)
                y_pred = np.argmax(y_pred, axis=1)
                y_pred = one_hot_encoder.transform(y_pred.reshape(-1, 1)).astype(np.bool)
                best_test_acc = accuracy_score(y_test, y_pred)

        accuracies.append(best_test_acc)
    print(np.mean(accuracies))
    return accuracies


def repeat(n_times):
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            results = [f(*args, **kwargs) for _ in range(n_times)]
            statistics = {}
            for key in results[0].keys():
                values = [r[key] for r in results]
                statistics[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values)}
            print_statistics(statistics, f.__name__)
            return statistics
        return wrapper
    return decorator


def prob_to_one_hot(y_pred):
    ret = np.zeros(y_pred.shape, np.bool)
    indices = np.argmax(y_pred, axis=1)
    for i in range(y_pred.shape[0]):
        ret[i][indices[i]] = True
    return ret


def print_statistics(statistics, function_name):
    print(f'(E) | {function_name}:', end=' ')
    for i, key in enumerate(statistics.keys()):
        mean = statistics[key]['mean']
        std = statistics[key]['std']
        print(f'{key}={mean:.4f}+-{std:.4f}', end='')
        if i != len(statistics.keys()) - 1:
            print(',', end=' ')
        else:
            print()


# borrow from GRACE [https://github.com/CRIPAC-DIG/GRACE/blob/master/eval.py]
# @repeat(5)
def label_classification(embeddings, y, ratio=0.1):
    X = embeddings.detach().cpu().numpy()
    Y = y.detach().cpu().numpy()
    Y = Y.reshape(-1, 1)
    onehot_encoder = OneHotEncoder(categories='auto').fit(Y)
    Y = onehot_encoder.transform(Y).toarray().astype(np.bool)

    X = normalize(X, norm='l2')

    X_train, X_test, y_train, y_test = train_test_split(X, Y,
                                                        test_size=1 - ratio)

    logreg = LogisticRegression(solver='liblinear')
    c = 2.0 ** np.arange(-10, 10)

    clf = GridSearchCV(estimator=OneVsRestClassifier(logreg),
                       param_grid=dict(estimator__C=c), n_jobs=8, cv=5,
                       verbose=0)
    clf.fit(X_train, y_train)

    y_pred = clf.predict_proba(X_test)
    y_pred = prob_to_one_hot(y_pred)

    acc = accuracy_score(y_test, y_pred)
    # return {"accuracy": acc}
    return acc


def heter_eval(model, z, y, train_mask, val_mask, test_mask, device):
    model.eval()
    num_classes = y.max().item()+1

    xent = torch.nn.CrossEntropyLoss()
    # z = torch.nn.functional.normalize(z, p=2, dim=1)
    log = LogReg(model.hidden, num_classes).to(device)
    opt = torch.optim.Adam(log.parameters(), lr=1e-2, weight_decay=0.0)

    train_embs = z[train_mask]
    val_embs = z[val_mask]
    test_embs = z[test_mask]

    best_acc_from_val = torch.zeros(1).cuda()
    best_val = torch.zeros(1).cuda()
    best_t = 0

    log.train()
    for i in range(100):
        opt.zero_grad()

        logits = log(train_embs)
        loss = xent(logits, y[train_mask].long())

        with torch.no_grad():
            ltra = log(train_embs)
            lv = log(val_embs)
            lt = log(test_embs)
            ltra_preds = torch.argmax(ltra, dim=1)
            lv_preds = torch.argmax(lv, dim=1)
            lt_preds = torch.argmax(lt, dim=1)
            train_acc = torch.sum(ltra_preds == y[train_mask]).float() / train_mask.sum()
            val_acc = torch.sum(lv_preds == y[val_mask]).float() / val_mask.sum()
            test_acc = torch.sum(lt_preds == y[test_mask]).float() / test_mask.sum()

            if val_acc > best_val:
                best_acc_from_val = test_acc
                best_val = val_acc
                best_t = i

        loss.backward()
        opt.step()
    return best_acc_from_val.cpu().item()


def eval(args, model, device):
    model.eval()
    load_dataset = getattr(importlib.import_module(f"dataset_apis.{args.dataset.lower()}"), 'load_eval_trainset')
    dataset = load_dataset()
    data = dataset[0].to(device)
    z = model.embed(data)
    if args.dataset.lower() in ['cora', 'citeseer', 'pubmed', 'dblp']:
        acc = label_classification(z, data.y, ratio=0.1)
    elif args.dataset.lower() in ['amazon_photos', 'amazon_computers', 'coauthor_cs', 'coauthor_physics']:
        acc = fit_logistic_regression(z, data.y)
    elif args.dataset.lower() == 'wikics':
        acc = fit_logistic_regression_preset_splits(z, data.y, data.train_mask, data.val_mask, data.test_mask)
    elif args.dataset.lower() in ["cornell", "wisconsin", "texas"]:
        acc_list = []
        for run in range(data.train_mask.shape[1]): # These datasets contains 10 different splits. Note: In our paper, we run 20 independent experiments. Different experiments use different splits.
            train_mask = data.train_mask[:, run%10]
            val_mask = data.val_mask[:, run%10]
            test_mask = data.test_mask[:, run%10]
            acc = heter_eval(model, z, data.y, train_mask, val_mask, test_mask, device)
            acc_list.append(acc)
        acc = np.mean(acc_list)
        print("Test acc: {:.2f}±{:.2f}".format(np.mean(acc_list)*100, np.std(acc_list)*100))
    else:
        raise NotImplementedError(f"{args.dataset} is not supported!")
    return acc
