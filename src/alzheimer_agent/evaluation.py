import numpy as np


def classification_metrics(y_true, reviewed_labels):
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, cohen_kappa_score, confusion_matrix
    if any(x not in ('CN', 'MCI', 'AD') for x in reviewed_labels):
        raise ValueError('Report unresolved cases separately; do not silently assign a class')
    labels = ['CN', 'MCI', 'AD']
    cm = confusion_matrix(y_true, reviewed_labels, labels=labels)
    specificity = []
    for i in range(3):
        tp = cm[i, i]; fn = cm[i].sum()-tp; fp = cm[:, i].sum()-tp; tn = cm.sum()-tp-fn-fp
        specificity.append(tn/(tn+fp) if tn+fp else np.nan)
    return dict(n=len(y_true), ACC=accuracy_score(y_true, reviewed_labels),
        macro_F1=f1_score(y_true, reviewed_labels, labels=labels, average='macro', zero_division=0),
        precision=precision_score(y_true, reviewed_labels, labels=labels, average='macro', zero_division=0),
        recall=recall_score(y_true, reviewed_labels, labels=labels, average='macro', zero_division=0),
        kappa=cohen_kappa_score(y_true, reviewed_labels, labels=labels), specificity=float(np.nanmean(specificity)))


def progression_metrics(y_true, risk, threshold=0.5):
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score
    y, p = np.asarray(y_true), np.asarray(risk)
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('Invalid risks')
    return dict(n=len(y), prevalence=float(y.mean()), threshold=threshold,
                ACC=accuracy_score(y, p >= threshold), F1=f1_score(y, p >= threshold, zero_division=0),
                AUROC=roc_auc_score(y, p) if len(set(y)) == 2 else None,
                AUPRC=average_precision_score(y, p) if len(set(y)) == 2 else None)


def age_metrics(ages, predicted_raw):
    from sklearn.metrics import r2_score
    a, p = np.asarray(ages), np.asarray(predicted_raw)
    error = p-a
    return dict(n=len(a), MAE=float(abs(error).mean()), RMSE=float(np.sqrt((error**2).mean())),
                R2=r2_score(a, p), Pearson=float(np.corrcoef(a, p)[0, 1]) if np.std(a) and np.std(p) else None,
                signed_bias=float(error.mean()))
