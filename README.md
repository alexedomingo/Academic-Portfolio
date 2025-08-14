# Academic Portfolio

## Franklin D. Roosevelt Rhetoric Analysis
Jupyter Notebook Linked

## Predicting Policyholder Retention
Confidential Client as per UConn Academics,. Below is the LightGBVM Code

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import shap
    import lightgbm
    from lightgbm import LGBMClassifier
    
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import (
        f1_score, roc_auc_score, roc_curve, classification_report,
        precision_recall_curve, brier_score_loss, average_precision_score
    )
    
    from imblearn.combine import SMOTETomek, SMOTEENN
    from imblearn.under_sampling import RandomUnderSampler
    
    def run_lgbm_cv_with_shap(dfdrop, target_col, resampler, resampler_name, n_splits=5):
        X = dfdrop.drop(columns=target_col)
        y = dfdrop[target_col]

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    tprs, aucs = [], []
    mean_fpr = np.linspace(0, 1, 100)
    precisions, aps = [], []
    mean_recall = np.linspace(0, 1, 100)
    feature_importances = pd.DataFrame()
    shap_importances_class1 = pd.DataFrame()  

    fold_metrics = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n=== Fold {fold} ===")
        X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]


        cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
        num_cols = X_train.select_dtypes(include=['number']).columns.tolist()

        for col in cat_cols:
            for df_ in [X_train, X_test]:
                if pd.api.types.is_categorical_dtype(df_[col]):
                    if 'missing' not in df_[col].cat.categories:
                        df_[col] = df_[col].cat.add_categories('missing')
                    df_[col] = df_[col].fillna('missing')
                else:
                    df_[col] = df_[col].fillna('missing').astype(str)

        for col in num_cols:
            X_train[col] = X_train[col].fillna(0)
            X_test[col] = X_test[col].fillna(0)


        label_encoders = {}
        X_train_enc = X_train.copy()
        for col in cat_cols:
            X_train_enc[col] = X_train[col].astype(str)
            le = LabelEncoder()
            X_train_enc[col] = le.fit_transform(X_train_enc[col])
            label_encoders[col] = le


        X_res_enc, y_res = resampler.fit_resample(X_train_enc, y_train)
        X_res = pd.DataFrame(X_res_enc, columns=X_train.columns)

  
        for col in cat_cols:
            le = label_encoders[col]
            X_res[col] = le.inverse_transform(X_res[col].astype(int)).astype(str)
            X_test[col] = X_test[col].astype(str)

 
        for col in cat_cols:
            X_res[col] = X_res[col].astype('category')
            X_test[col] = X_test[col].astype('category')

        model = LGBMClassifier(
            objective='binary',
            learning_rate=0.05,
            n_estimators=1000,
            max_depth=6,
            num_leaves=31,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        )

        model.fit(
            X_res, y_res,
            eval_set=[(X_test, y_test)],
            eval_metric='f1',
            categorical_feature=cat_cols,
            callbacks=[
                lightgbm.early_stopping(stopping_rounds=50),
                lightgbm.log_evaluation(period=0)
            ]
        )

  
        y_proba = model.predict_proba(X_test)[:, 1]
        p, r, thresh = precision_recall_curve(y_test, y_proba)
        f1s = 2 * (p * r) / (p + r + 1e-8)
        best_idx = np.argmax(f1s)
        best_thresh = thresh[best_idx]
        y_pred = (y_proba >= best_thresh).astype(int)

        auc_val = roc_auc_score(y_test, y_proba)
        pr_auc = average_precision_score(y_test, y_proba)
        f1 = f1_score(y_test, y_pred)
        brier = brier_score_loss(y_test, y_proba)

        print(f"Fold {fold} | Threshold={best_thresh:.4f}, F1={f1:.4f}, ROC AUC={auc_val:.4f}, PR-AUC={pr_auc:.4f}, Brier={brier:.4f}")
        print(classification_report(y_test, y_pred, digits=4))


        fold_metrics.append({
            'fold': fold,
            'threshold': best_thresh,
            'f1': f1,
            'roc_auc': auc_val,
            'pr_auc': pr_auc,
            'brier': brier
        })


        fpr, tpr, _ = roc_curve(y_test, y_proba)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(auc_val)


        pr_interp = np.interp(mean_recall, r[::-1], p[::-1])  
        precisions.append(pr_interp)
        aps.append(pr_auc)

    
        fold_importance = pd.DataFrame({
            'feature': model.feature_name_,
            f'importance_fold_{fold}': model.feature_importances_
        })
        if feature_importances.empty:
            feature_importances = fold_importance
        else:
            feature_importances = feature_importances.merge(fold_importance, on='feature')


        explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
        shap_values = explainer(X_test) 



        shap_vals_class1 = shap_values.values  

        class1_indices = (y_test == 1)
        if class1_indices.sum() > 0:
            shap_class1_mean = np.abs(shap_vals_class1[class1_indices]).mean(axis=0)
            fold_shap_importance = pd.DataFrame({
                'feature': X_test.columns,
                f'shap_importance_fold_{fold}': shap_class1_mean
            })
            if shap_importances_class1.empty:
                shap_importances_class1 = fold_shap_importance
            else:
                shap_importances_class1 = shap_importances_class1.merge(fold_shap_importance, on='feature')
        else:
            print(f"Warning: No class 1 samples in test fold {fold} for SHAP class 1 importance.")


    plt.figure(figsize=(10, 8))
    mean_tpr = np.mean(tprs, axis=0)
    std_tpr = np.std(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)

    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.3f ± %0.03f)' % (mean_auc, std_auc),
             lw=3, alpha=0.8)

    plt.fill_between(mean_fpr,
                     np.maximum(mean_tpr - std_tpr, 0),
                     np.minimum(mean_tpr + std_tpr, 1),
                     color='grey', alpha=0.3, label='± 1 std. dev.')

    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', lw=2)
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title(f'{resampler_name} Mean ROC Curve with {n_splits}-Fold CV', fontsize=16)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True)
    plt.show()


    plt.figure(figsize=(10, 8))
    mean_precision = np.mean(precisions, axis=0)
    std_precision = np.std(precisions, axis=0)
    mean_ap = np.mean(aps)
    std_ap = np.std(aps)

    plt.plot(mean_recall, mean_precision, color='darkorange',
             label=r'Mean PR (AP = %0.3f ± %0.03f)' % (mean_ap, std_ap),
             lw=3, alpha=0.8)

    plt.fill_between(mean_recall,
                     np.maximum(mean_precision - std_precision, 0),
                     np.minimum(mean_precision + std_precision, 1),
                     color='grey', alpha=0.3, label='± 1 std. dev.')

    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel('Recall', fontsize=14)
    plt.ylabel('Precision', fontsize=14)
    plt.title(f'{resampler_name} Mean Precision-Recall Curve with {n_splits}-Fold CV', fontsize=16)
    plt.legend(loc="lower left", fontsize=12)
    plt.grid(True)
    plt.show()


    feature_importances['importance_mean'] = feature_importances.loc[:, feature_importances.columns != 'feature'].mean(axis=1)
    feature_importances = feature_importances.sort_values(by='importance_mean', ascending=False)

    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance_mean', y='feature', data=feature_importances.head(15), palette='viridis')
    plt.title(f'{resampler_name} Mean Feature Importance (Top 15)', fontsize=16)
    plt.xlabel('Mean Importance', fontsize=14)
    plt.ylabel('Feature', fontsize=14)
    plt.tight_layout()
    plt.show()


    if not shap_importances_class1.empty:
        shap_importances_class1['shap_importance_mean'] = shap_importances_class1.loc[:, shap_importances_class1.columns != 'feature'].mean(axis=1)
        shap_importances_class1 = shap_importances_class1.sort_values(by='shap_importance_mean', ascending=False)

        plt.figure(figsize=(12, 8))
        sns.barplot(x='shap_importance_mean', y='feature', data=shap_importances_class1.head(15), palette='magma')
        plt.title(f'{resampler_name} Mean SHAP Feature Importance for Class 1 (Top 15)', fontsize=16)
        plt.xlabel('Mean |SHAP Value|', fontsize=14)
        plt.ylabel('Feature', fontsize=14)
        plt.tight_layout()
        plt.show()
    else:
        print("No SHAP class 1 importance to plot.")


    summary_df = pd.DataFrame(fold_metrics)


    precs_0, recs_0, f1s_0 = [], [], []
    precs_1, recs_1, f1s_1 = [], [], []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        X_test_fold = X.iloc[test_idx]
        y_test_fold = y.iloc[test_idx]
        y_proba_fold = model.predict_proba(X_test_fold)[:, 1]
        thresh = summary_df.loc[summary_df['fold'] == fold, 'threshold'].values[0]
        y_pred_fold = (y_proba_fold >= thresh).astype(int)

        report = classification_report(y_test_fold, y_pred_fold, output_dict=True)
        if '0' in report:
            precs_0.append(report['0']['precision'])
            recs_0.append(report['0']['recall'])
            f1s_0.append(report['0']['f1-score'])
        if '1' in report:
            precs_1.append(report['1']['precision'])
            recs_1.append(report['1']['recall'])
            f1s_1.append(report['1']['f1-score'])

    avg_report = {
        'class_0': {
            'precision': np.mean(precs_0),
            'recall': np.mean(recs_0),
            'f1-score': np.mean(f1s_0)
        },
        'class_1': {
            'precision': np.mean(precs_1),
            'recall': np.mean(recs_1),
            'f1-score': np.mean(f1s_1)
        }
    }

    print("\n--- Average classification report across folds ---")
    print(pd.DataFrame(avg_report).T)


    avg_brier = summary_df['brier'].mean()
    print(f"\nAverage Brier Score across folds: {avg_brier:.4f}")

    return summary_df, feature_importances, shap_importances_class1, avg_report, avg_brier

summary_metrics, feat_importances, shap_class1_importances, avg_cls_report, avg_brier = run_lgbm_cv_with_shap(
    dfdrop, 'Key Response Variable ', RandomUnderSampler(random_state=42), "RandomUnderSampler", n_splits=5
)
print(summary_metrics)

