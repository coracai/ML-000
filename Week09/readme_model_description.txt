*******3个模型：
1. 原始特征模型：   
2. 原始特征+衍生变量（installment_feat）模型：
train['continuous_installment_feat']= train['continuous_installment'] / ((train['continuous_annual_inc']+1) / 12)
test['continuous_installment_feat']= test['continuous_installment'] / ((test['continuous_annual_inc']+1) / 12)
3. 原始特征+CATBOOST ENCODER（只对discrete特征进行CATBOOST ENCODER）模型：

import category_encoders as ce
cat_features = discrete_column
CatBoost_enc = ce.CatBoostEncoder(cols = cat_features)
CatBoost_enc.fit(train[cat_features],train["loan_status"])
train = train.join(CatBoost_enc.transform(train[cat_features]).add_suffix("_count"))
test = test.join(CatBoost_enc.transform(test[cat_features]).add_suffix("_count"))

*******模型结果对比：
1. 原始特征模型：
预测正确的个数：45902   预测正确的比例：0.918040
2. 原始特征+衍生变量（installment_feat）模型：
预测正确的个数：45912   预测正确的比例：0.918240
3.3. 原始特征+CATBOOST ENCODER模型：
预测正确的个数：45917   预测正确的比例：0.918340

模型训练过程：
* 确定特征
* 模型调参：
1. lgb.cv 得到estimator个数（弱分类器数目）
2. GridSearchCV对num_leaves，min_child_samples，bagging_fraction，feature_fraction进行调参，进而得到新的estimator个数
3. 用老师提供的wrapper进行hyperopt调参，并用search_k_fold函数得到最佳参数
4. 用得到的最佳参数，使用train_k_fold来训练并对测试集进行预测，最终的预测结果是对5折交叉验证模型的预测结果取平均
6. 根据得到的预测值，以0.5为区分点，大于0.5的取1，反之取0，得到最终的准确率