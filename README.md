# machine-learning

## Books

### andrew-ng

the Andrew Ng's machine learning course practice

### introduction_to_machine_learning_with_python

Andreas C Müller & Sarah Guido's machine learning book's code

### Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, 2nd Edition,

作者 Aurelien Geron, 由O'Reilly 出版，书号978-1-492-03264-9 [github](https://github.com/ageron/handson-ml2)

## Note

### 精度指标

> TP 真正类 预测为正类且正确（原本是正类）
>
> FP 假正类 预测为正类但错误（原本是负类）
>
> TN 真负类 预测为负类且正确（原本是负类）
>
> FN 假负类 预测为负类但错误（原本是正类）

* 精度 TP / (TP + FP) 预测正类时候预测对的概率 (预测正类的准确率)
* 召回率 TP / (TP + FN) 正类中被正确预测为正类的概率 (样本中正类被包含的概率)
* 假正类率 1 - 真负类率
* 真负类率 被正确预测为负类的负类实例比率 也称为特异度