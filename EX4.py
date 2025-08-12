import pandas as pd
from sklearn.tree import DecisionTreeClassifier,plot_tree
import matplotlib.pyplot as plt
data=pd.read_csv('ex4.csv')
df=pd.DataFrame(data)
x=df[['student_hrs','attendance']]
y=df['result']
clf=DecisionTreeClassifier(criterion='entropy',random_state=0)
clf.fit(x,y)
plt.figure(figsize=(4,6))
plot_tree(clf,feature_names=['study_hrs','attendance'],class_names=['0','1'],filled=True)
plt.show()
new=[[5,85]]
pred=clf.predict(new)
print('predict for new students:','0'if pred[0]==1 else '0')
