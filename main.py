import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

iris = pd.read_csv('/Users/Magic Systems/Downloads/IRIS.csv')

print(iris.head())
print(iris.describe())
print(iris['species'].value_counts())

sns.pairplot(iris,hue='species',palette='coolwarm')

iris.plot(kind='scatter', x='sepal_length' , y='sepal_width')

sns.set_style("whitegrid")
sns.FacetGrid(iris, hue="species" , height=4) \
   .map(plt.scatter ,"sepal_length", "sepal_width") \
   .add_legend()


sns.set_style("whitegrid")
sns.pairplot(iris ,hue='species', height=3 )

plt.show()

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(iris.drop('species',axis=1))
scaled_features = scaler.transform(iris.drop('species',axis=1))
df_feat = pd.DataFrame(scaled_features,columns=iris.columns[:-1])
print(df_feat.head())

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(scaled_features,iris['species'],
                                                    test_size=0.30)


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)

pred = knn.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))

error_rate = []
for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))


plt.figure(figsize=(10, 6))
plt.plot(range(1, 40), error_rate, color='blue', linestyle='dashed', marker='o',
             markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')

plt.show()

knn = KNeighborsClassifier(n_neighbors=17)

knn.fit(X_train,y_train)
pred = knn.predict(X_test)

print('WITH K=5')
print('\n')
print(confusion_matrix(y_test,pred))
print('\n')
print(classification_report(y_test,pred))


from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
result = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(result)
result1 = classification_report(y_test, y_pred)
print("Classification Report:",)
print (result1)
result2 = accuracy_score(y_test,y_pred)
print("Accuracy:",result2)


X_new = [[3, 5, 4, 2], [5, 4, 3, 2]]
print(knn.predict(X_new))

print(knn.predict([[3, 5, 4, 2]]))








