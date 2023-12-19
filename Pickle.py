import numpy as np
import pickle 
loaded_model=pickle.load(open('/Users/shivanshmahajan/Desktop/DataScinece/Machine Learning/Projects/diabetes.svc/trained_model.sav','rb'))

input_data=(5,145,72,19,195,25.8,0.587,100)
input_data=np.asarray(input_data)
input_data_reshaped=input_data.reshape(1,-1)

prediction=loaded_model.predict(input_data_reshaped)

print(prediction)
if(prediction==0):
    print('the person is not diabetic')
else:
    print('the person is  diabetic')

    