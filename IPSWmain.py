import torch
import torch.utils.data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder

device = torch.device('cpu')
import warnings
warnings.filterwarnings("ignore")

def match_dtypes(training_df,testing_df,target_name='TARGET'):
	"""
	This function converts dataframe to match columns in accordance with the 
	training dataframe.
	"""
	for column_name in training_df.drop([target_name],axis=1).columns:
		testing_df[column_name]= testing_df[column_name].astype(train[column_name].dtype)

	return testing_df

#panda transformer does not work here so I wrote a function
#to conververt categorial data with string value to integer values
def Mytransformer(theKey, df):
	names=df[theKey].unique()
	dictName={}
	for c in range(len(names)):
		dictName[names[c]]=c
	trVect=[]
	for f in df[theKey]:
		trVect=trVect+[dictName[f]]
	return np.array(trVect)

def leaningClassifirofSensetive(X_num, SensVector,nEpoches=20):
	N=np.max(SensVector)+1
	Classifier_adv = torch.nn.Sequential(
			# Layer 1 - 117 -> 256
			torch.nn.Linear(X_num.shape[1], 256),
			torch.nn.LeakyReLU(),
			torch.nn.BatchNorm1d(256),
			# Layer 2 - 256 -> 512
			torch.nn.Linear(256, 512),
			torch.nn.LeakyReLU(),
			torch.nn.BatchNorm1d(512),
			# Layer 3 - 512 -> 512
			torch.nn.Linear(512, 1024),
			torch.nn.LeakyReLU(),
			torch.nn.BatchNorm1d(1024),
			# Layer 4 - 512 -> 512
			torch.nn.Linear(1024, 512),
			torch.nn.LeakyReLU(),
			torch.nn.BatchNorm1d(512),
			# Layer 5 - 512 -> 256
			torch.nn.Linear(512, 256),
			torch.nn.LeakyReLU(),
			torch.nn.BatchNorm1d(256),
			# Output
			torch.nn.Linear(256, N),
			).to(device)

	Opt_Classifier_adv = torch.optim.Adam(Classifier_adv.parameters())
	loss_classifier_adv = torch.nn.CrossEntropyLoss()

	X_tensor = torch.tensor(X_num, device=device).double()
	y_tensor = torch.tensor(np.array(SensVector), device=device).double().unsqueeze(1)

	train_loader = torch.utils.data.DataLoader(torch.cat((X_tensor, y_tensor), 1), batch_size=512, shuffle=True)

	for epoch in range(nEpoches):
		loss_avg = 0
		num = 0
		for batch in train_loader:
			x, y = batch[:, 0:-1], batch[:, -1]

			y_pred = Classifier_adv(x.float())

			loss = loss_classifier_adv(y_pred, y.long())
			loss_avg += loss
			num += 1

			Opt_Classifier_adv.zero_grad()
			loss.backward()
			Opt_Classifier_adv.step()
		print("loss: ", (loss_avg/num).item())

	return Classifier_adv

def EvluateModel(modelS, X_num, Target):
	y_pred=[]
	X_tensor = torch.tensor(X_num, device=device).double()
	y_tensor = torch.tensor(np.array(SensVector), device=device).double().unsqueeze(1)

	train_loader = torch.utils.data.DataLoader(torch.cat((X_tensor, y_tensor), 1), batch_size=512, shuffle=True)



	Classifier_adv.eval()
	with torch.no_grad():
		for batch in train_loader:
			x, y = batch[:, 0:-1], batch[:, -1]

			pred = Classifier_adv(x.float())
			pred=pred.numpy()

			y_pred =y_pred+list(np.argmax(pred,1))
        
	y_pred=np.array(y_pred).squeeze()
	print(y_pred.shape)
	from sklearn.metrics import accuracy_score,classification_report,roc_auc_score
	print("The accuracy in general is : ", accuracy_score(np.array(Target),y_pred))
	print("\n")
	print("The classification report is as follows:\n", classification_report(np.array(Target),y_pred))



def redandclean():
	data = pd.read_csv("cleanedata/cleandata.csv")
	
	train=data.copy()

	print('Training Features shape: ', train.shape)

	(train['DAYS_BIRTH']/-365).describe()

	thousand_anomalies = train[(train['DAYS_EMPLOYED']/365>=900) & (train['DAYS_EMPLOYED']/365<=1100)]
	
	#Most anomalies were able to repay on time. But how can they be contrasted in relation to non anomalies?
	# get the index of anomalies and non anomalies
	anomalies_index = pd.Index(thousand_anomalies.index)
	non_anomalies_index = train.index.difference(anomalies_index)
	# get the anomalies records
	non_anomalies = train.iloc[non_anomalies_index]
	# get the anomaly targets
	anomalies_target = thousand_anomalies['TARGET'].value_counts()
	non_anomalies_target = non_anomalies['TARGET'].value_counts()
	# find the default rate for anomalies and non anomalies

	print("Anomalies have a default rate of {}%".format(100*anomalies_target[1]/(anomalies_target[1]+anomalies_target[0])))

	# Create an anomalous flag column
	train['DAYS_EMPLOYED_ANOM'] = train["DAYS_EMPLOYED"] == 365243

	# Replace the anomalous values with nan
	train['DAYS_EMPLOYED'] = train['DAYS_EMPLOYED'].replace({365243: np.nan})
	# Looking at the years employed for anomalies
	from sklearn.preprocessing import Imputer
	# poly_fitting_vars = ['EXT_SOURCE_3', 'EXT_SOURCE_2', 'EXT_SOURCE_1','DAYS_BIRTH']
	imputer = Imputer(missing_values='NaN', strategy='median')
	# train[poly_fitting_vars] = imputer.fit_transform(train[poly_fitting_vars])
	# train[poly_fitting_vars].shape

	# from sklearn.preprocessing import PolynomialFeatures
	# poly_feat = PolynomialFeatures(degree=4)
	# poly_interaction_train = poly_feat.fit_transform(train[poly_fitting_vars])

	# train['DIR'] = train['AMT_CREDIT']/train['AMT_INCOME_TOTAL']
	# train['AIR'] = train['AMT_ANNUITY']/train['AMT_INCOME_TOTAL']
	# train['ACR'] = train['AMT_ANNUITY']/train['AMT_CREDIT']
	# train['DAR'] = train['DAYS_EMPLOYED']/train['DAYS_BIRTH']
	
	sensetiveFeatures=['CODE_GENDER', 'NAME_INCOME_TYPE','NAME_FAMILY_STATUS','OCCUPATION_TYPE','ORGANIZATION_TYPE']
	X_num=train.copy()

	X_num=X_num.drop(columns=sensetiveFeatures)
	target = X_num['TARGET']
	X_num=X_num.drop(columns=['TARGET','Unnamed: 0'])
	X_num = pd.get_dummies(X_num)
	X_num = imputer.fit_transform(X_num)
	SenstiveData=train[sensetiveFeatures].copy()

	a=train.groupby(sensetiveFeatures).count()
	# print(a['Unnamed: 0'],np.max(a['Unnamed: 0']),np.min(a['Unnamed: 0']), np.mean(a['Unnamed: 0']),np.std(a['Unnamed: 0']) )
	#5262 1 24.483358459932738 136.4728162223554
	
	for i in range(1):	
		SensVector=Mytransformer(sensetiveFeatures[i], SenstiveData)
		model=leaningClassifirofSensetive(X_num, SensVector,nEpoches=20)

	

redandclean()











