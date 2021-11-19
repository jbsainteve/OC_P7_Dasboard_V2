import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import shap
import plotly.express as px
# Set the style of plots
plt.style.use('fivethirtyeight')


# Drop sécurisé d'une liste de colonnes
# -------------------------------------
def drop_columns(columns_to_drop, dataset):
	for column in columns_to_drop:
		try:
			dataset.drop(column, axis=1, inplace=True)
		except:
			print('colonne {} absente du jeu de donnée'.format(column))
			
		return (dataset)

@st.cache(allow_output_mutation=True)
def load_data():
	app_test = pd.read_csv('application_test.csv')
	app_test_domain = pd.read_csv('app_test_domain.csv')

	app_train = pd.read_csv('application_train_reduit.csv')
	#app_train_domain = pd.read_csv('app_train_domain.csv')

	app_test = drop_columns(['Unnamed: 0'], app_test)
	app_train = drop_columns(['Unnamed: 0'], app_train)
	app_test_domain = drop_columns(['Unnamed: 0'], app_test_domain)
	#app_train_domain = drop_columns(['Unnamed: 0'], app_train_domain)
	app_train['AGE']=app_train["DAYS_BIRTH"]/-365

	with open('test_domain.bin', 'rb') as f_in:
		test_domain = pickle.load(f_in)
		f_in.close()

	#return app_train, app_test, app_train_domain, app_test_domain, test_domain
	return app_train, app_test, app_test_domain, test_domain

@st.cache(allow_output_mutation=True)
def load_model():
	'''loading the trained model'''
	# with open('model_clf_xgb.bin', 'rb') as f_in:
	# 	clf_xgb = pickle.load(f_in)
	# 	f_in.close()
			
	with open('model_clf_rf.bin', 'rb') as f_in:
		rf = pickle.load(f_in)
		f_in.close()

	# return clf_xgb, clf_rf
	return rf
	
@st.cache
def load_income_population(sample):
	df_income = pd.DataFrame(sample["AMT_INCOME_TOTAL"])
	df_income = df_income.loc[df_income['AMT_INCOME_TOTAL'] < 200000, :]
	return df_income

@st.cache
def load_infos_gen(data):
	lst_infos = [data.shape[0],
					 round(data["AMT_INCOME_TOTAL"].mean(), 2),
					 round(data["AMT_CREDIT"].mean(), 2)]

	nb_credits = lst_infos[0]
	rev_moy = lst_infos[1]
	credits_moy = lst_infos[2]

	targets = data.TARGET.value_counts()

	return nb_credits, rev_moy, credits_moy, targets

def identite_client(data, id):
	data_client = data[data.index == int(id)]
	return data_client

#@st.cache
def make_prediction(num, data, model):
	item = data[num]
	#print (type(test_domain))
	#print (type(item))
	#print (test_domain.shape)
	item = item.reshape(1,243)
	#print (item.shape)
	# test_sample = test_domain[num] # pour Lime

	xgb_pred = model.predict_proba(item)
	return (xgb_pred)    


def main():


	# Loading data……
	#app_train, app_test, app_train_domain, app_test_domain, test_domain = load_data()
	app_train, app_test, app_test_domain, test_domain = load_data()

	# on refait un DataFrame
	test_domain_df = pd.DataFrame (data=test_domain, columns=app_test_domain.columns)

	id_client = 11
	model_rf = load_model()

	#######################################
	# SIDEBAR
	#######################################

	#Title display
	html_temp = """
	<div style="background-color: tomato; margin:auto, padding:10px; border-radius:10px">
	<h1 style="color: white; text-align:center">Dashboard</h1>
	</div>
	<p style="font-size: 20px; font-weight: bold; text-align:center">Aide à la décision d'octroi de crédits</p>
	"""
	st.markdown(html_temp, unsafe_allow_html=True)

	#Customer ID selection
	st.sidebar.header("**Informations générales**")

	# Input client number
	chk_id = st.sidebar.number_input('Numéro du client', min_value=1, max_value=48000)

	#Loading general info
	nb_credits, rev_moy, credits_moy, targets = load_infos_gen(app_train)


	### Display of information in the sidebar ###
	#Number of loans in the sample
	st.sidebar.markdown("<u>Nombre de prêts en base :</u>", unsafe_allow_html=True)
	st.sidebar.text(nb_credits)

	#Average income
	st.sidebar.markdown("<u>Revenus moyens ($US) :</u>", unsafe_allow_html=True)
	st.sidebar.text(rev_moy)

	#AMT CREDIT
	st.sidebar.markdown("<u>Moyens des prêts ($US) :</u>", unsafe_allow_html=True)
	st.sidebar.text(credits_moy)
	
	#PieChart
	#st.sidebar.markdown("<u>......</u>", unsafe_allow_html=True)
	fig, ax = plt.subplots(figsize=(5,5))
	plt.pie(targets, explode=[0, 0.1], labels=['No default', 'Default'], autopct='%1.1f%%', startangle=90)
	st.sidebar.pyplot(fig)
		

	#######################################
	# HOME PAGE - MAIN CONTENT
	#######################################
	#Display Customer ID from Sidebar
	st.write("Numéro du client :", chk_id)

	#Customer information display : Customer Gender, Age, Family status, Children, …
	st.header("**Informations sur le clients**")

	st.write("**Genre : **", app_test.loc[chk_id,'CODE_GENDER'])
	st.write("**Status familial : **", app_test.loc[chk_id,"NAME_FAMILY_STATUS"])
	st.write("**nombre d'enfants : **{:.0f}".format(app_test.loc[chk_id,"CNT_CHILDREN"]))

	if st.checkbox("Montrer les informations détaillées du client ?"):

		#Age distribution plot
		data_age = (app_train.DAYS_BIRTH / -365) #load_age_population(data)
		fig, ax = plt.subplots(figsize=(10, 5))
		sns.histplot(data_age, edgecolor = 'k', color="goldenrod", bins=20)
		ax.axvline(int(app_test.loc[chk_id,"DAYS_BIRTH"] / -365), color="green", linestyle='--')
		ax.set(title='Age du client', xlabel='Age', ylabel='')
		st.pyplot(fig)
		
		st.subheader("*Revenus (USD)*")
		st.write("**Revenu total : **{:.0f}".format(app_test.loc[chk_id,"AMT_INCOME_TOTAL"]))
		st.write("**Crédit montant : **{:.0f}".format(app_test.loc[chk_id,"AMT_CREDIT"]))
		st.write("**Crédit annuités : **{:.0f}".format(app_test.loc[chk_id,"AMT_ANNUITY"]))
		st.write("**Prix du biens acheté : **{:.0f}".format(app_test.loc[chk_id,"AMT_GOODS_PRICE"]))
		
		data_income = load_income_population(app_train)
		fig, ax = plt.subplots(figsize=(10, 5))
		sns.histplot(data_income["AMT_INCOME_TOTAL"], edgecolor = 'k', color="goldenrod", bins=10)
		ax.axvline(int(app_test.loc[chk_id,"AMT_INCOME_TOTAL"]), color="green", linestyle='--')
		ax.set(title='Revenus du client', xlabel='Income (USD)', ylabel='')
		st.pyplot(fig)
		
	else:
		st.markdown("<i>…</i>", unsafe_allow_html=True)

	# Customer solvability display
	st.header("**Données du client**")
	prediction = make_prediction(chk_id, test_domain, model_rf)
	st.write("**Probabilité de défaut de paiement : **", round(prediction[0,1],4))

	st.markdown("<u>Données du client</u>", unsafe_allow_html=True)
	st.table(identite_client(app_test, chk_id))
	
	#Feature importance / description
	if st.checkbox("Afficher l'importance des données selon Shap ?"):
		shap.initjs()
		X = test_domain_df.loc[chk_id,:]
		number = st.slider("Choisir le nombre de features …", 0, 20, 5)

		fig, ax = plt.subplots(figsize=(10, 10))
		explainer = shap.TreeExplainer(model_rf)
		shap_values = explainer.shap_values(test_domain_df.iloc[0:10,:])
		#shap.summary_plot(shap_values[0], X, plot_type ="bar", max_display=number, color_bar=False, plot_size=(5, 5))
		#shap.summary_plot(shap_values[0], test_domain_df.iloc[0:100,:], plot_type ="bar", max_display=number, color_bar=False, plot_size=(5, 5))
		shap.summary_plot(shap_values[1], test_domain_df.iloc[0:10,:], max_display=number)
		#shap.summary_plot(shap_values[1], test_domain_df.iloc[0:10,:], plot_type ="bar", max_display=number, color_bar=False, plot_size=(5, 5))
		st.pyplot(fig)
		
	else:
		st.markdown("<i>…</i>", unsafe_allow_html=True)
			
	#Filtre de comparaison avec des clients similaires
	if st.checkbox("Afficher les filtres de comparaison ?"):
		submit = 0

		my_form = st.form(key = "filtre")

		seuil_age = my_form.slider("Seuil de filtrage autour de l'âge du client en années", min_value=0, max_value = 73 )
		seuil_sexe = my_form.selectbox('Filtrage sur le sexe', ['M', 'F', 'Tout'], key=2)
		seuil_revenu = my_form.number_input(label = "Seuil de filtrage (+ ou -) autour du revenu du client en USD")

		filtre_age = my_form.checkbox("Filtrer sur l'âge ?")
		filtre_genre = my_form.checkbox("Filtrer sur le genre ?")
		filtre_revenu = my_form.checkbox("Filtrer sur le revenu ?")

		submit = my_form.form_submit_button(label = "Valider")

		age_client = int(app_test.loc[chk_id,"DAYS_BIRTH"] / -365)
		sexe_client = app_test.loc[chk_id,'CODE_GENDER']
		revenu_client = int(app_test.loc[chk_id,"AMT_INCOME_TOTAL"])
		app_filtre=app_train

		if (filtre_age):
			mask1 = ((app_train['AGE'] >= age_client-seuil_age) & (app_train['AGE'] <= age_client+seuil_age))
			app_filtre = app_filtre[mask1]

		if (filtre_genre):
			if (seuil_sexe=='M' or seuil_sexe=='F'):
				mask2 = (app_train['CODE_GENDER'] == seuil_sexe)
				app_filtre = app_filtre[mask2]

		if (filtre_revenu):
			mask3 = ((app_train['AMT_INCOME_TOTAL'] >= revenu_client-seuil_revenu) & (app_train['AMT_INCOME_TOTAL'] <= revenu_client+seuil_revenu))
			app_filtre = app_filtre[mask3]

		st.write("**Nombre de clients répondant aux critères : **", app_filtre.shape[0])
		targets = app_filtre.TARGET.value_counts()
		if (len(targets) == 0):
			targets[0]=0
			targets[1]=0
		if (len(targets) == 1):
			targets[1]=0

		st.write("**Nombre de clients sans défaut : **", targets[0])
		st.write("**Nombre de clients en défaut : **", targets[1])

		if (submit):
			#PieChart
			fig, ax = plt.subplots(figsize=(5,5))
			plt.pie(targets, explode=[0, 0.1], labels=['No default', 'Default'], autopct='%1.1f%%', startangle=90)
			st.pyplot(fig)

		
	else:
		st.markdown("<i>…</i>", unsafe_allow_html=True)
		  
	st.markdown('***')
	st.markdown("Merci d'avoir visité cette page développée dans le cadre du projet 7 d'OpenClassRoom")


if __name__ == '__main__':
	main()