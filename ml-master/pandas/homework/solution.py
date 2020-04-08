#1. Import pandas as pd.
import pandas as pd
import os
#2. Read Salaries.csv as a dataframe called sal.
_dir = os.path.dirname(os.path.realpath(__file__))
sal = pd.read_csv(_dir + '/Salaries.csv')
#3. Check the head of the DataFrame.
sal.head(2)
#4. Use the .info() method to find out fow many entries there are.
sal.info()
#5. What is the average BasePay?
sal['BasePay'].mean()
#6. What is the highest amount of OVertimePay in the dataset?
sal['OvertimePay'].max()
#7. What is the job title of JOSEPH DRISCOLL? Note: Use all caps, otherwise you may get an answer that doesn't match up (there is also a lowercase Joseph Driscoll).
sal[sal['EmployeeName']=='JOSEPH DRISCOLL']['JobTitle']
#8. How much does JOSEPH DRISCOLL make (including benefits)?
sal[sal['EmployeeName']=='Joseph Driscoll']['TotalPayBenefits']
#9. What is the name of highest paid person (including benefits)?
sal[sal['TotalPayBenefits'] == sal['TotalPayBenefits'].max()]['EmployeeName']
#sal.loc[sal['TotalPayBenefits'].idxmax()]
sal.iloc[sal['TotalPayBenefits'].idxmax()]

#10. What is hte name of lowest paid person (including benefits)? Do you notice something strange about how much he or she is paid?
sal[sal['TotalPayBenefits']==sal['TotalPayBenefits'].min()]
#11. What was the average (mean) BasePay of all employees per year? (2011-2014) ?
sal.groupby('Year').mean()['BasePay']
#12. How many uniuque job titles are there?
sal['JobTitle'].nunique()
#13. What are the top 5 most common jobs?
sal['JobTitle'].value_counts().head(5)
#14. How many Job Titles were represented by only one person in 2013? (e.g. Job Titles with only one occurence in 2013) ?
sum(sal[sal['Year']==2013]['JobTitle'].value_counts() == 1)
#15. How many people have the word Chief in thier job title? (This is pretty tricky)
def chief_string(title):
    if 'chief' in title.lower().split():
        return True
    else:
        return False

sum(sal['JobTitle'].apply(lambda x: chief_string(x)))

#16. Bonus: Is there a correlation between length of the Job Title string and Salary?
sal['title_len'] = sal['JobTitle'].apply(len)
sal[['TotalPayBenefits', 'title_len']].corr()