import csv
import pandas as pd
df = pd.read_csv('Attendance.csv')
df.drop_duplicates(subset=['Name'],inplace=True)
print(df.columns)
df = df.reset_index(drop=True)
print(df)
df.to_csv("CleanAttendance.csv")