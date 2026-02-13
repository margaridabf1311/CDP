import pandas as pd
import numpy as np


logon_df = pd.read_csv('logon.csv')       
device_df = pd.read_csv('device.csv')    
email_df = pd.read_csv('email.csv')       
file_df = pd.read_csv('file.csv')   

print(logon_df.head())
print(device_df.head())
print(email_df.head())
print(file_df.head())

# Informações gerais
print(logon_df.info())
print(device_df.info())
print(logon_df['user'].value_counts().head())

logon_df['date'] = pd.to_datetime(logon_df['date'],dayfirst=True, format='%m/%d/%Y %H:%M:%S')

activity_counts = logon_df.groupby(['user', 'activity']).size().unstack(fill_value=0)
activity_counts.columns = [f'num_{col.lower()}' for col in activity_counts.columns]

print(activity_counts.head())


def preprocess_log(df, date_col='date'):
    df = df.drop_duplicates(subset='id')  # remover duplicados
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')  # datas inválidas -> NaT
    df = df.dropna(subset=[date_col])  # remover linhas sem data
    return df

# ----------------------------
# Carregar dados e pré-processar
# ----------------------------
logon_df = preprocess_log(pd.read_csv('logon.csv'))
device_df = preprocess_log(pd.read_csv('device.csv'))
email_df = preprocess_log(pd.read_csv('email.csv'))
file_df = preprocess_log(pd.read_csv('file.csv'))

# ----------------------------
# PC features
# ----------------------------
pc_features = logon_df.groupby(['user', 'activity']).size().unstack(fill_value=0)
pc_features.columns = [f'pc_{col.lower()}' for col in pc_features.columns]

# Logins fora do horário normal (00h-06h)
logon_df['hour'] = logon_df['date'].dt.hour
night_logins = logon_df[logon_df['hour'].between(0,6)]
night_counts = night_logins.groupby('user').size()
pc_features['pc_logins_night'] = night_counts
pc_features['pc_logins_night'] = pc_features['pc_logins_night'].fillna(0)

# ----------------------------
# Device features
# ----------------------------
device_features = device_df.groupby(['user', 'activity']).size().unstack(fill_value=0)
device_features.columns = [f'device_{col.lower()}' for col in device_features.columns]

# ----------------------------
# Email features
# ----------------------------
email_df['attachments'] = email_df['attachments'].fillna(0)
email_features = email_df.groupby('user').agg(
    num_emails=('id', 'count'),
    num_emails_with_attachments=('attachments', 'sum')
)
email_features['ratio_emails_with_attachments'] = email_features['num_emails_with_attachments'] / (email_features['num_emails'] + 1)

# ----------------------------
# File/download features
# ----------------------------
file_df['size'] = file_df['size'].fillna(0)
file_features = file_df.groupby('user').agg(
    num_files_downloaded=('id', 'count'),
    total_file_size=('size', 'sum')
)
file_features['files_per_logon'] = file_features['num_files_downloaded'] / (pc_features.get('pc_logon', pd.Series(1)) + 1)

# ----------------------------
# Combinar todas as features numa tabela final
# ----------------------------
features = pc_features.join(device_features, how='left').fillna(0)
features = features.join(email_features, how='left').fillna(0)
features = features.join(file_features, how='left').fillna(0)

# ----------------------------
# Preenchimento final de NaNs (caso ainda existam)
# ----------------------------
features = features.fillna(0)

# ----------------------------
# Visualizar tabela final
# ----------------------------
print(features.head())
print("\nNúmero de utilizadores:", features.shape[0])
print("Número de features:", features.shape[1])