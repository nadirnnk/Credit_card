import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
import joblib
import os
import numpy as np
from scipy.stats import norm


def preprocess_data(input_path, output_path, scaler_path):
    # Load dataset
    df = pd.read_csv(input_path)

    # Drop NA values if any (shouldn’t be in this dataset)
    df.dropna(inplace=True)

    # Scale features
    scaler = RobustScaler()

    df['scaled_amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1,1))
    df['scaled_time'] = scaler.fit_transform(df['Time'].values.reshape(-1,1))
    df.drop(['Time','Amount'], axis=1, inplace=True)

    scaled_amount = df['scaled_amount']
    scaled_time = df['scaled_time']

    df.drop(['scaled_amount', 'scaled_time'], axis=1, inplace=True)
    df.insert(0, 'scaled_amount', scaled_amount)
    df.insert(1, 'scaled_time', scaled_time)

    # Amount and Time are Scaled!
    print(df.head())

    # Save the scaler for later use (in API)
    joblib.dump(scaler, scaler_path)




    # Save processed dataset
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Processed data saved to {output_path}")


def undersample_data(inpath, outpath):

    df = pd.read_csv(inpath)

    df = df.sample(frac=1)
    # amount of fraud classes 492 rows.
    fraud_df = df.loc[df['Class'] == 1]
    non_fraud_df = df.loc[df['Class'] == 0][:492]

    normal_distributed_df = pd.concat([fraud_df, non_fraud_df])

    # Shuffle dataframe rows
    new_df = normal_distributed_df.sample(frac=1, random_state=42)

    print(new_df.head())

    # Save processed dataset
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    new_df.to_csv(outpath, index=False)
    print(f"Processed data saved to {outpath}")

def anomaly(inpath, outpath):
    new_df = pd.read_csv(inpath)
    # Check for anomalies in the dataset
    v14_fraud = new_df['V14'].loc[new_df['Class'] == 1].values
    q25, q75 = np.percentile(v14_fraud, 25), np.percentile(v14_fraud, 75)
    print('Quartile 25: {} | Quartile 75: {}'.format(q25, q75))
    v14_iqr = q75 - q25
    print('iqr: {}'.format(v14_iqr))

    v14_cut_off = v14_iqr * 1.5
    v14_lower, v14_upper = q25 - v14_cut_off, q75 + v14_cut_off
    print('Cut Off: {}'.format(v14_cut_off))
    print('V14 Lower: {}'.format(v14_lower))
    print('V14 Upper: {}'.format(v14_upper))

    outliers = [x for x in v14_fraud if x < v14_lower or x > v14_upper]
    print('Feature V14 Outliers for Fraud Cases: {}'.format(len(outliers)))
    print('V10 outliers:{}'.format(outliers))

    new_df = new_df.drop(new_df[(new_df['V14'] > v14_upper) | (new_df['V14'] < v14_lower)].index)
    print('----' * 44)

    # -----> V12 removing outliers from fraud transactions
    v12_fraud = new_df['V12'].loc[new_df['Class'] == 1].values
    q25, q75 = np.percentile(v12_fraud, 25), np.percentile(v12_fraud, 75)
    v12_iqr = q75 - q25

    v12_cut_off = v12_iqr * 1.5
    v12_lower, v12_upper = q25 - v12_cut_off, q75 + v12_cut_off
    print('V12 Lower: {}'.format(v12_lower))
    print('V12 Upper: {}'.format(v12_upper))
    outliers = [x for x in v12_fraud if x < v12_lower or x > v12_upper]
    print('V12 outliers: {}'.format(outliers))
    print('Feature V12 Outliers for Fraud Cases: {}'.format(len(outliers)))
    new_df = new_df.drop(new_df[(new_df['V12'] > v12_upper) | (new_df['V12'] < v12_lower)].index)
    print('Number of Instances after outliers removal: {}'.format(len(new_df)))
    print('----' * 44)


    # Removing outliers V10 Feature
    v10_fraud = new_df['V10'].loc[new_df['Class'] == 1].values
    q25, q75 = np.percentile(v10_fraud, 25), np.percentile(v10_fraud, 75)
    v10_iqr = q75 - q25

    v10_cut_off = v10_iqr * 1.5
    v10_lower, v10_upper = q25 - v10_cut_off, q75 + v10_cut_off
    print('V10 Lower: {}'.format(v10_lower))
    print('V10 Upper: {}'.format(v10_upper))
    outliers = [x for x in v10_fraud if x < v10_lower or x > v10_upper]
    print('V10 outliers: {}'.format(outliers))
    print('Feature V10 Outliers for Fraud Cases: {}'.format(len(outliers)))
    new_df = new_df.drop(new_df[(new_df['V10'] > v10_upper) | (new_df['V10'] < v10_lower)].index)
    print('Number of Instances after outliers removal: {}'.format(len(new_df)))

    # Save processed dataset
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    new_df.to_csv(outpath, index=False)
    print(f"Processed data saved to {outpath}")



if __name__ == "__main__":
    preprocess_data(
        input_path="data/raw/creditcard.csv",
        output_path="data/processed/processed_data.csv",
        scaler_path="src/data_prep/scaler.joblib"
    )
    undersample_data(inpath="data/processed/processed_data.csv",
                     outpath="data/processed/undersampled_data.csv"
                     )
    anomaly(inpath="data/processed/undersampled_data.csv",
             outpath="data/processed/anomaly_data.csv"
             )
