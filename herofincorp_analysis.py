"""
Hero FinCorp - Loan portfolio & customer behavior analysis
File: herofincorp_analysis.py


Expected CSV files (place them in the same working directory):
    - customers.csv
    - applications.csv
    - transactions.csv

"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# File paths (adjust if needed)
CUSTOMERS_CSV = "customers.csv"
APPLICATIONS_CSV = "applications.csv"
TRANSACTIONS_CSV = "transactions.csv"

def assert_files_exist(files):
    missing = [f for f in files if not os.path.exists(f)]
    if missing:
        print("ERROR: The following required files are missing in the working directory:")
        for m in missing:
            print(" -", m)
        sys.exit(1)

def load_data():
    print("Loading datasets...")
    customers = pd.read_csv(CUSTOMERS_CSV)
    applications = pd.read_csv(APPLICATIONS_CSV)
    transactions = pd.read_csv(TRANSACTIONS_CSV)
    print("Loaded: customers {}, applications {}, transactions {}".format(
        customers.shape, applications.shape, transactions.shape))
    return customers, applications, transactions

def basic_missing_summary(df, name, n=10):
    miss = df.isnull().sum().sort_values(ascending=False)
    pct = (miss / len(df) * 100).round(2)
    summary = pd.concat([miss, pct], axis=1, keys=['missing_count','missing_pct'])
    print(f"\\nMissing in {name} (top {n}):\\n{summary.head(n)}")
    return summary

def prepare_transactions(transactions):
    # Standardize date column
    if 'Transaction_Date' in transactions.columns:
        transactions['Transaction_Date'] = pd.to_datetime(transactions['Transaction_Date'], errors='coerce')
    return transactions

def build_loan_aggregation(transactions):
    print("Building loan-level aggregation from transactions...")
    agg = transactions.groupby('Loan_ID').agg(
        total_paid=('Amount','sum'),
        n_transactions=('Transaction_ID','count'),
        last_remaining_balance=('Remaining_Balance','last'),
        avg_overdue_fee=('Overdue_Fee','mean'),
        first_txn_date=('Transaction_Date','min'),
        last_txn_date=('Transaction_Date','max')
    ).reset_index()
    print("Loan aggregation shape:", agg.shape)
    return agg

def merge_datasets(loan_agg, applications, customers):
    print("Merging loan aggregates with applications and customers (if available)...")
    merged = loan_agg.merge(applications, how='left', on='Loan_ID')
    # Ensure Customer_ID exists
    if 'Customer_ID' not in merged.columns and 'Customer_ID_x' in merged.columns:
        merged.rename(columns={'Customer_ID_x':'Customer_ID'}, inplace=True)
    if 'Customer_ID' in merged.columns:
        merged = merged.merge(customers, how='left', on='Customer_ID')
    print("Merged shape:", merged.shape)
    return merged

def compute_proxy_default(merged, days_cutoff=180):
    # Heuristic proxy default: last transaction older than cutoff and remaining_balance > 0
    merged['first_txn_date'] = pd.to_datetime(merged['first_txn_date'], errors='coerce')
    merged['last_txn_date'] = pd.to_datetime(merged['last_txn_date'], errors='coerce')
    global_max = merged['last_txn_date'].max()
    if pd.isna(global_max):
        print("Warning: Global max of last_txn_date is NaT. Proxy default cannot be computed reliably.")
        merged['Proxy_Default_Flag'] = 0
        return merged
    cutoff = global_max - pd.Timedelta(days=days_cutoff)
    merged['Proxy_Default_Flag'] = ((merged['last_txn_date'] < cutoff) & (merged['last_remaining_balance'] > 0)).astype(int)
    print("Proxy defaults (heuristic) count:", merged['Proxy_Default_Flag'].sum(), "out of", len(merged))
    return merged

def correlation_analysis(merged, out_png="correlation_heatmap.png"):
    numeric_cols = ['Credit_Score','Proxy_Default_Flag','total_paid','last_remaining_balance','avg_overdue_fee','n_transactions']
    present = [c for c in numeric_cols if c in merged.columns]
    if len(present) < 2:
        print("Not enough numeric columns present for correlation analysis. Present numeric:", present)
        return None
    corr = merged[present].corr()
    print("\\nCorrelation matrix:\\n", corr)
    # Save heatmap
    fig, ax = plt.subplots(figsize=(6,5))
    cax = ax.imshow(corr, interpolation='nearest', aspect='auto')
    ax.set_xticks(range(len(corr.columns))); ax.set_xticklabels(corr.columns, rotation=45, ha='right')
    ax.set_yticks(range(len(corr.index))); ax.set_yticklabels(corr.index)
    fig.colorbar(cax)
    plt.title("Correlation matrix (proxy features)")
    plt.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print("Saved correlation heatmap to", out_png)
    return corr

def segment_customers(merged, out_csv="segment_stats.csv"):
    seg = merged.copy()
    if 'Credit_Score' in seg.columns:
        seg['credit_band'] = pd.cut(seg['Credit_Score'], bins=[-1,300,580,670,740,800,1000],
                                    labels=['Very Low','Low','Fair','Good','Very Good','Excellent'])
    else:
        seg['credit_band'] = 'Unknown'
    if 'Annual_Income' in seg.columns:
       
        seg['Annual_Income'] = seg['Annual_Income'].fillna(seg['Annual_Income'].median())
        try:
            seg['income_band'] = pd.qcut(seg['Annual_Income'], 4, labels=['Low','Medium','High','Very High'])
        except Exception:
            seg['income_band'] = 'Unknown'
    else:
        seg['income_band'] = 'Unknown'
    seg_stats = seg.groupby(['credit_band','income_band']).agg(
        customers=('Customer_ID','nunique'),
        proxy_default_rate=('Proxy_Default_Flag','mean'),
        avg_remaining_balance=('last_remaining_balance','mean')
    ).reset_index()
    seg_stats.to_csv(out_csv, index=False)
    print("Saved segment stats to", out_csv)
    return seg_stats

def monthly_default_trend(merged, out_png="monthly_proxy_defaults.png"):
    loan_last = merged[['Loan_ID','last_txn_date','Proxy_Default_Flag']].copy()
    loan_last['month'] = loan_last['last_txn_date'].dt.to_period('M').astype(str)
    monthly = loan_last.groupby('month')['Proxy_Default_Flag'].sum().reset_index()
    # plot
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(monthly['month'], monthly['Proxy_Default_Flag'], marker='o')
    ax.set_xticks(monthly['month'][::max(1,len(monthly)//12)])
    ax.set_xticklabels(monthly['month'][::max(1,len(monthly)//12)], rotation=45, ha='right')
    ax.set_title("Monthly Proxy Defaults (count)")
    ax.set_ylabel("Number of Proxy Defaults")
    plt.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print("Saved monthly proxy defaults chart to", out_png)
    return monthly

def application_insights(applications):
    if 'Approval_Status' in applications.columns:
        print("\\nApplication approval counts:")
        print(applications['Approval_Status'].value_counts())
    if 'Rejection_Reason' in applications.columns:
        print("\\nTop rejection reasons:")
        print(applications['Rejection_Reason'].value_counts().head(10))

def save_samples(merged, out_csv="merged_sample_head200.csv"):
    merged.head(200).to_csv(out_csv, index=False)
    print("Saved merged sample head to", out_csv)

def main():
    required = [CUSTOMERS_CSV, APPLICATIONS_CSV, TRANSACTIONS_CSV]
    assert_files_exist(required)
    customers, applications, transactions = load_data()

    # Basic summaries
    basic_missing_summary(customers, "customers")
    basic_missing_summary(applications, "applications")
    basic_missing_summary(transactions, "transactions")

    transactions = prepare_transactions(transactions)
    loan_agg = build_loan_aggregation(transactions)
    merged = merge_datasets(loan_agg, applications, customers)
    merged = compute_proxy_default(merged, days_cutoff=180)

    corr = correlation_analysis(merged)
    seg_stats = segment_customers(merged)
    monthly = monthly_default_trend(merged)
    application_insights(applications)
    save_samples(merged)

    print("\\nAnalysis complete. Check the saved output files in the working directory.")

if __name__ == "__main__":
    main()
