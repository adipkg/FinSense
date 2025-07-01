import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Read the CSV file
df = pd.read_csv('table.csv')

# Step 2: Convert 'Date' column to datetime
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)

# Step 3: Plot
plt.figure(figsize=(12, 7))

plt.plot(df['Date'], df['Balanced'], marker='o', label='Balanced Fund')
plt.plot(df['Date'], df['Growth'], marker='o', label='Growth Fund')
plt.plot(df['Date'], df['Protector'], marker='o', label='Protector Fund')
plt.plot(df['Date'], df['Secure'], marker='o', label='Secure Fund')

plt.xlabel('Date')
plt.ylabel('Fund Value')
plt.title('Fund Values Over Time')
plt.grid(True)
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()