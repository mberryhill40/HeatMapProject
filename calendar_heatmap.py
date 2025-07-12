import os
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.colorbar import ColorbarBase
from datetime import datetime, timedelta
import calendar
import numpy as np

# --- Configuration ---
SAVE_DIR = r"D:\Users\Mberr\OneDrive\Desktop\HistData"
os.makedirs(SAVE_DIR, exist_ok=True)

plt.rcParams.update({
    'font.size': 9,
    'font.family': 'DejaVu Sans',
    'axes.titlesize': 12,
    'axes.titleweight': 'bold'
})

# --- Color Map Setup ---
cdict = {
    'red':   ((0.0, 0.55, 0.55),
              (0.5, 1.0, 1.0),
              (1.0, 0.0, 0.0)),
    'green': ((0.0, 0.0, 0.0),
              (0.5, 1.0, 1.0),
              (1.0, 0.55, 0.55)),
    'blue':  ((0.0, 0.0, 0.0),
              (0.5, 1.0, 1.0),
              (1.0, 0.0, 0.0))
}
heatmap_cmap = LinearSegmentedColormap('RedWhiteGreen', segmentdata=cdict, N=256)

RETURN_MIN = -5
RETURN_MAX = 5

# --- Helper Functions ---

def draw_calendar(ax, year, month, df_month, norm, cmap):
    cal = calendar.Calendar(firstweekday=0)  # Monday start
    month_days = cal.monthdayscalendar(year, month)  # Correct method here

    ax.cla()

    for week_idx, week in enumerate(month_days):
        for day_idx, day in enumerate(week):
            if day == 0:
                continue
            if day_idx >= 5:  # Skip weekends (Sat=5, Sun=6)
                continue

            date_str = f"{year}-{month:02d}-{day:02d}"
            if date_str in df_month.index:
                val = df_month.loc[date_str, 'Daily Return %']
                color = cmap(norm(val))
                label = f"{day}\n{val:.2f}%"
            else:
                color = 'white'
                label = ""

            rect = patches.Rectangle(
                (day_idx, -week_idx), 1, 1,
                linewidth=0.8, edgecolor='lightgray', facecolor=color
            )
            ax.add_patch(rect)
            ax.text(day_idx + 0.5, -week_idx + 0.6, label,
                    ha='center', va='top', fontsize=7)

    ax.set_xlim(0, 5)
    ax.set_ylim(-len(month_days), 0.5)
    ax.set_xticks(range(5))
    ax.set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri'])
    ax.tick_params(left=False, bottom=False)
    ax.set_yticks([])
    ax.set_title(f"{calendar.month_name[month]} {year}", fontsize=12, fontweight='bold')
    ax.axis('off')

# --- Main Execution ---
ticker = input("Enter ticker symbol: ").upper()
end_date = datetime.today()
start_date = end_date - timedelta(days=365)

data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)

if data.empty:
    print("No data fetched. Check the ticker symbol and try again.")
    exit()

# Flatten MultiIndex columns if present
if isinstance(data.columns, pd.MultiIndex):
    data.columns = [' '.join(map(str, col)).strip() for col in data.columns.values]

possible_close_cols = [col for col in data.columns if col.lower().startswith('close')]
if possible_close_cols:
    close_col = possible_close_cols[0]
else:
    print("Close price column not found after flattening.")
    exit()

data['Daily Return %'] = data[close_col].pct_change() * 100
data = data.dropna(subset=['Daily Return %'])

data['Date'] = data.index
if not pd.api.types.is_datetime64_any_dtype(data['Date']):
    data['Date'] = pd.to_datetime(data['Date'])

data['Date_str'] = data['Date'].dt.strftime('%Y-%m-%d')
data.set_index('Date_str', inplace=True)

months = sorted(set((d.year, d.month) for d in data['Date']))

n_cols = 4
n_rows = (len(months) + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5.5, n_rows * 5))
axes = axes.flatten()

norm = Normalize(vmin=RETURN_MIN, vmax=RETURN_MAX)
cmap = heatmap_cmap

for idx, (year, month) in enumerate(months):
    ax = axes[idx]
    df_month = data[(data['Date'].dt.year == year) & (data['Date'].dt.month == month)][['Daily Return %']]
    draw_calendar(ax, year, month, df_month, norm, cmap)

    month_weeks = len(calendar.Calendar(firstweekday=0).monthdayscalendar(year, month))
    avg_return = df_month['Daily Return %'].mean()
    txt = f"Avg Daily Return:\n{avg_return:.2f}%"
    ax.text(2.5, -month_weeks - 0.3, txt, ha='center', va='top', fontsize=8, color='black')

for j in range(idx + 1, len(axes)):
    axes[j].axis('off')

cbar_ax = fig.add_axes([0.2, 0.04, 0.6, 0.03])
cb = ColorbarBase(cbar_ax, cmap=cmap, norm=norm, orientation='horizontal')
cb.set_label('Daily Return %')

plt.subplots_adjust(left=0.05, right=0.95, top=0.93, bottom=0.10, hspace=0.8, wspace=0.25)

fig.suptitle(f"{ticker} Daily Return Calendar Heatmap\n({start_date.date()} to {end_date.date()})",
             fontsize=16, fontweight='bold')

output_path = os.path.join(SAVE_DIR, f"{ticker}_calendar_heatmap.png")
plt.savefig(output_path, dpi=300)
plt.close()

print(f"Saved heatmap to {output_path}")

