# Market dashboard

Every weekday evening, GitHub runs `build_dashboard.py`, checks the S&P 500 and your
watchlist against their 200-day averages, decides the market regime (bull / caution / bear)
and publishes a one-page dashboard at your GitHub Pages address.

## Setup (about 10 minutes, one time)
1. On github.com click **+ → New repository**. Name it `dashboard`. Choose **Public**
   (free GitHub Pages needs a public repo; the page shows market data only, nothing about your account).
2. Click **uploading an existing file** and drag in everything from this folder
   (including the hidden `.github` folder). Commit.
3. **Settings → Pages → Build and deployment → Source: GitHub Actions.**
4. **Actions** tab → **Nightly dashboard** → **Run workflow**. Wait about 3–6 minutes.
5. Your address: `https://YOUR-USERNAME.github.io/dashboard/`. Bookmark it.

## Change the watchlist
Edit `watchlist.csv` on github.com (pencil icon). Columns: symbol, theme, stop_pct, note.
The next run picks it up. Keep it to about 50 names.

## Change account size / rules
Account size is in `build_dashboard.py` (`ACCOUNT_SIZE`, used for the "max shares" column).
Thresholds (dip window, breadth levels, cash reserve) are at the top of the same file.
