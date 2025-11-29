mport os
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from sklearn.linear_model import LinearRegression
from scipy import stats

warnings.filterwarnings("ignore")

# -------------------------
# Load API key securely
# -------------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Small helper to stop early if key missing
if not GROQ_API_KEY:
    st.set_page_config(page_title="ReStockerAI - Local Preview", page_icon="📦", layout="wide")
    st.title("ReStockerAI - Preview (No GROQ key)")
    st.error("🚨 GROQ_API_KEY tidak ditemukan. Set environment variable GROQ_API_KEY di .env atau Streamlit Secrets.")
    st.stop()

# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="ReStockerAI", page_icon="🤖📦", layout="wide")
st.title("🤖 ReStockerAI – Predictive Supply Engine")
st.write("Upload data persediaan (mutasi atau snapshot). Dapatkan prediksi stok, rekomendasi restock, deteksi anomali, dan ringkasan AI.")

uploaded_file = st.file_uploader("📂 Upload file (Excel / CSV). Format transaksi: Item, Date, Change  — atau snapshot: Item, Stock", type=["xlsx", "csv"])

# Useful params on sidebar
st.sidebar.header("Pengaturan Prediksi")
forecast_horizon_days = st.sidebar.selectbox("Horizon prediksi (hari)", [7, 14, 30, 60], index=2)
lead_time_days = st.sidebar.number_input("Lead time (hari)", min_value=1, max_value=180, value=7)
safety_multiplier = st.sidebar.slider("Safety stock multiplier", 0.5, 3.0, 1.5)
zscore_threshold = st.sidebar.slider("Anomaly z-score threshold", 2.0, 4.0, 3.0)

if uploaded_file:
    # Read file
    try:
        if uploaded_file.name.lower().endswith(".csv"):
            df_raw = pd.read_csv(uploaded_file)
        else:
            df_raw = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Error membaca file: {e}")
        st.stop()

    st.subheader("📋 Data Mentah")
    st.dataframe(df_raw.head(200))

    # Detect format
    cols = [c.strip() for c in df_raw.columns]
    df = df_raw.copy()
    df.columns = cols

    # Standardize columns lower
    col_lc = [c.lower() for c in cols]

    if all(x in col_lc for x in ["item", "date", "change"]):
        mode = "transaction"
        df = df.rename(columns={cols[col_lc.index("item")]: "Item",
                                cols[col_lc.index("date")]: "Date",
                                cols[col_lc.index("change")]: "Change"})
        df["Date"] = pd.to_datetime(df["Date"])
        st.success("✅ Terdeteksi format TRANSACTION (Item, Date, Change).")
    elif all(x in col_lc for x in ["item", "stock"]):
        mode = "snapshot"
        df = df.rename(columns={cols[col_lc.index("item")]: "Item",
                                cols[col_lc.index("stock")]: "Stock"})
        st.success("✅ Terdeteksi format SNAPSHOT (Item, Stock).")
    else:
        st.error("⚠️ Kolom tidak sesuai. Gunakan format transaksi (Item, Date, Change) atau snapshot (Item, Stock).")
        st.stop()

    # -------------------------
    # PROCESSING TRANSACTION MODE
    # -------------------------
    if mode == "transaction":
        # Aggregate daily per item
        df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()
        # Compute cumulative stock per item over time (assume starting point zero unless snapshot provided)
        grouped = df.groupby(["Item", "Date"])["Change"].sum().reset_index()
        items = grouped["Item"].unique().tolist()

        # Build time series per item with continuous date index
        ts_list = []
        for item in items:
            g = grouped[grouped["Item"] == item].set_index("Date").sort_index()
            # reindex to full date range
            full_idx = pd.date_range(g.index.min(), g.index.max(), freq="D")
            g = g.reindex(full_idx, fill_value=0)
            g.index.name = "Date"
            g = g.rename_axis(None).reset_index()
            g["Item"] = item
            # cumulative stock (assume start at 0)
            g["Cumulative"] = g["Change"].cumsum()
            ts_list.append(g[["Item", "Date", "Change", "Cumulative"]])
        ts = pd.concat(ts_list, ignore_index=True)

        # Show example for first item
        st.subheader("📈 Contoh time series (item pertama)")
        first_item = items[0]
        st.write(f"Item: *{first_item}*")
        st.dataframe(ts[ts["Item"] == first_item].tail(30))

        # -------------------------
        # PREDICTION (per item) — LinearRegression on time index -> cumulative
        # -------------------------
        st.subheader("🔮 Prediksi Stok per Item (Linear Regression sederhana)")
        forecasts = []
        recommendations = []
        anomalies = []

        for item in items:
            item_df = ts[ts["Item"] == item].copy()
            # require at least 7 points
            if item_df.shape[0] < 7:
                continue

            # train on days -> cumulative
            item_df = item_df.sort_values("Date")
            item_df["day_idx"] = (item_df["Date"] - item_df["Date"].min()).dt.days
            X = item_df[["day_idx"]].values
            y = item_df["Cumulative"].values

            model = LinearRegression()
            model.fit(X, y)

            last_day = item_df["Date"].max()
            horizon = forecast_horizon_days
            future_days = np.arange(item_df["day_idx"].max() + 1, item_df["day_idx"].max() + 1 + horizon).reshape(-1, 1)
            y_pred = model.predict(future_days)

            # Prepare forecast df
            future_dates = [last_day + timedelta(days=int(i)) for i in range(1, horizon + 1)]
            fdf = pd.DataFrame({
                "Item": item,
                "Date": future_dates,
                "Predicted_Cumulative": y_pred
            })
            forecasts.append(fdf)

            # Usage estimate: average daily outflow (negative changes)
            neg_changes = item_df[item_df["Change"] < 0]["Change"].abs()
            avg_daily_usage = neg_changes.mean() if not neg_changes.empty else 0.0
            std_usage = neg_changes.std() if not neg_changes.empty else 0.0

            # Estimated current stock = last cumulative
            current_stock = float(item_df["Cumulative"].iloc[-1])

            # Safety stock & reorder point
            safety_stock = safety_multiplier * (std_usage * np.sqrt(lead_time_days) if std_usage > 0 else 0)
            reorder_point = (avg_daily_usage * lead_time_days) + safety_stock

            # Days until stockout (based on avg usage)
            days_until_stockout = (current_stock / avg_daily_usage) if avg_daily_usage > 0 else np.inf

            recommendations.append({
                "Item": item,
                "Current_Stock": current_stock,
                "Avg_Daily_Usage": avg_daily_usage,
                "Std_Usage": std_usage,
                "Safety_Stock": safety_stock,
                "Reorder_Point": reorder_point,
                "Days_Until_Stockout": days_until_stockout
            })

            # -------------------------
            # Anomaly detection on daily changes using z-score
            # -------------------------
            if item_df["Change"].abs().std() > 0:
                zscores = stats.zscore(item_df["Change"].fillna(0))
                anomalous_idxs = np.where(np.abs(zscores) > zscore_threshold)[0]
                for idx in anomalous_idxs:
                    anomalies.append({
                        "Item": item,
                        "Date": item_df.iloc[idx]["Date"],
                        "Change": float(item_df.iloc[idx]["Change"]),
                        "Zscore": float(zscores[idx])
                    })

        if forecasts:
            forecast_df = pd.concat(forecasts, ignore_index=True)
            st.write(f"Menampilkan prediksi untuk {len(forecasts)} item.")
            # Show forecasts for selected item
            sel_item = st.selectbox("Pilih item untuk lihat prediksi", items)
            sel_forecast = forecast_df[forecast_df["Item"] == sel_item]
            sel_history = ts[ts["Item"] == sel_item].set_index("Date")[["Cumulative"]].reset_index()
            fig = px.line(pd.concat([sel_history.rename(columns={"Cumulative": "Value"}), 
                                     sel_forecast.rename(columns={"Predicted_Cumulative":"Value"})], ignore_index=True),
                          x="Date", y="Value", color=sel_forecast["Item"].apply(lambda x: sel_item))
            fig.update_layout(title=f"Riwayat & Prediksi - {sel_item}", legend_title_text="")
            st.plotly_chart(fig, use_container_width=True)

        # Recommendations table
        rec_df = pd.DataFrame(recommendations).sort_values("Days_Until_Stockout")
        st.subheader("📌 Rekomendasi & Reorder Point")
        st.dataframe(rec_df)

        # Anomalies
        if anomalies:
            anom_df = pd.DataFrame(anomalies).sort_values(["Item", "Date"])
            st.subheader("⚠️ Anomali Detected (berdasarkan z-score)")
            st.dataframe(anom_df)
        else:
            st.info("Tidak ditemukan anomali berdasarkan threshold saat ini.")

        # -------------------------
        # AI SUMMARY (Groq)
        # -------------------------
        st.subheader("🤖 Ringkasan AI (Groq) - Insights & Rekomendasi")
        # Prepare concise summary to send to LLM (limit size)
        top_rec = rec_df.head(10).to_string(index=False)
        top_anom = anom_df.head(10).to_string(index=False) if anomalies else "Tidak ada anomali signifikan."

        system_prompt = "You are an AI inventory analyst. Provide concise, actionable insights and recommended next steps for inventory management based on the provided summary."
        user_prompt = f"""Summary (top recommendations):
{top_rec}

Top anomalies:
{top_anom}

Please give:
1) 3 key insights (short)
2) 3 action recommendations (prioritized)
3) Any caution/warning items
"""

        try:
            response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                model="llama-3.1-8b-instant",
            )
            ai_text = response.choices[0].message.content
            st.markdown(f"*AI Insight:*\n\n{ai_text}")
        except Exception as e:
            st.error(f"GROQ API error: {e}")

        # -------------------------
        # AI Chat interactive
        # -------------------------
        st.subheader("💬 Chat dengan AI tentang data persediaan")
        user_query = st.text_input("Tanyakan ke AI (mis. 'Apa item yang paling berisiko habis?'):")
        if user_query:
            try:
                # Build compact context
                context = f"Top recommendations:\n{top_rec}\nTop anomalies:\n{top_anom}\nUser question: {user_query}"
                chat_resp = client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": "You are an AI inventory analyst helping users interpret inventory analytics."},
                        {"role": "user", "content": context}
                    ],
                    model="llama-3.1-8b-instant",
                )
                st.write(chat_resp.choices[0].message.content)
            except Exception as e:
                st.error(f"GROQ API error: {e}")

    # -------------------------
    # PROCESSING SNAPSHOT MODE
    # -------------------------
    elif mode == "snapshot":
        st.subheader("📦 Analisis Snapshot Stok")
        snapshot = df[["Item", "Stock"]].copy()
        snapshot["Stock"] = pd.to_numeric(snapshot["Stock"], errors="coerce").fillna(0)
        st.dataframe(snapshot)

        # Basic metrics
        snapshot["Category"] = snapshot["Item"]
        total_items = snapshot.shape[0]
        total_stock = snapshot["Stock"].sum()
        st.write(f"Total item: *{total_items}, Total stock (sum): *{total_stock}**")

        # Visual: top 20 items by stock
        st.subheader("📊 Top 20 by Stock")
        top20 = snapshot.sort_values("Stock", ascending=False).head(20)
        fig = px.bar(top20, x="Item", y="Stock", title="Top 20 Items by Stock")
        st.plotly_chart(fig, use_container_width=True)

        # Simple reorder recommendation based on quantiles & lead time (heuristic)
        q75 = snapshot["Stock"].quantile(0.25)  # small stock threshold
        snapshot["Reorder_Recommended"] = snapshot["Stock"] < q75
        st.subheader("📌 Rekomendasi Reorder (heuristik quantile)")
        st.dataframe(snapshot.sort_values("Stock").head(50))

        # AI Summary
        st.subheader("🤖 Ringkasan AI (Groq) - Snapshot Insights")
        snap_summary = snapshot.describe().to_string()
        prompt = f"Snapshot summary:\n{snap_summary}\nTop 10 items:\n{top20.to_string(index=False)}\nProvide 3 quick insights and 3 prioritized recommendations."
        try:
            resp = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are an AI inventory analyst."},
                    {"role": "user", "content": prompt}
                ],
                model="llama-3.1-8b-instant",
            )
            st.write(resp.choices[0].message.content)
        except Exception as e:
            st.error(f"GROQ API error: {e}")

        st.subheader("💬 Chat dengan AI (Snapshot)")
        user_q = st.text_input("Tanyakan tentang snapshot:", key="snap_chat")
        if user_q:
            try:
                resp = client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": "You are an AI inventory analyst."},
                        {"role": "user", "content": f"Snapshot top20:\n{top20.to_string(index=False)}\nQuestion: {user_q}"}
                    ],
                    model="llama-3.1-8b-instant",
                )
                st.write(resp.choices[0].message.content)
