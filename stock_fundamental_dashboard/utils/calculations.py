import numpy as np

# Example thresholds (customize as needed)
THRESHOLDS = {
    "PE": 15,
    "PB": 1.5,
    "ROE": 12,
    "NetMargin": 10,
    "Score_Green": 4,  # At least 4 criteria
    "Score_Blue": 2,
}

def calc_all_ratios(d):
    try:
        PE = d['Price'] / d['EPS'] if d.get('EPS', 0) else 0
        PB = d['Price'] / d['BVPS'] if d.get('BVPS', 0) else 0
        ROE = d.get('ROE', 0)
        NetMargin = d.get('NetMargin', 0)
        # Scoring
        score = 0
        if PE <= THRESHOLDS["PE"] and PE > 0: score += 1
        if PB <= THRESHOLDS["PB"] and PB > 0: score += 1
        if ROE >= THRESHOLDS["ROE"]: score += 1
        if NetMargin >= THRESHOLDS["NetMargin"]: score += 1
        # Color
        if score >= THRESHOLDS["Score_Green"]:
            color = "green"
        elif score >= THRESHOLDS["Score_Blue"]:
            color = "blue"
        else:
            color = "red"
        d.update({
            "PE": PE, "PB": PB, "Score": score, "ScoreColor": color
        })
        return d
    except Exception as e:
        print("Error:", e)
        return d

def get_radar_chart_data(row):
    metrics = ["PE", "PB", "ROE", "NetMargin"]
    values = [
        min(THRESHOLDS['PE'] / row['PE'], 1) if row['PE'] else 0,
        min(THRESHOLDS['PB'] / row['PB'], 1) if row['PB'] else 0,
        min(row['ROE'] / THRESHOLDS['ROE'], 1),
        min(row['NetMargin'] / THRESHOLDS['NetMargin'], 1)
    ]
    radar_data = {"metrics": metrics, "values": values}
    return radar_data, metrics

def get_historic_chart(row):
    import plotly.graph_objs as go
    # Demo: Plot TTM and previous quarter for revenue/net profit
    # (In practice, pass a real history array!)
    quarters = ['Q4', 'Q1', 'Q2', 'Q3', 'Q4']
    revenues = [row.get('Revenue', 0), row.get('TTM_Revenue', 0), row.get('Revenue', 0), row.get('TTM_Revenue', 0), row.get('Revenue', 0)]
    net_profits = [row.get('NetProfit', 0), row.get('TTM_NetProfit', 0), row.get('NetProfit', 0), row.get('TTM_NetProfit', 0), row.get('NetProfit', 0)]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=quarters, y=revenues, name='Revenue'))
    fig.add_trace(go.Bar(x=quarters, y=net_profits, name='Net Profit'))
    fig.update_layout(barmode='group', title="Demo: Revenue/Net Profit History")
    return fig
