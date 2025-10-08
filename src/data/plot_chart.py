import matplotlib.pyplot as plt
from src.data.yfinance_data import YFinanceData

def plot_chart():
    data = YFinanceData("^N225")
    df = data.get_data()
    if df.empty:
        # データが空の場合はメッセージ付きの空画像を保存
        fig = plt.figure()
        plt.text(0.5, 0.5, "No data in range", ha="center", va="center")
        plt.axis('off')
        fig.savefig("data/chart.png")
        plt.close(fig)
        return
    df.plot()
    plt.savefig("data/chart.png")
    plt.close()

if __name__ == "__main__":
    plot_chart()